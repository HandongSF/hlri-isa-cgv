# Reachable Floor Waypoint Implementation Plan

## Goal

Prevent Depth PointNav from chasing unreachable waypoint coordinates caused by
projecting a pixel on an object surface, furniture top, wall, or other
non-traversable geometry.

Current flow:

```text
goal_mask
-> centroid pixel
-> depth backprojection
-> world waypoint
-> PointNav rho/theta
```

Problem:

```text
If the pixel lies on an object surface, the generated waypoint can be inside or
on top of an object. PointNav then tries to reach an impossible coordinate and
may circle around it.
```

New flow:

```text
goal_mask
-> raw visual anchor pixel
-> raw visual anchor 3D point
-> nearest reachable floor waypoint near that anchor
-> PointNav rho/theta
```

The raw pixel-derived point is treated as an object/visual anchor. It is not
used directly as a navigation target unless it is verified as traversable floor.

## Design Requirements

1. The primary algorithm must be compatible with real robot navigation.
2. Habitat `pathfinder` may be used as an optional simulator validation oracle,
   not as the only source of navigability.
3. The final PointNav target must be a reachable floor point, not an object
   surface point.
4. The implementation must preserve the existing PointNav contract:

   ```python
   {
       "depth": depth_tensor,
       "pointgoal_with_gps_compass": [[rho, theta]],
   }
   ```

5. If no reachable floor waypoint is found, do not fall back to the raw object
   point by default. Return an invalid waypoint and trigger replan/rotate.

## Key Concept

There are two different points:

```text
raw_anchor_world
  The 3D point obtained from the selected pixel and depth.
  It may lie on an object and may be unreachable.

reachable_floor_world
  The navigation target selected near raw_anchor_world.
  It must lie on locally traversable floor and be reachable from the agent.
```

`DepthWaypoint.world_position` should eventually store `reachable_floor_world`.
For debugging, keep `raw_anchor_world` separately.

## Proposed Data Model

Extend the waypoint model or add a new dataclass:

```python
@dataclass
class DepthWaypoint:
    pixel_u: int
    pixel_v: int
    initial_depth: Optional[float]
    world_position: Optional[np.ndarray]      # final reachable floor target
    valid: bool
    failure_reason: Optional[str] = None
    raw_world_position: Optional[np.ndarray] = None
    target_kind: str = "raw_depth"
    clearance: Optional[float] = None
    floor_score: Optional[float] = None
```

`target_kind` values:

```text
raw_depth
reachable_floor
depth_traversability
habitat_navmesh_checked
```

## Algorithm Overview

### 1. Extract Visual Anchor Pixel

Keep the current centroid extractor as a baseline:

```python
extract_waypoint_pixel_from_mask(goal_mask)
```

Add a more object-aware extractor later:

```python
extract_anchor_pixel_from_mask(goal_mask, mode="bottom_center")
```

Recommended first order:

```text
1. bottom-center of object mask
2. centroid fallback
```

Reason:

Object centroids often lie on the object body. Bottom-center is closer to the
object-floor contact region and usually produces better local search centers.

### 2. Convert Anchor Pixel to Raw 3D Point

Reuse the current geometry path:

```python
raw_waypoint = build_depth_waypoint_from_pixel(...)
raw_anchor_world = raw_waypoint.world_position
```

This result is only the search anchor. It should not be considered reachable
until verified.

### 3. Build Local Traversability From Depth

For real robot compatibility, generate local traversability from the current
depth image.

Inputs:

```python
depth_metric: np.ndarray
camera_intrinsics: np.ndarray
camera_position: np.ndarray
camera_rotation: np.ndarray
agent_xy: np.ndarray
```

Process:

```text
1. Back-project depth pixels into a local/world point cloud.
2. Estimate the local floor height from low points near the robot.
3. Rasterize points into a local 2D grid around the robot.
4. Mark cells as obstacle if points occupy robot body height.
5. Mark cells as floor candidates if height is close to floor height and local
   slope/height variance is small.
6. Inflate obstacles by robot radius plus safety margin.
7. Flood fill from the robot cell to identify reachable free cells.
```

Initial config:

```python
grid_resolution_m = 0.05
grid_radius_m = 3.0
robot_radius_m = 0.18
safety_margin_m = 0.10
floor_height_tolerance_m = 0.15
obstacle_min_height_m = 0.12
obstacle_max_height_m = 1.20
min_depth_m = 0.5
max_depth_m = 5.0
```

Notes:

- The exact robot radius should become a config value.
- For Habitat ObjectNav evaluation, these defaults are enough for a first pass.
- On a physical robot, the floor height should come from calibrated camera
  extrinsics, IMU gravity alignment, or a robust ground plane estimate.

### 4. Sample Candidate Floor Targets Near Anchor

Use `raw_anchor_world[:2]` as the object/visual anchor.

Generate candidates in a local ring around the anchor:

```text
radius: 0.35m to 1.50m
step: 0.15m
angles: full circle or biased toward agent-facing side
```

Recommended first pass:

```python
standoff_distances = [0.45, 0.65, 0.85, 1.05, 1.25]
angle_offsets_deg = [0, -30, 30, -60, 60, -90, 90, 180]
```

The base direction should point from the object anchor toward the agent:

```python
base_dir = normalize(agent_xy - raw_anchor_xy)
candidate_xy = raw_anchor_xy + rotate(base_dir, angle_offset) * standoff
```

This makes the first candidates lie between the robot and the object, which is
often the most visible and reachable region.

### 5. Validate Candidates Against Local Traversability

Candidate checks:

```text
1. Candidate lies inside local grid.
2. Candidate cell is traversable floor.
3. Candidate cell remains free after obstacle inflation.
4. Candidate is connected to the robot cell by flood fill.
5. Candidate has enough clearance from obstacles.
```

Do not compare candidate height directly to `raw_anchor_world[2]` unless the raw
anchor pixel is already classified as floor. Object pixels can be much higher
than the floor.

Preferred height check:

```text
candidate floor height ~= estimated local floor height
```

### 6. Score Candidates

Select the best candidate with a simple weighted score:

```python
score = (
    1.0 * distance(candidate_xy, raw_anchor_xy)
    + 0.5 * path_distance_from_agent
    + 0.4 * obstacle_clearance_penalty
    + 0.2 * abs(standoff - preferred_standoff)
)
```

Lower score is better.

Recommended defaults:

```python
preferred_standoff_m = 0.85
min_clearance_m = robot_radius_m + safety_margin_m
```

Tie breaker:

```text
Prefer candidates with smaller heading change from current robot heading.
```

### 7. Optional Habitat Navmesh Check

In simulation, Habitat `pathfinder` may be used only as a post-check after
local depth validation. Do not implement a Habitat-first shortcut.

Optional check:

```text
1. Convert candidate xy to Habitat-compatible 3D point.
2. pathfinder.snap_point(candidate)
3. Reject if snap distance is too large.
4. Reject if geodesic path from agent to snapped point is invalid.
5. Reject if the snapped point changes the selected waypoint too much.
```

Recommended thresholds:

```python
max_navmesh_snap_m = 0.5
max_geodesic_to_euclidean_ratio = 3.0
```

Reason:

This check can catch simulator-specific geometry mismatches, but the selected
waypoint must still come from depth-based traversability.

## Proposed API

Add to `data_utils/depth_pointnav_geometry.py` or a new file
`data_utils/reachable_waypoint.py`.

Recommended new file:

```text
data_utils/reachable_waypoint.py
```

Core API:

```python
@dataclass
class ReachableWaypointConfig:
    grid_resolution_m: float = 0.05
    grid_radius_m: float = 3.0
    robot_radius_m: float = 0.18
    safety_margin_m: float = 0.10
    floor_height_tolerance_m: float = 0.15
    obstacle_min_height_m: float = 0.12
    obstacle_max_height_m: float = 1.20
    preferred_standoff_m: float = 0.85
    standoff_distances_m: tuple[float, ...] = (0.45, 0.65, 0.85, 1.05, 1.25)
    angle_offsets_deg: tuple[float, ...] = (0, -30, 30, -60, 60, -90, 90, 180)
    use_habitat_navmesh_check: bool = False
    max_navmesh_snap_m: float = 0.5
```

Main function:

```python
def resolve_reachable_floor_waypoint(
    raw_waypoint: DepthWaypoint,
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    agent_xy: np.ndarray,
    heading: float,
    cfg: ReachableWaypointConfig,
    habitat_pathfinder: Optional[object] = None,
) -> DepthWaypoint:
    ...
```

Return behavior:

```text
valid=True
  world_position is the reachable floor target.
  raw_world_position stores the original depth-projected point.
  target_kind is "depth_traversability" or "habitat_navmesh_checked".

valid=False
  world_position is None.
  raw_world_position stores the original depth-projected point if available.
  failure_reason describes the failure.
```

Failure reasons:

```text
raw_waypoint_invalid
no_floor_points
no_reachable_cells
no_candidate_in_grid
no_reachable_candidate
navmesh_check_failed
```

## Integration Point

Current code in `objnav_benchmark.py`:

```python
def build_current_depth_waypoint(obs, goal_mask):
    waypoint_pixel = extract_waypoint_pixel_from_mask(goal_mask)
    if waypoint_pixel is None:
        return None
    _, heading, camera_position, camera_rotation = get_vlfm_pose_from_obs(obs)
    depth_metric = restore_metric_depth_from_habitat(obs["depth"], min_depth, max_depth)
    return build_depth_waypoint_from_pixel(...)
```

Target flow:

```python
def build_current_depth_waypoint(obs, goal_mask):
    waypoint_pixel = extract_anchor_pixel_from_mask(goal_mask, mode="bottom_center")
    if waypoint_pixel is None:
        return None

    agent_xy, heading, camera_position, camera_rotation = get_vlfm_pose_from_obs(obs)
    depth_metric = restore_metric_depth_from_habitat(obs["depth"], min_depth, max_depth)

    raw_waypoint = build_depth_waypoint_from_pixel(...)
    if raw_waypoint is None or not raw_waypoint.valid:
        return raw_waypoint

    return resolve_reachable_floor_waypoint(
        raw_waypoint=raw_waypoint,
        depth_metric=depth_metric,
        camera_intrinsics=camera_intrinsics,
        camera_position=camera_position,
        camera_rotation=camera_rotation,
        agent_xy=agent_xy,
        heading=heading,
        cfg=reachable_waypoint_cfg,
        habitat_pathfinder=None,  # optional post-check only
    )
```

`compute_relative_pointgoal(...)` remains unchanged. It should receive the final
reachable waypoint position.

## Logging And Debugging

Log these fields for each waypoint:

```text
anchor_pixel
raw_anchor_world
reachable_floor_world
target_kind
initial_depth
failure_reason
candidate_count
reachable_candidate_count
selected_score
rho
theta
```

Optional debug visualization:

```text
1. RGB frame with anchor pixel and selected floor pixel/target projection.
2. Local traversability grid with obstacle/free/reachable/candidate/selected cells.
3. Topdown Habitat map with raw anchor and final reachable waypoint.
```

## Progress Watchdog

Even with better waypoint selection, keep a runtime failure detector.

Trigger replan when:

```text
1. rho does not decrease by at least 0.15m over 8-12 PointNav steps.
2. repeated collisions occur near the same waypoint.
3. PointNav emits STOP before object detection success and rho is still large.
```

Recommended state:

```python
last_rho_values: deque[float]
collision_count_for_waypoint: int
failed_waypoint_cache: list[np.ndarray]
```

If a waypoint fails, avoid selecting nearly the same final target again in the
same local planning cycle.

## Test Plan

### Unit Tests

1. Empty mask returns no waypoint.
2. Object centroid on a high obstacle does not become the final target.
3. A reachable free-floor candidate near the anchor is selected.
4. Candidate selection rejects inflated obstacle cells.
5. Candidate selection rejects cells not connected to the robot cell.
6. If no candidate is valid, function returns `valid=False`.
7. Center floor pixel still produces a forward `theta` close to zero.

### Simulation Checks

1. For visible objects on furniture, final waypoint lies on nearby floor.
2. During forward motion, `rho` generally decreases.
3. Agent does not orbit around object-centered raw anchors.
4. Replan triggers when no reachable floor target is found.
5. Optional navmesh check rejects candidates that snap too far.

### Real-Robot Readiness Checks

1. No required dependency on Habitat pathfinder.
2. Traversability uses only RGB-D/depth, intrinsics, and camera pose.
3. Robot radius and safety margin are configurable.
4. Floor estimation does not assume a globally flat Habitat navmesh.

## Implementation Phases

### Phase 1: Depth-Based Traversability

1. Add depth point cloud backprojection.
2. Build local occupancy/elevation grid.
3. Inflate obstacles.
4. Flood fill reachable free cells.
5. Use this grid for candidate validation.
6. Store raw and final waypoint fields for debugging.
7. Add progress watchdog.

This is the first implementation target. Do not add a Habitat navmesh-only
candidate selector.

### Phase 2: Simulator Post-Check

1. Use depth traversability first.
2. Use Habitat pathfinder only as optional rejection/post-check during Habitat
   runs.
3. Compare metrics with and without navmesh checking.
4. Keep failure/replan behavior identical across both modes.

## Open Decisions

1. Exact robot radius and safety margin for the target platform.
2. Whether to use only depth, or also YOLOE/SAM floor masks when available.
3. Whether `DepthWaypoint` should be extended in place or replaced by a new
   `ResolvedDepthWaypoint` dataclass.
4. Whether the local grid should be maintained across frames or rebuilt from
   only the current frame for the first implementation.

## Recommended Next Step

Implement the depth-based traversability path first. Keep Habitat pathfinder out
of the target selection loop unless it is explicitly enabled as a post-check.
