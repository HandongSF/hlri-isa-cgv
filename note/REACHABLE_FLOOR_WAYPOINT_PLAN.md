# Reachable Floor Waypoint Plan

## Goal

Prevent PointNav from chasing unreachable 3D targets created by projecting a
pixel on an object surface. The first implementation focuses on replacing
obvious object-surface targets before PointNav receives them.

Current flow:

```text
goal_mask -> pixel -> 3D point -> PointNav goal
```

Target flow:

```text
goal_mask -> anchor pixel -> raw 3D anchor -> reachable floor point in current view -> PointNav goal
```

## Core Idea

The pixel-derived 3D point is only an anchor. The actual PointNav target must be
a reachable floor point near that anchor.

## Core Algorithm

1. Extract an anchor pixel from `goal_mask`.
2. Convert that pixel into a raw 3D point with the existing depth backprojection.
3. Build a small local traversability map from the current depth.
4. Search below the anchor pixel for a reachable floor candidate in the current
   image.
5. If that search fails, search the whole current image for reachable floor
   candidates.
6. Choose the candidate whose 3D position is closest to the raw anchor.
7. Use that candidate as the final waypoint for `rho, theta`.

## Traversability Definition

The local map only needs to answer one question:

```text
Which nearby floor cells are reachable by the robot?
```

Minimal construction:

1. Back-project depth into 3D points.
2. Rasterize points into a local 2D grid.
3. Mark obstacle cells.
4. Mark free floor cells.
5. Inflate obstacles by robot radius.
6. Flood fill from the robot cell to get reachable free cells.

## Candidate Search

The search runs in two stages.

Stage 1:

```text
Start from the anchor pixel and move downward in image y.
Return the first pixel that is classified as reachable floor.
```

Stage 2:

```text
If Stage 1 fails, scan the whole current image for reachable floor pixels.
Choose the floor point whose 3D position is closest to raw_anchor_world.
```

If neither stage finds a reachable floor point, return an invalid waypoint so
the planner can replan.

## Required Code Changes

### 1. Geometry layer

Keep using:

```python
build_depth_waypoint_from_pixel(...)
```

Add:

```python
extract_anchor_pixel_from_mask(...)
```

### 2. New reachable-waypoint resolver

Add a new module:

```text
data_utils/reachable_waypoint.py
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
) -> DepthWaypoint:
    ...
```

This function:

1. Builds the local traversability map.
2. Searches below the anchor pixel for reachable floor.
3. Falls back to the closest reachable floor point in the current image.
4. Returns the resolved floor waypoint.

### 3. Benchmark integration

Current:

```python
build_depth_waypoint_from_pixel(...)
```

Target:

```python
raw_waypoint = build_depth_waypoint_from_pixel(...)
final_waypoint = resolve_reachable_floor_waypoint(...)
```

`compute_relative_pointgoal(...)` stays unchanged.

## Requirements

1. The final PointNav target is the resolved reachable floor point.
2. Traversability is computed from the current depth observation.
3. A missing reachable floor result produces an invalid waypoint and a
   replan request.

## First Success Criterion

The first implementation is successful when object-surface anchors are replaced
by reachable floor waypoints before PointNav receives the goal.
