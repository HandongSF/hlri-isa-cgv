# Offset Waypoint Plan

## Goal

Prevent PointNav from using the raw pixel-projected object surface as the final
navigation target.

Current flow:

```text
goal_mask -> anchor pixel -> raw 3D anchor -> PointNav goal
```

Target flow:

```text
goal_mask -> anchor pixel -> raw 3D anchor -> 0.1m offset waypoint -> PointNav goal
```

## Core Idea

The pixel-derived 3D point is only an anchor. The PointNav target is the anchor
shifted 0.1m toward the agent.

```text
waypoint_xy = raw_anchor_xy + normalize(agent_xy - raw_anchor_xy) * 0.1m
```

This keeps the target slightly in front of the object surface while preserving
the original waypoint direction.

## Core Algorithm

1. Extract an anchor pixel from `goal_mask`.
2. Convert that pixel into a raw 3D anchor with the existing depth
   backprojection.
3. Compute the 2D direction from the raw anchor to the agent.
4. Move the anchor 0.1m along that direction.
5. Use the offset point as the final waypoint for `rho, theta`.

If the anchor-to-agent direction cannot be computed, return an invalid waypoint
so the planner can replan.

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

### 2. Offset waypoint resolver

Use:

```text
voca/navigation/waypoint/reachable.py
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

This function returns `target_kind="offset_from_anchor"` when the offset is
successfully created.

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
