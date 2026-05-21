from collections import deque
from typing import Optional, Tuple

import numpy as np

from data_utils.depth_pointnav_geometry import DepthWaypoint, pixel_to_world_point


GRID_RESOLUTION_M = 0.05
GRID_RADIUS_M = 6.0
ROBOT_RADIUS_M = 0.20
FLOOD_FILL_START_RADIUS_M = 1.50
FLOOR_HEIGHT_TOLERANCE_M = 0.15
OBSTACLE_MIN_HEIGHT_M = 0.12
OBSTACLE_MAX_HEIGHT_M = 1.20
ANCHOR_COLUMN_RADIUS_PX = 8
OFFSET_FALLBACK_M = 0.10


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def _backproject_depth(
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = _squeeze_depth(depth_metric)
    height, width = depth.shape
    ys, xs = np.indices((height, width))
    valid = np.isfinite(depth) & (depth > 0.0)

    valid_xs = xs[valid].astype(np.float32)
    valid_ys = ys[valid].astype(np.float32)
    depth_values = depth[valid].astype(np.float32)

    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])

    x_image = (valid_xs - cx) * depth_values / fx
    y_image = (valid_ys - cy) * depth_values / fy
    points_camera = np.stack([depth_values, -x_image, -y_image], axis=-1)

    rotation = np.asarray(camera_rotation, dtype=np.float32)
    position = np.asarray(camera_position, dtype=np.float32)
    points_world = points_camera @ rotation.T + position

    return depth, valid, points_world.astype(np.float32)


def _estimate_floor_height(points_world: np.ndarray, camera_z: float) -> Optional[float]:
    if points_world.size == 0:
        return None
    below_camera = points_world[:, 2] < float(camera_z) - 0.1
    candidates = points_world[below_camera, 2]
    if candidates.size < 20:
        candidates = points_world[:, 2]
    if candidates.size == 0:
        return None
    return float(np.percentile(candidates, 10))


def _grid_indices(points_xy: np.ndarray, agent_xy: np.ndarray, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = grid_size // 2
    rel = (points_xy - np.asarray(agent_xy, dtype=np.float32)) / GRID_RESOLUTION_M
    gx = np.rint(rel[:, 0]).astype(np.int32) + center
    gy = np.rint(rel[:, 1]).astype(np.int32) + center
    inside = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
    return gx, gy, inside


def _inflate_obstacles(obstacle: np.ndarray) -> np.ndarray:
    radius_cells = int(np.ceil(ROBOT_RADIUS_M / GRID_RESOLUTION_M))
    if radius_cells <= 0 or not np.any(obstacle):
        return obstacle.copy()

    inflated = obstacle.copy()
    ys, xs = np.where(obstacle)
    height, width = obstacle.shape
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            shifted_y = ys + dy
            shifted_x = xs + dx
            inside = (shifted_y >= 0) & (shifted_y < height) & (shifted_x >= 0) & (shifted_x < width)
            inflated[shifted_y[inside], shifted_x[inside]] = True
    return inflated


def _find_flood_fill_start(free: np.ndarray) -> Optional[Tuple[int, int]]:
    center_y = free.shape[0] // 2
    center_x = free.shape[1] // 2
    if free[center_y, center_x]:
        return center_y, center_x

    radius_cells = int(np.ceil(FLOOD_FILL_START_RADIUS_M / GRID_RESOLUTION_M))
    y0 = max(0, center_y - radius_cells)
    y1 = min(free.shape[0], center_y + radius_cells + 1)
    x0 = max(0, center_x - radius_cells)
    x1 = min(free.shape[1], center_x + radius_cells + 1)
    ys, xs = np.where(free[y0:y1, x0:x1])
    if len(xs) == 0:
        return None

    abs_ys = ys + y0
    abs_xs = xs + x0
    distances = (abs_ys - center_y) ** 2 + (abs_xs - center_x) ** 2
    best = int(np.argmin(distances))
    return int(abs_ys[best]), int(abs_xs[best])


def _reachable_cells(free: np.ndarray) -> np.ndarray:
    reachable = np.zeros_like(free, dtype=bool)
    start = _find_flood_fill_start(free)
    if start is None:
        return reachable

    queue: deque[Tuple[int, int]] = deque([start])
    reachable[start] = True
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= free.shape[0] or nx < 0 or nx >= free.shape[1]:
                continue
            if reachable[ny, nx] or not free[ny, nx]:
                continue
            reachable[ny, nx] = True
            queue.append((ny, nx))
    return reachable


def _build_reachable_floor_pixels(
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    agent_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    depth, valid, points_world = _backproject_depth(
        depth_metric,
        camera_intrinsics,
        camera_position,
        camera_rotation,
    )
    reachable_floor_pixels = np.zeros(depth.shape, dtype=bool)
    world_by_pixel = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    if points_world.size == 0:
        return reachable_floor_pixels, world_by_pixel

    floor_height = _estimate_floor_height(points_world, float(camera_position[2]))
    if floor_height is None:
        return reachable_floor_pixels, world_by_pixel

    grid_size = int(np.ceil((GRID_RADIUS_M * 2.0) / GRID_RESOLUTION_M)) + 1
    floor_cells = np.zeros((grid_size, grid_size), dtype=bool)
    obstacle_cells = np.zeros((grid_size, grid_size), dtype=bool)

    points_xy = points_world[:, :2]
    gx, gy, inside = _grid_indices(points_xy, agent_xy, grid_size)
    heights = points_world[:, 2] - floor_height
    floor_points = np.abs(heights) <= FLOOR_HEIGHT_TOLERANCE_M
    obstacle_points = (heights > OBSTACLE_MIN_HEIGHT_M) & (heights < OBSTACLE_MAX_HEIGHT_M)

    floor_inside = inside & floor_points
    obstacle_inside = inside & obstacle_points
    floor_cells[gy[floor_inside], gx[floor_inside]] = True
    obstacle_cells[gy[obstacle_inside], gx[obstacle_inside]] = True

    free_cells = floor_cells & ~_inflate_obstacles(obstacle_cells)
    reachable = _reachable_cells(free_cells)

    valid_indices = np.where(valid)
    valid_ys = valid_indices[0]
    valid_xs = valid_indices[1]
    world_by_pixel[valid_ys, valid_xs] = points_world

    pixel_inside = inside & floor_points
    pixel_reachable = np.zeros(points_world.shape[0], dtype=bool)
    pixel_reachable[pixel_inside] = reachable[gy[pixel_inside], gx[pixel_inside]]
    reachable_floor_pixels[valid_ys[pixel_reachable], valid_xs[pixel_reachable]] = True

    return reachable_floor_pixels, world_by_pixel


def _search_below_anchor(reachable_floor_pixels: np.ndarray, anchor_pixel: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    u, v = anchor_pixel
    height, width = reachable_floor_pixels.shape
    if u < 0 or u >= width or v < 0 or v >= height:
        return None

    x0 = max(0, u - ANCHOR_COLUMN_RADIUS_PX)
    x1 = min(width, u + ANCHOR_COLUMN_RADIUS_PX + 1)
    for y in range(v + 1, height):
        xs = np.where(reachable_floor_pixels[y, x0:x1])[0]
        if len(xs) == 0:
            continue
        abs_xs = xs + x0
        best_x = int(abs_xs[np.argmin(np.abs(abs_xs - u))])
        return best_x, int(y)
    return None


def _search_closest_floor(
    reachable_floor_pixels: np.ndarray,
    world_by_pixel: np.ndarray,
    raw_anchor_world: np.ndarray,
) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(reachable_floor_pixels)
    if len(xs) == 0:
        return None

    points = world_by_pixel[ys, xs]
    valid = np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return None

    valid_points = points[valid]
    valid_xs = xs[valid]
    valid_ys = ys[valid]
    distances = np.linalg.norm(valid_points[:, :2] - np.asarray(raw_anchor_world[:2], dtype=np.float32), axis=1)
    best = int(np.argmin(distances))
    return int(valid_xs[best]), int(valid_ys[best])


def _offset_from_anchor(raw_anchor_world: np.ndarray, agent_xy: np.ndarray) -> Optional[np.ndarray]:
    raw_xy = np.asarray(raw_anchor_world[:2], dtype=np.float32)
    agent_xy = np.asarray(agent_xy, dtype=np.float32)
    direction = agent_xy - raw_xy
    norm = float(np.linalg.norm(direction))
    if norm < 1e-4:
        return None

    fallback = np.asarray(raw_anchor_world, dtype=np.float32).copy()
    fallback[:2] = raw_xy + direction / norm * OFFSET_FALLBACK_M
    return fallback


def resolve_reachable_floor_waypoint(
    raw_waypoint: DepthWaypoint,
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    agent_xy: np.ndarray,
) -> DepthWaypoint:
    if raw_waypoint is None or not raw_waypoint.valid or raw_waypoint.world_position is None:
        return raw_waypoint

    depth = _squeeze_depth(depth_metric)
    reachable_floor_pixels, world_by_pixel = _build_reachable_floor_pixels(
        depth,
        camera_intrinsics,
        camera_position,
        camera_rotation,
        agent_xy,
    )
    anchor_pixel = (raw_waypoint.pixel_u, raw_waypoint.pixel_v)
    resolved_pixel = _search_below_anchor(reachable_floor_pixels, anchor_pixel)
    target_kind = "reachable_floor_below_anchor"
    if resolved_pixel is None:
        resolved_pixel = _search_closest_floor(
            reachable_floor_pixels,
            world_by_pixel,
            raw_waypoint.world_position,
        )
        target_kind = "reachable_floor_current_view"

    if resolved_pixel is None:
        fallback_world = _offset_from_anchor(raw_waypoint.world_position, agent_xy)
        if fallback_world is None:
            return DepthWaypoint(
                raw_waypoint.pixel_u,
                raw_waypoint.pixel_v,
                raw_waypoint.initial_depth,
                None,
                False,
                "no_reachable_floor",
                raw_world_position=raw_waypoint.world_position,
                target_kind="unresolved",
            )
        return DepthWaypoint(
            raw_waypoint.pixel_u,
            raw_waypoint.pixel_v,
            raw_waypoint.initial_depth,
            fallback_world,
            True,
            raw_world_position=raw_waypoint.world_position,
            target_kind="offset_from_anchor",
        )

    u, v = resolved_pixel
    depth_value = float(depth[v, u])
    point_world = pixel_to_world_point(
        resolved_pixel,
        depth_value,
        camera_intrinsics,
        camera_position,
        camera_rotation,
        depth.shape[0],
    )
    return DepthWaypoint(
        u,
        v,
        depth_value,
        point_world,
        True,
        raw_world_position=raw_waypoint.world_position,
        target_kind=target_kind,
    )
