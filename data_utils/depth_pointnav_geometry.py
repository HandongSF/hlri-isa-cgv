from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DepthWaypoint:
    pixel_u: int
    pixel_v: int
    initial_depth: Optional[float]
    world_position: Optional[np.ndarray]
    valid: bool
    failure_reason: Optional[str] = None
    raw_world_position: Optional[np.ndarray] = None
    target_kind: str = "raw_depth"


@dataclass
class PointGoal:
    rho: float
    theta: float


def extract_waypoint_pixel_from_mask(goal_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if goal_mask is None:
        return None
    mask = np.asarray(goal_mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def extract_anchor_pixel_from_mask(goal_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if goal_mask is None:
        return None
    mask = np.asarray(goal_mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    bottom_y = int(ys.max())
    band_height = max(2, int(round(mask.shape[0] * 0.03)))
    bottom_band = ys >= bottom_y - band_height
    if np.any(bottom_band):
        return int(round(float(xs[bottom_band].mean()))), int(round(float(ys[bottom_band].mean())))
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.float32, copy=False)


def restore_metric_depth_from_habitat(
    depth_norm: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    depth = _squeeze_depth(depth_norm)
    return depth * float(max_depth - min_depth) + float(min_depth)


def lookup_valid_depth(
    depth_metric: np.ndarray,
    pixel: Tuple[int, int],
    min_depth: float,
    max_depth: float,
    window: int = 5,
) -> Optional[float]:
    depth = _squeeze_depth(depth_metric)
    u, v = pixel
    height, width = depth.shape
    if u < 0 or u >= width or v < 0 or v >= height:
        return None

    def _valid(values: np.ndarray) -> np.ndarray:
        return values[np.isfinite(values) & (values > min_depth) & (values < max_depth)]

    center = depth[v, u]
    if np.isfinite(center) and min_depth < float(center) < max_depth:
        return float(center)

    radius = max(0, int(window) // 2)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    valid = _valid(depth[y0:y1, x0:x1].reshape(-1))
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _pixel_to_camera_point(
    pixel: Tuple[int, int],
    depth_value: float,
    camera_intrinsics: np.ndarray,
    image_height: int,
) -> np.ndarray:
    u, v = pixel
    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])

    x_image = (float(u) - cx) * depth_value / fx
    y_image = (float(v) - cy) * depth_value / fy
    return np.array([depth_value, -x_image, -y_image], dtype=np.float32)


def pixel_to_world_point(
    pixel: Tuple[int, int],
    depth_value: float,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    image_height: int,
) -> np.ndarray:
    point_camera = _pixel_to_camera_point(pixel, depth_value, camera_intrinsics, image_height)
    rotation = np.asarray(camera_rotation, dtype=np.float32)
    position = np.asarray(camera_position, dtype=np.float32)
    return (rotation @ point_camera + position).astype(np.float32)


def build_depth_waypoint_from_pixel(
    pixel: Tuple[int, int],
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> DepthWaypoint:
    depth = _squeeze_depth(depth_metric)
    u, v = pixel
    depth_value = lookup_valid_depth(depth, pixel, min_depth, max_depth)
    if depth_value is None:
        height, width = depth.shape
        if u < 0 or u >= width or v < 0 or v >= height:
            return DepthWaypoint(u, v, None, None, False, "pixel_out_of_bounds")
        depth_value = float(max_depth)
        target_kind = "max_depth_fallback"
    else:
        target_kind = "raw_depth"

    point_world = pixel_to_world_point(
        pixel,
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
        point_world.astype(np.float32),
        True,
        target_kind=target_kind,
    )


def rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )


def compute_relative_pointgoal(
    waypoint_world: np.ndarray,
    current_agent_xy: np.ndarray,
    current_heading: float,
) -> PointGoal:
    goal = np.asarray(waypoint_world[:2], dtype=np.float32)
    agent_xy = np.asarray(current_agent_xy, dtype=np.float32)
    local_goal = rotation_matrix(-float(current_heading)) @ (goal - agent_xy)
    rho = float(np.linalg.norm(local_goal))
    theta = float(np.arctan2(local_goal[1], local_goal[0]))
    return PointGoal(rho=rho, theta=theta)
