from typing import Optional

import numpy as np

from data_utils.depth_pointnav_geometry import DepthWaypoint


OFFSET_FALLBACK_M = 0.10


def _offset_from_anchor(raw_anchor_world: np.ndarray, agent_xy: np.ndarray) -> Optional[np.ndarray]:
    raw_xy = np.asarray(raw_anchor_world[:2], dtype=np.float32)
    agent_xy = np.asarray(agent_xy, dtype=np.float32)
    direction = agent_xy - raw_xy
    norm = float(np.linalg.norm(direction))
    if norm < 1e-4:
        return None

    waypoint_world = np.asarray(raw_anchor_world, dtype=np.float32).copy()
    waypoint_world[:2] = raw_xy + direction / norm * OFFSET_FALLBACK_M
    return waypoint_world


def resolve_reachable_floor_waypoint(
    raw_waypoint: DepthWaypoint,
    depth_metric: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    agent_xy: np.ndarray,
) -> DepthWaypoint:
    del depth_metric, camera_intrinsics, camera_position, camera_rotation

    if raw_waypoint is None or not raw_waypoint.valid or raw_waypoint.world_position is None:
        return raw_waypoint

    waypoint_world = _offset_from_anchor(raw_waypoint.world_position, agent_xy)
    if waypoint_world is None:
        return DepthWaypoint(
            raw_waypoint.pixel_u,
            raw_waypoint.pixel_v,
            raw_waypoint.initial_depth,
            None,
            False,
            "no_offset_direction",
            raw_world_position=raw_waypoint.world_position,
            target_kind="unresolved",
        )

    return DepthWaypoint(
        raw_waypoint.pixel_u,
        raw_waypoint.pixel_v,
        raw_waypoint.initial_depth,
        waypoint_world,
        True,
        raw_world_position=raw_waypoint.world_position,
        target_kind="offset_from_anchor",
    )
