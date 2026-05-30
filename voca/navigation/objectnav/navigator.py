from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from settings import DEFAULT_DEVICE, POINTNAV_CHECKPOINT
from voca.navigation.pointnav import DepthPointNavConfig, DepthPointNavController
from voca.navigation.pointnav.geometry import (
    DepthWaypoint,
    build_depth_waypoint_from_pixel,
    compute_relative_pointgoal,
    extract_anchor_pixel_from_mask,
    restore_metric_depth_from_habitat,
)
from voca.navigation.waypoint import resolve_reachable_floor_waypoint
from voca.planning import VLMPlanner


@dataclass
class VOCANavigatorConfig:
    pointnav_policy_path: str = POINTNAV_CHECKPOINT
    device: str = DEFAULT_DEVICE
    camera_intrinsics: Optional[np.ndarray] = None
    min_depth: float = 0.5
    max_depth: float = 5.0
    camera_height: float = 0.88


@dataclass
class VOCANavigatorAction:
    action: int
    request_replan: bool
    waypoint: Optional[DepthWaypoint]


class VOCANavigator:
    def __init__(self, yoloe_model, cfg: VOCANavigatorConfig):
        if cfg.camera_intrinsics is None:
            raise ValueError("camera_intrinsics is required")
        self.cfg = cfg
        self.camera_intrinsics = cfg.camera_intrinsics
        self.min_depth = float(cfg.min_depth)
        self.max_depth = float(cfg.max_depth)
        self.camera_height = float(cfg.camera_height)

        try:
            self.planner = VLMPlanner(yoloe_model)
        except TypeError:
            self.planner = VLMPlanner(yoloe_model, yoloe_model)

        self.controller = DepthPointNavController(
            DepthPointNavConfig(
                pointnav_policy_path=cfg.pointnav_policy_path,
                device=cfg.device,
            )
        )
        self.current_waypoint: Optional[DepthWaypoint] = None
        self.goal_flag = False
        self.pending_verify = False

    def reset(self, object_goal: str) -> None:
        self.planner.reset(object_goal)
        self.controller.reset()
        self.current_waypoint = None
        self.goal_flag = False
        self.pending_verify = False

    @staticmethod
    def yaw_to_rotation(yaw: float) -> np.ndarray:
        c = np.cos(float(yaw))
        s = np.sin(float(yaw))
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def get_vlfm_pose_from_obs(self, obs) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        if "gps" not in obs or "compass" not in obs:
            raise KeyError("depth_pointnav requires Habitat gps and compass observations")
        gps = np.asarray(obs["gps"], dtype=np.float32).reshape(-1)
        heading = float(np.asarray(obs["compass"]).reshape(-1)[0])
        agent_xy = np.array([gps[0], -gps[1]], dtype=np.float32)
        camera_position = np.array([agent_xy[0], agent_xy[1], self.camera_height], dtype=np.float32)
        return agent_xy, heading, camera_position, self.yaw_to_rotation(heading)

    def build_current_depth_waypoint(self, obs, goal_mask) -> Optional[DepthWaypoint]:
        waypoint_pixel = extract_anchor_pixel_from_mask(goal_mask)
        if waypoint_pixel is None:
            return None
        agent_xy, _, camera_position, camera_rotation = self.get_vlfm_pose_from_obs(obs)
        depth_metric = restore_metric_depth_from_habitat(obs["depth"], self.min_depth, self.max_depth)
        raw_waypoint = build_depth_waypoint_from_pixel(
            pixel=waypoint_pixel,
            depth_metric=depth_metric,
            camera_intrinsics=self.camera_intrinsics,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        return resolve_reachable_floor_waypoint(
            raw_waypoint=raw_waypoint,
            depth_metric=depth_metric,
            camera_intrinsics=self.camera_intrinsics,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            agent_xy=agent_xy,
        )

    def refresh_depth_waypoint(self, obs, goal_mask) -> Optional[DepthWaypoint]:
        waypoint = self.build_current_depth_waypoint(obs, goal_mask)
        if waypoint is None or not waypoint.valid:
            print("depth waypoint failed", None if waypoint is None else waypoint.failure_reason)
            self.current_waypoint = None
            return None
        self.controller.on_new_waypoint()
        self.current_waypoint = waypoint
        return waypoint

    def query_priors_text(self):
        return self.planner.query_priors_text()

    def make_plan(self, pano_images):
        return self.planner.make_plan(pano_images)

    def apply_priors_on_image(self, *args, **kwargs):
        return self.planner.apply_priors_on_image(*args, **kwargs)

    def are_bboxes_similar(self, *args, **kwargs):
        return self.planner.are_bboxes_similar(*args, **kwargs)

    @property
    def last_bboxes(self):
        return getattr(self.planner, "_last_bboxes", [])

    def act(self, obs) -> VOCANavigatorAction:
        waypoint = self.current_waypoint
        if waypoint is None or not waypoint.valid:
            return VOCANavigatorAction(action=0, request_replan=not self.goal_flag, waypoint=waypoint)

        agent_xy, heading, _, _ = self.get_vlfm_pose_from_obs(obs)
        pointgoal = compute_relative_pointgoal(
            waypoint_world=waypoint.world_position,
            current_agent_xy=agent_xy,
            current_heading=heading,
        )
        print(
            "depth_pointnav",
            "pixel", (waypoint.pixel_u, waypoint.pixel_v),
            "depth", waypoint.initial_depth,
            "target", waypoint.target_kind,
            "pn_step", self.controller.steps_for_waypoint,
            "rho", pointgoal.rho,
            "theta", pointgoal.theta,
        )

        if pointgoal.rho < self.controller.cfg.pointnav_stop_radius:
            action = 0
            request_replan = not self.goal_flag
            if request_replan:
                self.current_waypoint = None
            return VOCANavigatorAction(action=action, request_replan=request_replan, waypoint=waypoint)

        action = self.controller.act(obs["depth"], pointgoal.rho, pointgoal.theta)
        request_replan = action == 0 and not self.goal_flag
        if request_replan:
            self.current_waypoint = None
        return VOCANavigatorAction(action=action, request_replan=request_replan, waypoint=waypoint)
