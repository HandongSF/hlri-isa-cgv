from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass
class ObjectNavPlan:
    goal_image: np.ndarray
    goal_mask: np.ndarray
    debug_image: np.ndarray
    vis_rgb: np.ndarray
    rotate: int
    pri_flag: bool
    obj_detected: bool


@dataclass
class LocalScanResult:
    deadlocked: bool
    turn_actions: List[int]
    goal_image: Optional[np.ndarray] = None
    goal_mask: Optional[np.ndarray] = None
    vis_rgb: Optional[np.ndarray] = None
    pri_flag: bool = False
    obj_detected: bool = False


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
        self.prev_boxes = None
        self.heading_offset = 0
        self.deadlock_llm_calls = 0
        self.verification_llm_calls = 0

    def reset(self, object_goal: str) -> None:
        self.planner.reset(object_goal)
        self.controller.reset()
        self.current_waypoint = None
        self.goal_flag = False
        self.pending_verify = False
        self.prev_boxes = None
        self.heading_offset = 0
        self.deadlock_llm_calls = 0
        self.verification_llm_calls = 0

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

    def make_initial_plan(self, pano_images) -> ObjectNavPlan:
        plan = self._make_objectnav_plan(pano_images)
        self._update_goal_state(plan.pri_flag, plan.obj_detected)
        return plan

    def make_verification_plan(self, pano_images) -> ObjectNavPlan:
        llm_calls_before = int(self.planner.llm_call_count)
        plan = self._make_objectnav_plan(pano_images)
        self.verification_llm_calls += int(self.planner.llm_call_count) - llm_calls_before
        return plan

    def apply_verification_plan(self, obs, plan: ObjectNavPlan) -> None:
        if plan.obj_detected:
            self._update_goal_state(pri_flag=plan.pri_flag, obj_detected=True)
        else:
            self._update_goal_state(
                pri_flag=plan.pri_flag,
                obj_detected=False,
                prev_boxes=self.last_bboxes,
            )
        self.refresh_depth_waypoint(obs, plan.goal_mask)

    def make_deadlock_plan(self, pano_images) -> ObjectNavPlan:
        llm_calls_before = int(self.planner.llm_call_count)
        plan = self._make_objectnav_plan(pano_images)
        self.deadlock_llm_calls += int(self.planner.llm_call_count) - llm_calls_before
        return plan

    def apply_deadlock_plan(self, obs, plan: ObjectNavPlan) -> None:
        self.prev_boxes = self.last_bboxes
        self._update_goal_state(plan.pri_flag, plan.obj_detected, prev_boxes=self.prev_boxes)
        self.refresh_depth_waypoint(obs, plan.goal_mask)

    def apply_priors_on_image(self, *args, **kwargs):
        return self.planner.apply_priors_on_image(*args, **kwargs)

    def are_bboxes_similar(self, *args, **kwargs):
        return self.planner.are_bboxes_similar(*args, **kwargs)

    def evaluate_local_scan(self, pano_images, angles) -> LocalScanResult:
        if self.prev_boxes is None:
            self.prev_boxes = self.last_bboxes

        (
            direction_image,
            debug_mask,
            pri_flag,
            obj_detected,
            debug_vis,
            curr_boxes,
            best_idx,
        ) = self.apply_priors_on_image(pano_images, return_boxes=True)

        if self.are_bboxes_similar(
            self.prev_boxes,
            curr_boxes,
            class_sensitive=False,
            ignore_classes=["floor", "ground", "flooring"],
            return_detail=False,
        ):
            return LocalScanResult(deadlocked=True, turn_actions=[])

        cur_deg = int(angles[-1])
        sel_deg = int(angles[best_idx])
        delta = sel_deg - cur_deg
        turns = abs(delta) // 30
        if delta < 0:
            turn_actions = [3] * turns
        elif delta > 0:
            turn_actions = [2] * turns
        else:
            turn_actions = []

        self.prev_boxes = curr_boxes
        self._update_goal_state(pri_flag, obj_detected, prev_boxes=curr_boxes)
        return LocalScanResult(
            deadlocked=False,
            turn_actions=turn_actions,
            goal_image=direction_image,
            goal_mask=debug_mask,
            vis_rgb=debug_vis,
            pri_flag=pri_flag,
            obj_detected=obj_detected,
        )

    def apply_local_scan_result(self, obs, result: LocalScanResult) -> None:
        if result.goal_mask is not None:
            self.refresh_depth_waypoint(obs, result.goal_mask)

    @property
    def last_bboxes(self):
        return getattr(self.planner, "_last_bboxes", [])

    def _make_objectnav_plan(self, pano_images) -> ObjectNavPlan:
        (
            goal_image,
            goal_mask,
            debug_image,
            vis_rgb,
            rotate,
            pri_flag,
            obj_detected,
        ) = self.planner.make_plan(pano_images)
        return ObjectNavPlan(
            goal_image=goal_image,
            goal_mask=goal_mask,
            debug_image=debug_image,
            vis_rgb=vis_rgb,
            rotate=rotate,
            pri_flag=pri_flag,
            obj_detected=obj_detected,
        )

    def _update_goal_state(self, pri_flag: bool, obj_detected: bool, prev_boxes=None) -> None:
        self.pending_verify = bool(pri_flag and not obj_detected)
        self.goal_flag = bool(obj_detected)
        if prev_boxes is not None:
            self.prev_boxes = prev_boxes

    def record_executed_action(self, action: int) -> None:
        if action == 4:
            self.heading_offset += 1
        elif action == 5:
            self.heading_offset -= 1

    def consume_heading_recovery_actions(self) -> List[int]:
        if self.heading_offset > 0:
            actions = [5] * self.heading_offset
        elif self.heading_offset < 0:
            actions = [4] * abs(self.heading_offset)
        else:
            actions = []
        self.heading_offset = 0
        return actions

    def should_verify(self, action: int) -> bool:
        return bool(self.pending_verify and action == 0)

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
