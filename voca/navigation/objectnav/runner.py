import os
import time
from dataclasses import dataclass
from typing import Dict, List

import cv2
import imageio
import numpy as np
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

from .navigator import VOCANavigator


@dataclass
class ObjectNavEpisodeRunnerConfig:
    output_dir: str = "./tmp"
    video_fps: int = 4


class ObjectNavEpisodeRunner:
    def __init__(self, habitat_env, navigator: VOCANavigator, cfg: ObjectNavEpisodeRunnerConfig = None):
        self.env = habitat_env
        self.navigator = navigator
        self.cfg = cfg or ObjectNavEpisodeRunnerConfig()

    def run_episode(self, episode_index: int) -> Dict[str, object]:
        obs = self.env.reset()
        stats = {
            "dist_m": 0.0,
            "prev": np.array(self.env.sim.get_agent_state().position, dtype=np.float32),
        }
        original_step = self.env.step

        def instrumented_step(action):
            obs_ = original_step(action)
            cur = np.array(self.env.sim.get_agent_state().position, dtype=np.float32)
            stats["dist_m"] += float(np.linalg.norm(cur - stats["prev"]))
            stats["prev"] = cur
            return obs_

        self.env.step = instrumented_step
        try:
            return self._run_episode_with_recording(obs, episode_index, stats)
        finally:
            self.env.step = original_step

    def _run_episode_with_recording(self, obs, episode_index: int, stats) -> Dict[str, object]:
        episode_dir = os.path.join(self.cfg.output_dir, "trajectory_%d" % episode_index)
        os.makedirs(episode_dir, exist_ok=False)
        fps_writer = imageio.get_writer(os.path.join(episode_dir, "fps.mp4"), fps=self.cfg.video_fps)
        topdown_writer = imageio.get_writer(os.path.join(episode_dir, "metric.mp4"), fps=self.cfg.video_fps)

        episode_images = [obs["rgb"]]
        episode_topdowns = [self._adjust_topdown(self.env.get_metrics())]
        episode_state = {"step_counter": 0}
        start_geodesic_m = float(self.env.get_metrics()["distance_to_goal"])

        self.navigator.reset(self.env.current_episode.object_category)
        episode_t0 = time.perf_counter()

        def step_and_record(action):
            obs_ = self.env.step(action)
            episode_images.append(obs_["rgb"])
            episode_topdowns.append(self._adjust_topdown(self.env.get_metrics()))
            episode_state["step_counter"] += 1
            return obs_

        def run_actions(obs_, actions):
            for action_ in actions:
                if self.env.episode_over:
                    break
                obs_ = step_and_record(action_)
            return obs_

        def append_debug_frame(image):
            episode_images.append(image)
            episode_images.append(image)

        try:
            self.navigator.query_priors_text()

            obs = run_actions(obs, [3] * 11)
            initial_plan = self.navigator.make_initial_plan(episode_images[-12:])
            obs = run_actions(obs, self._rotation_actions_for_plan(initial_plan))
            append_debug_frame(initial_plan.vis_rgb)
            self.navigator.apply_initial_plan(obs, initial_plan)

            while not self.env.episode_over:
                nav_action = self.navigator.act(obs)
                action = nav_action.action
                request_replan = nav_action.request_replan

                if not request_replan:
                    self.navigator.record_executed_action(action)
                    obs = step_and_record(action)
                    continue

                if self.env.episode_over:
                    break

                obs = run_actions(obs, self.navigator.consume_heading_recovery_actions())
                if self.env.episode_over:
                    break

                if self.navigator.should_verify(action):
                    obs = run_actions(obs, [3] * 11)
                    if self.env.episode_over:
                        break

                    verify_plan = self.navigator.make_verification_plan(episode_images[-12:])
                    obs = run_actions(obs, self._rotation_actions_for_plan(verify_plan))
                    append_debug_frame(verify_plan.vis_rgb)
                    self.navigator.apply_verification_plan(obs, verify_plan)
                    continue

                obs = run_actions(obs, [3] * 3)
                if self.env.episode_over:
                    break

                pano_images, pano_angles, obs = self._collect_local_scan(obs, step_and_record, episode_images)
                if self.env.episode_over or len(pano_images) == 0:
                    break

                scan_result = self.navigator.evaluate_local_scan(pano_images, pano_angles)

                if scan_result.deadlocked:
                    obs = run_actions(obs, [3] * 11)
                    if self.env.episode_over:
                        break

                    deadlock_plan = self.navigator.make_deadlock_plan(episode_images[-12:])
                    obs = run_actions(obs, self._rotation_actions_for_plan(deadlock_plan))
                    append_debug_frame(deadlock_plan.vis_rgb)
                    self.navigator.apply_deadlock_plan(obs, deadlock_plan)
                else:
                    obs = run_actions(obs, scan_result.turn_actions)
                    append_debug_frame(scan_result.vis_rgb)
                    self.navigator.apply_local_scan_result(obs, scan_result)

                print("action", action)
                print("goal _flag", self.navigator.goal_flag)
                print("step_counter", episode_state["step_counter"])
                episode_state["step_counter"] = 0

            episode_time_sec = time.perf_counter() - episode_t0
            self._write_video_frames(fps_writer, episode_images)
            self._write_video_frames(topdown_writer, episode_topdowns)
            return self._episode_metrics(start_geodesic_m, episode_time_sec, stats)
        finally:
            fps_writer.close()
            topdown_writer.close()

    def _collect_local_scan(self, obs, step_and_record, episode_images):
        pano_images: List[np.ndarray] = [episode_images[-1]]
        pano_angles = [-90]
        for k in range(6):
            if self.env.episode_over:
                break
            obs = step_and_record(2)
            pano_images.append(obs["rgb"])
            pano_angles.append(-90 + 30 * (k + 1))
        return pano_images, pano_angles, obs

    @staticmethod
    def _rotation_actions_for_plan(plan):
        rotate_steps = min(11 - plan.rotate, 1 + plan.rotate)
        action = 3 if plan.rotate <= 6 else 2
        return [action] * rotate_steps

    @staticmethod
    def _adjust_topdown(metrics):
        return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 1024), cv2.COLOR_BGR2RGB)

    @staticmethod
    def _write_video_frames(writer, frames):
        for frame in frames:
            writer.append_data(frame)

    def _episode_metrics(self, start_geodesic_m: float, episode_time_sec: float, stats) -> Dict[str, object]:
        metrics = self.env.get_metrics()
        row = {
            "object_goal": self.env.current_episode.object_category,
            "success": metrics["success"],
            "spl": metrics["spl"],
            "start_distance_to_goal": start_geodesic_m,
            "final_distance_to_goal": metrics["distance_to_goal"],
            "episode_time_sec": float(episode_time_sec),
            "num_steps": int(metrics.get("num_steps", 0)),
            "total_distance_m": float(stats["dist_m"]),
        }
        row.update(self.navigator.runtime_metrics())
        return row
