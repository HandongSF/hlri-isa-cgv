import os
from settings import (
    DEFAULT_CUDA_VISIBLE_DEVICES,
    DEFAULT_DEVICE,
    POINTNAV_CHECKPOINT,
    YOLOE_CHECKPOINT_PATH,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_CUDA_VISIBLE_DEVICES)
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import argparse
import csv
import time

import cv2
import habitat
import imageio
import numpy as np
from habitat.config.default_structured_configs import NumStepsMeasurementConfig
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from voca.habitat import habitat_camera_intrinsic, hm3d_config
from voca.navigation.objectnav import VOCANavigator, VOCANavigatorConfig
from voca.perception import initialize_yoloe_model


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics["top_down_map"], 1024), cv2.COLOR_BGR2RGB)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=400)
    parser.add_argument("--pointnav_policy_path", type=str, default=POINTNAV_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser.parse_known_args()[0]


def rotation_actions_for_plan(plan):
    rotate_steps = min(11 - plan.rotate, 1 + plan.rotate)
    action = 3 if plan.rotate <= 6 else 2
    return [action] * rotate_steps


args = get_args()
habitat_config = hm3d_config(
    stage="val",
    episodes=args.eval_episodes,
    depth_pointnav=True,
)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)

OmegaConf.set_readonly(habitat_config, False)

with open_dict(habitat_config.habitat.task.measurements):
    if "num_steps" not in habitat_config.habitat.task.measurements:
        habitat_config.habitat.task.measurements.num_steps = NumStepsMeasurementConfig()

habitat_env = habitat.Env(habitat_config)
camera_intrinsics = habitat_camera_intrinsic(habitat_config)
depth_sensor_cfg = habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
min_depth = float(depth_sensor_cfg.min_depth)
max_depth = float(depth_sensor_cfg.max_depth)
camera_height = float(habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1])
print("controller      =", "depth_pointnav")
print("task_actions     =", habitat_config.habitat.task.actions)

DETECT_OBJECTS = ["bed", "sofa", "chair", "plant", "tv", "toilet", "floor"]
yoloe_model = initialize_yoloe_model(
    weights=YOLOE_CHECKPOINT_PATH,
    device=args.device,
    classes=DETECT_OBJECTS,
    prompt_mode="text",
)

navigator = VOCANavigator(
    yoloe_model,
    VOCANavigatorConfig(
        pointnav_policy_path=args.pointnav_policy_path,
        device=args.device,
        camera_intrinsics=camera_intrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
        camera_height=camera_height,
    ),
)
evaluation_metrics = []


for i in tqdm(range(args.eval_episodes)):
    obs = habitat_env.reset()

    # Wrap step to accumulate per-episode Euclidean travel distance.
    _stats = {
        "dist_m": 0.0,
        "prev": np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32),
    }
    _orig_step = habitat_env.step

    def _instrumented_step(action):
        obs_ = _orig_step(action)
        cur = np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32)
        _stats["dist_m"] += float(np.linalg.norm(cur - _stats["prev"]))
        _stats["prev"] = cur
        return obs_

    habitat_env.step = _instrumented_step

    episode_dir = "./tmp/trajectory_%d" % i
    os.makedirs(episode_dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % episode_dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % episode_dir, fps=4)

    start_geodesic_m = float(habitat_env.get_metrics()["distance_to_goal"])
    navigator.reset(habitat_env.current_episode.object_category)
    episode_images = [obs["rgb"]]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]
    episode_state = {"step_counter": 0}

    def step_and_record(action):
        obs_ = habitat_env.step(action)
        episode_images.append(obs_["rgb"])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        episode_state["step_counter"] += 1
        return obs_

    def run_actions(obs_, actions):
        for action_ in actions:
            if habitat_env.episode_over:
                break
            obs_ = step_and_record(action_)
        return obs_

    def append_debug_frame(image):
        episode_images.append(image)
        episode_images.append(image)

    # Measure per-episode compute time (exclude video I/O).
    episode_t0 = time.perf_counter()

    navigator.query_priors_text()

    obs = run_actions(obs, [3] * 11)
    initial_plan = navigator.make_initial_plan(episode_images[-12:])
    obs = run_actions(obs, rotation_actions_for_plan(initial_plan))
    append_debug_frame(initial_plan.vis_rgb)
    navigator.controller.reset()
    navigator.refresh_depth_waypoint(obs, initial_plan.goal_mask)

    while not habitat_env.episode_over:
        nav_action = navigator.act(obs)
        action = nav_action.action
        request_replan = nav_action.request_replan

        if not request_replan:
            navigator.record_executed_action(action)
            obs = step_and_record(action)
            continue

        if habitat_env.episode_over:
            break

        obs = run_actions(obs, navigator.consume_heading_recovery_actions())
        if habitat_env.episode_over:
            break

        if navigator.should_verify(action):
            obs = run_actions(obs, [3] * 11)
            if habitat_env.episode_over:
                break

            verify_plan = navigator.make_verification_plan(episode_images[-12:])
            obs = run_actions(obs, rotation_actions_for_plan(verify_plan))
            append_debug_frame(verify_plan.vis_rgb)
            navigator.apply_verification_plan(obs, verify_plan)
            continue

        obs = run_actions(obs, [3] * 3)
        if habitat_env.episode_over:
            break

        pano7 = [episode_images[-1]]
        angles7 = [-90]
        for k in range(6):
            if habitat_env.episode_over:
                break
            obs = step_and_record(2)
            pano7.append(obs["rgb"])
            angles7.append(-90 + 30 * (k + 1))

        if habitat_env.episode_over or len(pano7) == 0:
            break

        scan_result = navigator.evaluate_local_scan(pano7, angles7)

        if scan_result.deadlocked:
            obs = run_actions(obs, [3] * 11)
            if habitat_env.episode_over:
                break

            deadlock_plan = navigator.make_deadlock_plan(episode_images[-12:])
            obs = run_actions(obs, rotation_actions_for_plan(deadlock_plan))
            append_debug_frame(deadlock_plan.vis_rgb)
            navigator.apply_deadlock_plan(obs, deadlock_plan)
        else:
            obs = run_actions(obs, scan_result.turn_actions)
            append_debug_frame(scan_result.vis_rgb)
            navigator.apply_local_scan_result(obs, scan_result)

        print("action", action)
        print("goal _flag", navigator.goal_flag)
        print("step_counter", episode_state["step_counter"])
        episode_state["step_counter"] = 0

    habitat_env.step = _orig_step

    episode_t1 = time.perf_counter()
    episode_time_sec = episode_t1 - episode_t0

    for img in episode_images:
        fps_writer.append_data(img)

    for top in episode_topdowns:
        topdown_writer.append_data(top)

    fps_writer.close()
    topdown_writer.close()

    evaluation_metrics.append({
        "episode": i,
        "object_goal": habitat_env.current_episode.object_category,
        "success": habitat_env.get_metrics()["success"],
        "spl": habitat_env.get_metrics()["spl"],
        "start_distance_to_goal": start_geodesic_m,
        "final_distance_to_goal": habitat_env.get_metrics()["distance_to_goal"],
        "llm_calls": int(navigator.planner.llm_call_count),
        "llm_calls_deadlock": int(navigator.deadlock_llm_calls),
        "llm_calls_verification": int(navigator.verification_llm_calls),
        "llm_success_calls": int(navigator.planner.llm_success_count),
        "llm_error_calls": int(navigator.planner.llm_error_count),
        "llm_avg_time_sec": float(np.mean(navigator.planner.llm_durations)) if len(navigator.planner.llm_durations) > 0 else 0.0,
        "llm_last_error": str(navigator.planner.llm_last_error) if navigator.planner.llm_last_error else "",
        "yoloe_detect_calls": int(len(navigator.planner.yoloe_durations)),
        "yoloe_detect_avg_time_sec": float(np.mean(navigator.planner.yoloe_durations)) if len(navigator.planner.yoloe_durations) > 0 else 0.0,
        "yoloe_detect_total_time_sec": float(np.sum(navigator.planner.yoloe_durations)) if len(navigator.planner.yoloe_durations) > 0 else 0.0,
        "episode_time_sec": float(episode_time_sec),
        "num_steps": int(habitat_env.get_metrics().get("num_steps", 0)),
        "total_distance_m": float(_stats["dist_m"]),
    })

    write_metrics(evaluation_metrics)
