import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import habitat
import argparse
import csv
import cv2
import imageio
import numpy as np
import time
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from cv_utils.yoloe_tools import initialize_yoloe_model
from omegaconf import OmegaConf, open_dict


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'], 1024), cv2.COLOR_BGR2RGB)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=1000)
    return parser.parse_known_args()[0]


args = get_args()
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)

OmegaConf.set_readonly(habitat_config, False)

from habitat.config.default_structured_configs import NumStepsMeasurementConfig
with open_dict(habitat_config.habitat.task.measurements):
    if "num_steps" not in habitat_config.habitat.task.measurements:
        habitat_config.habitat.task.measurements.num_steps = NumStepsMeasurementConfig()

habitat_env = habitat.Env(habitat_config)

DETECT_OBJECTS = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
yoloe_model = initialize_yoloe_model(
    weights=YOLOE_CHECKPOINT_PATH,
    device="cuda:0",
    classes=DETECT_OBJECTS,
    prompt_mode="text",
)

try:
    nav_planner = GPT4V_Planner(yoloe_model)
except TypeError:
    nav_planner = GPT4V_Planner(yoloe_model, yoloe_model)

nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
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

    dir = "./tmp/trajectory_%d" % i
    os.makedirs(dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % dir, fps=4)
    heading_offset = 0
    step_counter = 0
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])  
    prev_boxes = None
    curr_boxes = None
    pending_verify = False  
    goal_flag = False

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # Measure per-episode compute time (exclude video I/O).
    episode_t0 = time.perf_counter()

    nav_planner.query_priors_text()

    for _ in range(11):
        obs = habitat_env.step(3)
        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        step_counter += 1
    goal_image, goal_mask, debug_image, vis_rgb, goal_rotate,  pri_flag, obj_detected = nav_planner.make_plan(episode_images[-12:])
    pending_verify = (pri_flag and (not obj_detected))
    goal_flag = obj_detected
    for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            obs = habitat_env.step(2)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        step_counter += 1

    episode_images.append(vis_rgb)
    episode_images.append(vis_rgb)
    nav_executor.reset(goal_image, goal_mask)


    while not habitat_env.episode_over:
        action, skill_image = nav_executor.step(obs['rgb'], habitat_env.sim.previous_step_collided)

        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            step_counter += 1

        else:
            if habitat_env.episode_over:
                break
            for _ in range(0, abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset -= 1
                    step_counter += 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset += 1
                    step_counter += 1

            if pending_verify and action == 0:
                for _ in range(11):
                    if habitat_env.episode_over: break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                (
                    verify_goal_image,
                    verify_goal_mask,
                    verify_debug_image,
                    verify_vis,
                    verify_rotate,
                    verify_pri_flag,
                    verify_obj_detected
                ) = nav_planner.make_plan(episode_images[-12:])

                for j in range(min(11 - verify_rotate, 1 + verify_rotate)):
                    if habitat_env.episode_over: break
                    if verify_rotate <= 6:
                        obs = habitat_env.step(3)
                    else:
                        obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                episode_images.append(verify_vis)
                episode_images.append(verify_vis)

                if verify_obj_detected:
                    goal_flag = True
                    pending_verify = False
                    continue

                elif verify_pri_flag:
                    pending_verify = True
                    goal_flag = False
                    prev_boxes = getattr(nav_planner, "_last_bboxes", [])
                else:
                    pending_verify = False
                    goal_flag = False
                    prev_boxes = getattr(nav_planner, "_last_bboxes", [])

                nav_executor.reset(verify_goal_image, verify_goal_mask)
                continue
            
            prev_boxes = getattr(nav_planner, "_last_bboxes", []) if prev_boxes is None else prev_boxes

            pano7 = []
            angles7 = []

            for _ in range(3):
                if habitat_env.episode_over: break
                obs = habitat_env.step(3)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                step_counter += 1

            pano7.append(episode_images[-1]); angles7.append(-90)

            for k in range(6):
                if habitat_env.episode_over: break
                obs = habitat_env.step(2)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                step_counter += 1
                pano7.append(obs['rgb'])
                angles7.append(-90 + 30*(k+1))  # -60, -30, 0, +30, +60, +90

            if habitat_env.episode_over or len(pano7) == 0:
                break

            (
                direction_image,   # goal_rgb
                debug_mask,
                pri_flag,
                obj_detected,
                debug_vis,         # vis_rgb
                curr_boxes,
                best_idx
            ) = nav_planner.apply_priors_on_image(pano7, return_boxes=True)


            if nav_planner.are_bboxes_similar(
                prev_boxes, curr_boxes,
                class_sensitive=False,
                ignore_classes=['floor','ground','flooring'],
                return_detail=False
            ):
                for _ in range(11):
                    if habitat_env.episode_over: break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                (
                    goal_image,
                    goal_mask,
                    debug_image2,
                    vis_rgb2,
                    goal_rotate,
                    pri_flag,
                    obj_detected
                ) = nav_planner.make_plan(episode_images[-12:])

                for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
                    if habitat_env.episode_over: break
                    if goal_rotate <= 6:
                        obs = habitat_env.step(3)
                    else:
                        obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                episode_images.append(vis_rgb2); episode_images.append(vis_rgb2)

                prev_boxes = getattr(nav_planner, "_last_bboxes", [])
                pending_verify = (pri_flag and (not obj_detected))
                goal_flag = obj_detected

            else:
                cur_deg = int(angles7[-1])           # +90
                sel_deg = int(angles7[best_idx])     # {-90,-60,-30,0,30,60,90}
                delta = sel_deg - cur_deg
                turns = abs(delta) // 30
                if delta < 0:
                    for _ in range(turns):
                        if habitat_env.episode_over: break
                        obs = habitat_env.step(3)
                        episode_images.append(obs['rgb'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        step_counter += 1
                elif delta > 0:
                    for _ in range(turns):
                        if habitat_env.episode_over: break
                        obs = habitat_env.step(2)
                        episode_images.append(obs['rgb'])
                        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                        step_counter += 1

                episode_images.append(debug_vis); episode_images.append(debug_vis)
                goal_image, goal_mask = direction_image, debug_mask
                pending_verify = pri_flag and (not obj_detected)
                goal_flag = obj_detected
                prev_boxes = curr_boxes

            print("action", action)
            print("goal _flag", goal_flag)
            print("step_counter", step_counter)
            step_counter = 0
            nav_executor.reset(goal_image, goal_mask)

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
        'episode': i,
        'object_goal': habitat_env.current_episode.object_category,
        'success': habitat_env.get_metrics()['success'],
        'spl': habitat_env.get_metrics()['spl'],
        'start_distance_to_goal': start_geodesic_m, 
        'final_distance_to_goal': habitat_env.get_metrics()['distance_to_goal'],
        'llm_calls': int(nav_planner.llm_call_count),
        'llm_avg_time_sec': float(np.mean(nav_planner.llm_durations)) if len(nav_planner.llm_durations) > 0 else 0.0,
        'episode_time_sec': float(episode_time_sec),
        'num_steps': int(habitat_env.get_metrics().get('num_steps', 0)),
        'total_distance_m': float(_stats['dist_m']),
    })

    write_metrics(evaluation_metrics)
