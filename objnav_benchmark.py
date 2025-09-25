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
from omegaconf import OmegaConf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

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
    parser.add_argument("--eval_episodes", type=int, default=200)
    # How often to call nav_executor.reset within a running episode (in env steps)
    parser.add_argument("--reset_interval", type=int, default=10)
    return parser.parse_known_args()[0]


args = get_args()
habitat_config = hm3d_config(stage='val', episodes=args.eval_episodes)
print("scene_dataset =", habitat_config.habitat.simulator.scene_dataset)
print("scenes_dir    =", habitat_config.habitat.dataset.scenes_dir)
print("data_path     =", habitat_config.habitat.dataset.data_path)
OmegaConf.set_readonly(habitat_config, False)
habitat_config.habitat.environment.max_episode_steps = 600  # 예: 600스텝에서 타임아웃
habitat_env = habitat.Env(habitat_config)

# ✅ YOLOE 초기화 (세그 가중치 필수)
DETECT_OBJECTS = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
yoloe_model = initialize_yoloe_model(
    weights=YOLOE_CHECKPOINT_PATH,   # 세그 지원 가중치
    device="cuda:0",
    classes=DETECT_OBJECTS,       # 텍스트 프롬프트 기본 세팅
    prompt_mode="text",
    conf_threshold=0.10,
    iou_threshold=0.50,
)

# ✅ 플래너/에이전트
#   - 플래너가 YOLOE 한 개만 받도록 바꿨다면 ↓ 그대로 사용
#   - 여전히 (dino_model, sam_model) 2개 인자를 받는 옛 버전이라면 같은 모델을 두 번 넣는다.
try:
    nav_planner = GPT4V_Planner(yoloe_model)
except TypeError:
    # fallback: 옛 시그니처 호환
    nav_planner = GPT4V_Planner(yoloe_model, yoloe_model)

nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
evaluation_metrics = []

for i in tqdm(range(args.eval_episodes)):
    find_goal = False
    obs = habitat_env.reset()
    dir = "./tmp/trajectory_%d" % i
    os.makedirs(dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % dir, fps=4)
    heading_offset = 0
    # step counters to rate-limit expensive resets
    step_counter = 0
    last_reset_step = 0
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])  

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # ====== ⏱️ 에피소드 '계산' 시간 측정 시작 (비디오 I/O 제외) ======
    episode_t0 = time.perf_counter()

    # a whole round planning process
    #LLM 첫 호출 뒤 방향 정하고 줄발
    for _ in range(11):
        obs = habitat_env.step(3)
        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        step_counter += 1
    goal_image, goal_mask, _, debug_image, goal_rotate, goal_flag = nav_planner.make_plan(episode_images[-12:])
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

    episode_images.append(debug_image)
    episode_images.append(debug_image)
    nav_executor.reset(goal_image, goal_mask)
    last_reset_step = step_counter

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

            # planning with priors
            direction_image, debug_mask, pri_flag, debug_image = nav_planner.apply_priors_on_image(obs['rgb'])
            episode_images.append(debug_image)
            episode_images.append(debug_image)
            goal_image, goal_mask = direction_image, debug_mask
            goal_flag = pri_flag  # 이후 while 루프 상단의 조건문에서 활용

            #임시 무한루프 끊기
            if step_counter == 0 and goal_flag == False :
                goal_flag = True

            print("action", action)
            print("goal _flag", goal_flag)
            print("step_counter", step_counter)
            step_counter = 0
            #action이 0이고 goal_flag가 false이면 웨이포인트를 새로 지정해야 하지만
            #웨이포인트를 지정하자마자 action이 0이 되는 경우 반복적인 pixnav정책 호출이 일어날 수 있음.
            # PixelNav reset (새 웨이포인트 반영) 
            nav_executor.reset(goal_image, goal_mask)


    # ====== ⏱️ 에피소드 '계산' 시간 측정 종료 ======
    episode_t1 = time.perf_counter()
    episode_time_sec = episode_t1 - episode_t0

    for image, topdown in zip(episode_images, episode_topdowns):
        fps_writer.append_data(image)
        topdown_writer.append_data(topdown)
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
        'episode_time_sec': float(episode_time_sec),  # ✅ 에피소드 총 계산 시간(초)
    })

    write_metrics(evaluation_metrics)