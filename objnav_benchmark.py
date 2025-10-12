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

# ✅ YOLOE 초기화 (세그 가중치 필수)
DETECT_OBJECTS = ['bed', 'sofa', 'chair', 'plant', 'tv', 'toilet', 'floor']
yoloe_model = initialize_yoloe_model(
    weights=YOLOE_CHECKPOINT_PATH,   # 세그 지원 가중치
    device="cuda:0",
    classes=DETECT_OBJECTS,       # 텍스트 프롬프트 기본 세팅
    prompt_mode="text",
)

# ✅ 플래너/에이전트
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

    # === NEW: 에피소드 총 이동거리(유클리드) 집계용 step 래핑 시작 ===
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
    # ===========================================================

    dir = "./tmp/trajectory_%d" % i
    os.makedirs(dir, exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4" % dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % dir, fps=4)
    heading_offset = 0
    # step counters to rate-limit expensive resets
    step_counter = 0
    start_geodesic_m = float(habitat_env.get_metrics()['distance_to_goal'])  
    prev_boxes = None
    curr_boxes = None

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # ====== ⏱️ 에피소드 '계산' 시간 측정 시작 (비디오 I/O 제외) ======
    episode_t0 = time.perf_counter()

    # === 첫 호출 후 Priors 적용
    nav_planner.query_priors_text()

    # a whole round planning process
    # LLM 첫 호출 뒤 방향 정하고 출발
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
            #웨이포인트 따라가는 중, 목표물을 봤다면 다시 웨이포인트 지정 후 목표물까지 이동!
            # if nav_planner.obs_goal_object(obs['rgb']) and goal_flag == False: 
            #     direction_image, debug_mask, pri_flag, debug_image = nav_planner.apply_priors_on_image(obs['rgb'])
            #     episode_images.append(debug_image)
            #     episode_images.append(debug_image)
            #     goal_image, goal_mask = direction_image, debug_mask
            #     goal_flag = True  # 이후 while 루프 상단의 조건문에서 활용
            #     print("action", action)
            #     print("goal _flag", goal_flag)
            #     print("step_counter", step_counter)
            #     step_counter = 0

            #     # heading 조정
            #     for _ in range(0, abs(heading_offset)):
            #         if habitat_env.episode_over:
            #             break
            #         if heading_offset > 0:
            #             obs = habitat_env.step(5)
            #             episode_images.append(obs['rgb'])
            #             episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            #             heading_offset -= 1
            #             step_counter += 1
            #         elif heading_offset < 0:
            #             obs = habitat_env.step(4)
            #             episode_images.append(obs['rgb'])
            #             episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            #             heading_offset += 1
            #             step_counter += 1

            #     nav_executor.reset(goal_image, goal_mask)  # PixelNav reset (새 웨이포인트 반영) 

        else:
            if habitat_env.episode_over:
                break
            # heading 조정
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
            
            prev_boxes = getattr(nav_planner, "_last_bboxes", []) if prev_boxes is None else prev_boxes
            direction_image, debug_mask, pri_flag, debug_image, curr_boxes = \
                nav_planner.apply_priors_on_image(obs['rgb'], return_boxes=True)
            episode_images.append(debug_image); episode_images.append(debug_image)

            if nav_planner.are_bboxes_similar(
                prev_boxes, curr_boxes,
                iou_thresh=0.85, center_tol=0.01, area_tol=0.10,   # 엄격 프리셋(원하면 조정)
                min_match_ratio=0.95,
                class_sensitive=True,
                ignore_classes=['floor','ground','flooring'],
                return_detail=False
            ):
                # 정책이 앞으로 가지 못한다고 판단 → 다시 VLM 호출
                for _ in range(11):
                    if habitat_env.episode_over: break
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                goal_image, goal_mask, _, debug_image2, goal_rotate, goal_flag = \
                    nav_planner.make_plan(episode_images[-12:])

                for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
                    if habitat_env.episode_over: break
                    if goal_rotate <= 6:
                        obs = habitat_env.step(3)
                    else:
                        obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    step_counter += 1

                episode_images.append(debug_image2); episode_images.append(debug_image2)

                # make_plan 내부에서 apply_priors_on_image가 다시 호출되어 _last_bboxes 갱신됨
                prev_boxes = getattr(nav_planner, "_last_bboxes", [])

                # 최신 goal 세팅은 make_plan 반환값 사용
                # goal_flag는 make_plan의 pri_flag
            else:
                # 유사하지 않음 → priors 결과로 그대로 진행
                goal_image, goal_mask = direction_image, debug_mask
                goal_flag = pri_flag

                # 다음 프레임 비교를 위해 prev 갱신
                prev_boxes = curr_boxes

            print("action", action)
            print("goal _flag", goal_flag)
            print("step_counter", step_counter)
            step_counter = 0
            nav_executor.reset(goal_image, goal_mask)

    # === NEW: 래핑 원복(중첩 방지) ===
    habitat_env.step = _orig_step

    # ====== ⏱️ 에피소드 '계산' 시간 측정 종료 ======
    episode_t1 = time.perf_counter()
    episode_time_sec = episode_t1 - episode_t0


    # --- 서로 길이 다르게, 각각 독립적으로 쓰기 ---
    for img in episode_images:
        # 필요시: img = img.astype(np.uint8)
        fps_writer.append_data(img)

    for top in episode_topdowns:
        # 필요시: top = top.astype(np.uint8)
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
        'episode_time_sec': float(episode_time_sec),  # ✅ 에피소드 총 계산 시간(초)

        # === NEW: 추가 지표 ===
        'num_steps': int(habitat_env.get_metrics().get('num_steps', 0)),  # Habitat 빌트인 스텝 수
        'total_distance_m': float(_stats['dist_m']),                      # 유클리드 총 이동거리(m)
    })

    write_metrics(evaluation_metrics)
