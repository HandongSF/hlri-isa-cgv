import os
import json
import math

# === 고정 파라미터 (네 코드 기준) ===
SCENE_PATH = "/home/gunminy/CGV-ISA/00646-UfhK7KNBg5u/UfhK7KNBg5u.basis.glb"

# 원래 중심 좌표 (그냥 첫 번째로 포함시킴)
SUCCESS_RADIUS_M = 2.0
OBJECT_GOAL = "bed"

EPISODES_PER_START = 4  # 5개 시작점 × 4회 = 20 episodes# === 새로운 시작 위치들 (총 5개) ===

# === 새로운 시작 위치들 (총 5개) ===
START_POSES = [
    # ([ 9.6938, 0.0383, -0.8666],  -440.0),
    # ([ 6.2769, 0.0383, -0.6637],   -90.0),
    # ([ 9.9784, 0.0383, 0.5916],  -720.0),
    # ([ 8.7667, 0.0383, -0.3268], -1150.0),
    ([10.6681, 0.0383, -0.9933], -120.0),
]

# === 침대 주변에서 직접 찍은 "성공으로 인정할 좌표들" ===
GOAL_POS_LIST_RAW = [
    [17.2573, 0.04, -0.0411],
    [17.0224, 0.04, -0.0411],
    [16.7875, 0.04, -0.0411],
    [16.5526, 0.04, -0.0411],
    [16.3176, 0.04, -0.0411],
    [16.1073, 0.04, -0.0411],
    [15.8146, 0.04, -0.1590],
    [15.6686, 0.04, -0.0496],
    [15.6500, 0.04,  0.1999],
    [15.6426, 0.04,  0.4445],
    [15.6352, 0.04,  0.6892],
    [15.6278, 0.04,  0.9339],
    [15.6204, 0.04,  1.1785],
    [15.6130, 0.04,  1.4232],
    [15.6564, 0.04,  1.6694],
    [15.7991, 0.04,  1.8778],
    [16.1591, 0.04,  1.8660],
    [16.6945, 0.04,  1.9512],
    [16.9294, 0.04,  1.8657],
    [17.1644, 0.04,  1.8089],
    [17.3993, 0.04,  1.8089],
]

# 중심 좌표 + 나머지 좌표들을 합쳐서 최종 goal 후보 리스트로 사용
GOAL_POS_LIST = GOAL_POS_LIST_RAW

# === 침대 근처에서 직접 캡쳐한 view_points 원시 데이터 (그대로 둠) ===
VIEW_POINTS_RAW = [
    # (position, yaw_deg) : 침대가 잘 보이는 시점들
    ([17.2573, 0.04, -0.0411],  910.0),
    ([17.0224, 0.04, -0.0411],  830.0),
    ([16.7875, 0.04, -0.0411],  830.0),
    ([16.5526, 0.04, -0.0411],  830.0),
    ([16.3176, 0.04, -0.0411],  830.0),
    ([16.1073, 0.04, -0.0411],  830.0),
    ([15.8146, 0.04, -0.1590],  810.0),
    ([15.6686, 0.04, -0.0496],  900.0),
    ([15.6500, 0.04,  0.1999],  910.0),
    ([15.6426, 0.04,  0.4445],  910.0),
    ([15.6352, 0.04,  0.6892],  910.0),
    ([15.6278, 0.04,  0.9339],  910.0),
    ([15.6204, 0.04,  1.1785],  910.0),
    ([15.6130, 0.04,  1.4232],  910.0),
    ([15.6564, 0.04,  1.6694],  910.0),
    ([15.7991, 0.04,  1.8778],  970.0),
    ([16.1591, 0.04,  1.8660], 1010.0),
    ([16.6945, 0.04,  1.9512], 1010.0),
    ([16.9294, 0.04,  1.8657], 1010.0),
    ([17.1644, 0.04,  1.8089], 1010.0),
    ([17.3993, 0.04,  1.8089], 1010.0),
]


def yaw_deg_to_quat_xyzw(yaw_deg: float):
    """Y 축 기준 yaw(deg) -> [x, y, z, w] 쿼터니언."""
    yaw_rad = math.radians(yaw_deg % 360.0)
    half = yaw_rad / 2.0
    x = 0.0
    y = math.sin(half)
    z = 0.0
    w = math.cos(half)
    return [x, y, z, w]


def make_view_point(position, yaw_deg: float):
    """Habitat ObjectNav용 view_point dict 생성."""
    quat = yaw_deg_to_quat_xyzw(yaw_deg)
    pos_r4 = [round(v, 4) for v in position]
    quat_r8 = [round(q, 8) for q in quat]

    return {
        "agent_state": {
            "position": pos_r4,
            "rotation": quat_r8,
        },
        "iou": 0.0,
    }


# 위에서 정의한 RAW 데이터 → view_points 리스트로 변환
BED_VIEW_POINTS = [make_view_point(pos, yaw) for pos, yaw in VIEW_POINTS_RAW]

episodes = []
ep_id = 0

for pos, yaw_deg in START_POSES:
    start_quat = yaw_deg_to_quat_xyzw(yaw_deg)

    # --- 여러 개의 goal 생성 (침대 주변 모든 좌표) ---
    goals = []
    for gid, goal_pos in enumerate(GOAL_POS_LIST):
        goals.append(
            {
                "position": [round(v, 4) for v in goal_pos],
                "radius": SUCCESS_RADIUS_M,
                "object_id": gid,                 # 각 goal마다 다른 id
                "object_category": OBJECT_GOAL,
                "view_points": BED_VIEW_POINTS,   # 공통 view_points (필요 없으면 None으로 바꿔도 됨)
            }
        )

    # 이 맵에서는 bed가 하나뿐이니까, 모든 에피소드가 동일 goals 세트 공유
    for _ in range(EPISODES_PER_START):
        episodes.append(
            {
                "episode_id": str(ep_id),
                "scene_id": SCENE_PATH,  # 절대 경로 그대로 사용 (scenes_dir=""로 둘 것)
                "start_position": [round(v, 4) for v in pos],
                "start_rotation": [round(v, 8) for v in start_quat],
                "object_category": OBJECT_GOAL,
                "goals": goals,          # ★ 중요: 여러 개의 목표 좌표
                "start_room": None,
                "shortest_paths": None,
                "info": {},
            }
        )
        ep_id += 1

dataset = {
    "episodes": episodes,
    "category_to_task_category_id": {
        "bed": 0,
    },
    "category_to_scene_annotation_category_id": {
        "bed": 0,
    },
}

# === GLB 옆에 JSON 파일로 저장 ===
out_dir = os.path.dirname(SCENE_PATH)
out_path = os.path.join(out_dir, "isa_00646_bed.json")

os.makedirs(out_dir, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Saved custom ObjectNav episodes to: {out_path}")
print("다음 명령으로 gzip 생성:")
print(f"  gzip -kf {out_path}")
print("그리고 objnav_benchmark.py 실행할 때 --episode_path 에 .json.gz 경로를 넘기면 됨.")
