import os
import numpy as np
import cv2
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

SCENE_DIR = "00646-UfhK7KNBg5u"
SCENE_GLb = os.path.join(SCENE_DIR, "UfhK7KNBg5u.basis.glb")
NAVMESH   = os.path.join(SCENE_DIR, "UfhK7KNBg5u.basis.navmesh")


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_GLb
    sim_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 1.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    step_size = 0.25
    turn_angle = 10.0

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=step_size),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=turn_angle),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=turn_angle),
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    ok = sim.pathfinder.load_nav_mesh(NAVMESH)
    print("Navmesh load:", ok)

    return sim


def set_agent_state(sim, position, yaw_deg=0.0):
    agent = sim.get_agent(0)
    state = agent.state
    state.position = np.array(position, dtype=np.float32)
    state.rotation = quat_from_angle_axis(
        np.deg2rad(yaw_deg), np.array([0, 1, 0], dtype=np.float32)
    )
    agent.set_state(state)


if __name__ == "__main__":
    sim = make_sim()

    start_pos = sim.pathfinder.get_random_navigable_point()
    current_yaw = 0.0
    set_agent_state(sim, start_pos, current_yaw)

    print("=== 00646 viewer ===")
    print("w: 앞, a: 좌회전, d: 우회전")
    print("p: 현재 pose 출력 (좌표 뽑기용)")
    print("r: navmesh 랜덤 위치로 텔레포트")
    print("q 또는 ESC: 종료")
    print("---------------------")

    while True:
        # 관측 받아서 이미지로 표시
        obs = sim.get_sensor_observations()
        rgba = obs["rgb"]  # (H, W, 4) RGBA, uint8
        rgb = rgba[:, :, :3]               # alpha 버리고
        bgr = rgb[:, :, ::-1].copy()       # RGB -> BGR

        cv2.imshow("00646 RGB", bgr)
        key = cv2.waitKey(0) & 0xFF  # 키 입력 기다리기

        if key == ord('w'):
            sim.step("move_forward")
        elif key == ord('a'):
            sim.step("turn_left")
            current_yaw += 10.0
        elif key == ord('d'):
            sim.step("turn_right")
            current_yaw -= 10.0
        elif key == ord('p'):
            st = sim.get_agent(0).state
            pos = st.position
            print(f"POS = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], YAW = {current_yaw:.1f} deg")
        elif key == ord('r'):
            p = sim.pathfinder.get_random_navigable_point()
            current_yaw = 0.0
            set_agent_state(sim, p, current_yaw)
            print("Teleported to:", p)
        elif key == ord('q') or key == 27:  # 'q' or ESC
            break
        else:
            print(f"Unknown key: {key}")

    sim.close()
    cv2.destroyAllWindows()
