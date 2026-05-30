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

import habitat
from habitat.config.default_structured_configs import NumStepsMeasurementConfig
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from voca.habitat import habitat_camera_intrinsic, hm3d_config
from voca.navigation.objectnav import (
    ObjectNavEpisodeRunner,
    VOCANavigator,
    VOCANavigatorConfig,
)
from voca.perception import initialize_yoloe_model


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=400)
    parser.add_argument("--pointnav_policy_path", type=str, default=POINTNAV_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser.parse_known_args()[0]


def build_habitat_env(eval_episodes):
    habitat_config = hm3d_config(
        stage="val",
        episodes=eval_episodes,
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
    print("controller      =", "depth_pointnav")
    print("task_actions     =", habitat_config.habitat.task.actions)
    return habitat_env, habitat_config


def build_navigator(args, habitat_config):
    camera_intrinsics = habitat_camera_intrinsic(habitat_config)
    depth_sensor_cfg = habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    min_depth = float(depth_sensor_cfg.min_depth)
    max_depth = float(depth_sensor_cfg.max_depth)
    camera_height = float(habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1])

    detect_objects = ["bed", "sofa", "chair", "plant", "tv", "toilet", "floor"]
    yoloe_model = initialize_yoloe_model(
        weights=YOLOE_CHECKPOINT_PATH,
        device=args.device,
        classes=detect_objects,
        prompt_mode="text",
    )

    return VOCANavigator(
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


def main():
    args = get_args()
    habitat_env, habitat_config = build_habitat_env(args.eval_episodes)
    navigator = build_navigator(args, habitat_config)
    runner = ObjectNavEpisodeRunner(habitat_env, navigator)

    evaluation_metrics = []
    for episode_index in tqdm(range(args.eval_episodes)):
        episode_metrics = runner.run_episode(episode_index)
        evaluation_metrics.append({"episode": episode_index, **episode_metrics})
        write_metrics(evaluation_metrics)


if __name__ == "__main__":
    main()
