import numpy as np


def habitat_camera_intrinsic(config):
    depth_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    rgb_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    assert depth_sensor.width == rgb_sensor.width, "The configuration of the depth camera should be the same as rgb camera."
    assert depth_sensor.height == rgb_sensor.height, "The configuration of the depth camera should be the same as rgb camera."
    assert depth_sensor.hfov == rgb_sensor.hfov, "The configuration of the depth camera should be the same as rgb camera."

    width = depth_sensor.width
    height = depth_sensor.height
    hfov = depth_sensor.hfov
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    return np.array(
        [
            [f, 0.0, xc],
            [0.0, f, zc],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
