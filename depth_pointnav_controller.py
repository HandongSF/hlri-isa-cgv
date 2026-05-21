from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from third_party.vlfm_pointnav.pointnav_policy import WrappedPointNavResNetPolicy
from constants import POINTNAV_CHECKPOINT


@dataclass
class DepthPointNavConfig:
    pointnav_policy_path: str = POINTNAV_CHECKPOINT
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    max_pointnav_steps: int = 32
    reset_pointnav_on_new_waypoint: bool = True
    device: str = "cuda"


def _shape_tuple(value: Union[Sequence[int], Tuple[int, int]]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("depth_image_shape must contain height and width")
    return int(value[0]), int(value[1])


def format_depth_for_pointnav(
    depth_norm: Union[np.ndarray, torch.Tensor],
    output_size: Tuple[int, int],
    device: Union[str, torch.device],
) -> torch.Tensor:
    if isinstance(depth_norm, np.ndarray):
        depth_tensor = torch.from_numpy(depth_norm)
    else:
        depth_tensor = depth_norm.detach()

    depth_tensor = depth_tensor.to(device=device, dtype=torch.float32)
    if depth_tensor.ndim == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(-1)
    elif depth_tensor.ndim == 3:
        if depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor.unsqueeze(0)
        else:
            depth_tensor = depth_tensor.unsqueeze(-1)
    elif depth_tensor.ndim != 4:
        raise ValueError(f"Unsupported depth shape: {tuple(depth_tensor.shape)}")

    depth_tensor = depth_tensor.clamp(0.0, 1.0)
    depth_tensor = depth_tensor.permute(0, 3, 1, 2)
    depth_tensor = F.interpolate(depth_tensor, size=_shape_tuple(output_size), mode="area")
    return depth_tensor.permute(0, 2, 3, 1).contiguous()


class DepthPointNavController:
    def __init__(self, cfg: DepthPointNavConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.depth_image_shape = _shape_tuple(cfg.depth_image_shape)
        self.policy = WrappedPointNavResNetPolicy(cfg.pointnav_policy_path, device=self.device)
        self._has_acted = False
        self._steps_for_waypoint = 0

    def reset(self) -> None:
        self.policy.reset()
        self._has_acted = False
        self._steps_for_waypoint = 0

    def on_new_waypoint(self) -> None:
        if self.cfg.reset_pointnav_on_new_waypoint:
            self.reset()

    @property
    def steps_for_waypoint(self) -> int:
        return self._steps_for_waypoint

    def act(self, depth_obs: Union[np.ndarray, torch.Tensor], rho: float, theta: float) -> int:
        if self.cfg.max_pointnav_steps > 0 and self._steps_for_waypoint >= self.cfg.max_pointnav_steps:
            return 0

        depth_tensor = format_depth_for_pointnav(depth_obs, self.depth_image_shape, self.device)
        rho_theta_tensor = torch.tensor([[rho, theta]], device=self.device, dtype=torch.float32)
        observations = {
            "depth": depth_tensor,
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        masks = torch.tensor([[self._has_acted]], device=self.device, dtype=torch.bool)
        action = self.policy.act(observations, masks, deterministic=True)
        self._has_acted = True
        self._steps_for_waypoint += 1
        return int(action.detach().cpu().numpy().reshape(-1)[0])
