"""PointNav-VO visual odometry inference.

This is a small, inference-only adaptation of the PointNav-VO ICCV 2021 code:
https://github.com/Xiaoming-Zhao/PointNav-VO
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F


MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
SUPPORTED_ACTIONS = (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT)


@dataclass
class PointNavVisualOdometryConfig:
    forward_checkpoint: str
    turn_checkpoint: str
    device: str = "cuda"
    rgb_key: str = "vo_rgb"
    depth_key: str = "vo_depth"
    image_width: int = 341
    image_height: int = 192
    min_depth: float = 0.1
    max_depth: float = 10.0
    hfov: float = 70.0
    discretized_depth_channels: int = 10


class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))
        self._distributed = distrib.is_initialized()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            new_mean = F.adaptive_avg_pool2d(x, 1).sum(0, keepdim=True)
            new_count = torch.full_like(self._count, x.size(0))
            if self._distributed:
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)
            new_mean /= new_count

            new_var = F.adaptive_avg_pool2d((x - new_mean).pow(2), 1).sum(0, keepdim=True)
            if self._distributed:
                distrib.all_reduce(new_var)
            new_var /= new_count

            m_a = self._var * self._count
            m_b = new_var * new_count
            m2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )
            self._var = m2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )
            self._count += new_count

        stdev = torch.sqrt(torch.max(self._var, torch.full_like(self._var, 1e-2)))
        return (x - self._mean) / stdev


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            _conv3x3(inplanes, planes, stride),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            _conv3x3(planes, planes),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.convs(x) + residual)


class ResNet18(nn.Module):
    def __init__(self, in_channels: int, base_planes: int = 32, ngroups: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = base_planes
        self.layer1 = self._make_layer(base_planes, 2, ngroups)
        self.layer2 = self._make_layer(base_planes * 2, 2, ngroups, stride=2)
        self.layer3 = self._make_layer(base_planes * 4, 2, ngroups, stride=2)
        self.layer4 = self._make_layer(base_planes * 8, 2, ngroups, stride=2)
        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2**5)

    def _make_layer(
        self, planes: int, blocks: int, ngroups: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.GroupNorm(ngroups, planes),
            )
        layers = [BasicBlock(self.inplanes, planes, ngroups, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, ngroups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().view(x.size(0), -1).contiguous()


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        discretized_depth_channels: int = 10,
    ):
        super().__init__()
        # [prev/cur] x [rgb(3), depth(1), discretized depth(10), top-down(1)]
        input_channels = 2 * (3 + 1 + discretized_depth_channels + 1)
        self.running_mean_and_var = RunningMeanAndVar(input_channels)
        self.backbone = ResNet18(input_channels, base_planes=32, ngroups=16)

        width, height = image_size
        final_width = int(np.ceil(width * self.backbone.final_spatial_compress))
        final_height = int(np.ceil(height * self.backbone.final_spatial_compress))
        compression_channels = int(round(2048 / (final_width * final_height)))
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, compression_channels),
            nn.ReLU(True),
        )
        self.output_shape = (compression_channels, final_height, final_width)

    def forward(self, observation_pairs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        paired_inputs = []
        for key in ("rgb", "depth", "discretized_depth", "top_down_view"):
            value = observation_pairs[key].permute(0, 3, 1, 2)
            if key == "rgb":
                value = value / 255.0
            split = value.size(1) // 2
            paired_inputs.append([value[:, :split], value[:, split:]])

        # Preserve the PointNav-VO input order:
        # prev rgb/depth/discretized-depth/top-down, then current equivalents.
        cnn_input = [item for pair in zip(*paired_inputs) for item in pair]
        x = torch.cat(cnn_input, dim=1)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        return self.compression(x)


class VisualOdometryCNN(nn.Module):
    def __init__(self, image_size: Tuple[int, int], discretized_depth_channels: int = 10):
        super().__init__()
        self.visual_encoder = ResNetEncoder(image_size, discretized_depth_channels)
        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(int(np.prod(self.visual_encoder.output_shape)), 512),
            nn.ReLU(True),
        )
        self.output_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 3))

    def forward(self, observation_pairs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.output_head(self.visual_fc(self.visual_encoder(observation_pairs)))


class NormalizedDepthToTopDownView:
    def __init__(
        self,
        min_depth: float,
        max_depth: float,
        image_height: int,
        image_width: int,
        hfov: float,
        rows_around_center: int = 50,
    ):
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.height = int(image_height)
        self.width = int(image_width)
        self.rows_around_center = int(rows_around_center)
        self.epsilon = 0.01

        u0 = self.width / 2
        v0 = self.height / 2
        # PointNav-VO uses this value directly, despite the original variable name
        # being hfov_rad. Keep the behavior for checkpoint compatibility.
        focal = (self.width / 2) / np.tan(float(hfov) / 2)
        self.k_inv = np.linalg.inv(
            np.array([[focal, 0, u0], [0, focal, v0], [0, 0, 1.0]])
        )

    def __call__(self, normalized_depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(normalized_depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        nonzero = np.argwhere(depth > 0)
        if nonzero.size == 0:
            return np.zeros((self.height, self.width, 1), dtype=np.float32)

        min_row, min_col = nonzero.min(axis=0)
        max_row, max_col = nonzero.max(axis=0)
        cropped = depth[min_row : max_row + 1, min_col : max_col + 1]
        cropped = cv2.GaussianBlur(
            cropped,
            (3, 3),
            sigmaX=0,
            sigmaY=0,
            borderType=cv2.BORDER_ISOLATED,
        )

        row_start = max(0, int(np.ceil(cropped.shape[0] / 2)) - self.rows_around_center)
        row_end = min(
            cropped.shape[0],
            int(np.ceil(cropped.shape[0] / 2)) + self.rows_around_center,
        )
        valid_depth = cropped[row_start:row_end]
        v_coords, u_coords = np.meshgrid(
            np.arange(valid_depth.shape[0]),
            np.arange(valid_depth.shape[1]),
            indexing="ij",
        )
        u_coords = u_coords.reshape(-1).astype(np.float32) + float(min_col) + 0.5
        v_coords = v_coords.reshape(-1).astype(np.float32) + 0.5
        homogeneous = np.array([u_coords, v_coords, np.ones_like(u_coords)])
        true_depth = (
            valid_depth.reshape(-1) * (self.max_depth - self.min_depth) + self.min_depth
        )
        coords_3d = self.k_inv @ homogeneous
        coords_3d *= true_depth
        coords_2d = coords_3d[[0, 2]]

        rightmost = np.array([self.width - 0.5, 0, 1.0])
        max_x = float((self.k_inv @ rightmost)[0] * self.max_depth)
        min_x = -max_x
        x_range = max_x - min_x

        ndc_x = (coords_2d[0] - min_x) / (x_range * (1 + self.epsilon))
        ndc_y = (coords_2d[1] - self.min_depth) / (
            (self.max_depth - self.min_depth) * (1 + self.epsilon)
        )
        pixel_rows = self.height - np.ceil(self.height * ndc_y)
        pixel_cols = np.floor(self.width * ndc_x)
        pixel_rows = pixel_rows.astype(np.int64)
        pixel_cols = pixel_cols.astype(np.int64)
        valid = (
            (pixel_rows >= 0)
            & (pixel_rows < self.height)
            & (pixel_cols >= 0)
            & (pixel_cols < self.width)
        )

        top_down = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(top_down, (pixel_rows[valid], pixel_cols[valid]), 1.0)
        max_count = float(top_down.max())
        if max_count > 0:
            top_down /= max_count
        return top_down[:, :, None]


def _legacy_checkpoint_load(path: str, device: torch.device) -> Dict[str, object]:
    """Load old PointNav-VO checkpoints without importing old Habitat."""
    import habitat.config.default as habitat_config_default

    had_config = hasattr(habitat_config_default, "Config")
    old_config = getattr(habitat_config_default, "Config", None)

    class LegacyConfig(dict):
        pass

    LegacyConfig.__module__ = "habitat.config.default"
    habitat_config_default.Config = LegacyConfig
    try:
        # These checkpoints are user-provided model files from PointNav-VO.
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        if had_config:
            habitat_config_default.Config = old_config
        else:
            delattr(habitat_config_default, "Config")


class PointNavVisualOdometry:
    """Estimate and integrate planar pose from adjacent RGB-D observations."""

    def __init__(self, cfg: PointNavVisualOdometryConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        image_size = (cfg.image_width, cfg.image_height)
        self.models = {
            action: VisualOdometryCNN(image_size, cfg.discretized_depth_channels).to(
                self.device
            )
            for action in SUPPORTED_ACTIONS
        }
        self._load_models()
        self.top_down = NormalizedDepthToTopDownView(
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            hfov=cfg.hfov,
        )
        self.reset()

    def _load_models(self) -> None:
        forward = _legacy_checkpoint_load(self.cfg.forward_checkpoint, self.device)
        turns = _legacy_checkpoint_load(self.cfg.turn_checkpoint, self.device)
        state_by_action = {
            MOVE_FORWARD: forward["model_states"][MOVE_FORWARD],
            TURN_LEFT: turns["model_states"][TURN_LEFT],
            TURN_RIGHT: turns["model_states"][TURN_RIGHT],
        }
        for action, model in self.models.items():
            model.load_state_dict(state_by_action[action])
            model.eval()

    def reset(self) -> None:
        self.xy = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.last_delta = np.zeros(3, dtype=np.float32)

    def pose(self) -> Tuple[np.ndarray, float]:
        return self.xy.copy(), float(self.heading)

    def update(
        self,
        prev_obs: Mapping[str, np.ndarray],
        cur_obs: Mapping[str, np.ndarray],
        action: int,
    ) -> np.ndarray:
        action = int(action)
        if action not in self.models:
            self.last_delta = np.zeros(3, dtype=np.float32)
            return self.last_delta.copy()

        observation_pairs = self._build_observation_pairs(prev_obs, cur_obs)
        with torch.no_grad():
            delta = self.models[action](observation_pairs)[0].cpu().numpy().astype(np.float32)
        self.last_delta = delta
        self._integrate(delta)
        return delta.copy()

    def _build_observation_pairs(
        self,
        prev_obs: Mapping[str, np.ndarray],
        cur_obs: Mapping[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        prev_rgb = self._validate_rgb(prev_obs[self.cfg.rgb_key])
        cur_rgb = self._validate_rgb(cur_obs[self.cfg.rgb_key])
        prev_depth = self._validate_depth(prev_obs[self.cfg.depth_key])
        cur_depth = self._validate_depth(cur_obs[self.cfg.depth_key])

        prev_dd = self._discretize_depth(prev_depth)
        cur_dd = self._discretize_depth(cur_depth)
        prev_top = self.top_down(prev_depth)
        cur_top = self.top_down(cur_depth)

        def pair(first: np.ndarray, second: np.ndarray) -> torch.Tensor:
            value = np.concatenate([first, second], axis=2)
            return torch.from_numpy(value).to(self.device, dtype=torch.float32).unsqueeze(0)

        return {
            "rgb": pair(prev_rgb, cur_rgb),
            "depth": pair(prev_depth, cur_depth),
            "discretized_depth": pair(prev_dd, cur_dd),
            "top_down_view": pair(prev_top, cur_top),
        }

    def _validate_rgb(self, value: np.ndarray) -> np.ndarray:
        rgb = np.asarray(value)
        expected = (self.cfg.image_height, self.cfg.image_width, 3)
        if rgb.shape != expected:
            raise ValueError(f"{self.cfg.rgb_key} must have shape {expected}, got {rgb.shape}")
        return rgb.astype(np.float32, copy=False)

    def _validate_depth(self, value: np.ndarray) -> np.ndarray:
        depth = np.asarray(value, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[:, :, None]
        expected = (self.cfg.image_height, self.cfg.image_width, 1)
        if depth.shape != expected:
            raise ValueError(
                f"{self.cfg.depth_key} must have shape {expected}, got {depth.shape}"
            )
        return np.clip(depth, 0.0, 1.0)

    def _discretize_depth(self, depth: np.ndarray) -> np.ndarray:
        bins = np.floor(depth[:, :, 0] * self.cfg.discretized_depth_channels).astype(
            np.int64
        )
        bins = np.clip(bins, 0, self.cfg.discretized_depth_channels - 1)
        return np.eye(self.cfg.discretized_depth_channels, dtype=np.float32)[bins]

    def _integrate(self, delta: np.ndarray) -> None:
        dx, dz, dyaw = [float(value) for value in delta]
        # PointNav-VO local coordinates are x-right, z-backward. VOCA uses
        # x-forward, y-left for its planar waypoint coordinates.
        local_forward_left = np.array([-dz, -dx], dtype=np.float32)
        c = np.cos(self.heading)
        s = np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        self.xy += rotation @ local_forward_left
        self.heading = float(np.arctan2(np.sin(self.heading + dyaw), np.cos(self.heading + dyaw)))
