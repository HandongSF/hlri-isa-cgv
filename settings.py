import os
from pathlib import Path
from typing import Union


PathLike = Union[Path, str]


def _path_from_env(name: str, default: PathLike) -> str:
    return str(Path(os.getenv(name, str(default))).expanduser())


def _dir_from_env(name: str, default: PathLike) -> str:
    path = _path_from_env(name, default)
    return path if path.endswith(os.sep) else path + os.sep


PROJECT_ROOT = Path(__file__).resolve().parent

# Runtime defaults. Override these with VOCA_* environment variables as needed.
DEFAULT_DEVICE = os.getenv("VOCA_DEVICE", "cuda:0")
DEFAULT_CUDA_VISIBLE_DEVICES = os.getenv("VOCA_CUDA_VISIBLE_DEVICES", "0")

# Habitat and dataset paths.
HABITAT_ROOT_DIR = _path_from_env("VOCA_HABITAT_ROOT", "~/habitat-lab")
HM3D_CONFIG_PATH = _path_from_env(
    "VOCA_HM3D_CONFIG_PATH",
    f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml",
)
MP3D_CONFIG_PATH = _path_from_env(
    "VOCA_MP3D_CONFIG_PATH",
    f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml",
)
DATA_ROOT = _path_from_env("VOCA_DATA_ROOT", PROJECT_ROOT / "data")
SCENE_PREFIX = _dir_from_env("VOCA_SCENE_DATASETS_DIR", Path(DATA_ROOT) / "scene_datasets")
EPISODE_PREFIX = _dir_from_env("VOCA_DATASETS_DIR", Path(DATA_ROOT) / "datasets")

# Detection, segmentation, and policy checkpoints.
CHECKPOINT_DIR = _path_from_env("VOCA_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")
GROUNDING_DINO_CONFIG_PATH = _path_from_env(
    "VOCA_GROUNDING_DINO_CONFIG",
    Path(CHECKPOINT_DIR) / "GroundingDINO_SwinB_cfg.py",
)
GROUNDING_DINO_CHECKPOINT_PATH = _path_from_env(
    "VOCA_GROUNDING_DINO_CHECKPOINT",
    Path(CHECKPOINT_DIR) / "groundingdino_swinb_cogcoor.pth",
)
SAM_ENCODER_VERSION = os.getenv("VOCA_SAM_ENCODER_VERSION", "vit_h")
SAM_CHECKPOINT_PATH = _path_from_env("VOCA_SAM_CHECKPOINT", Path(CHECKPOINT_DIR) / "sam_vit_h_4b8939.pth")
POLICY_CHECKPOINT = _path_from_env("VOCA_POLICY_CHECKPOINT", Path(CHECKPOINT_DIR) / "pixelnav_A.ckpt")
POINTNAV_CHECKPOINT = _path_from_env("VOCA_POINTNAV_CHECKPOINT", Path(CHECKPOINT_DIR) / "pointnav_weights.pth")
YOLOE_CHECKPOINT_PATH = _path_from_env("VOCA_YOLOE_CHECKPOINT", Path(CHECKPOINT_DIR) / "yoloe-11l-seg.pt")
