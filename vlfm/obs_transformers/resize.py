from dataclasses import dataclass
from typing import Tuple

from habitat_baselines.config.default_structured_configs import ObsTransformConfig


@dataclass
class ResizeConfig(ObsTransformConfig):
    type: str = "Resize"
    size: Tuple[int, int] = (224, 224)
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"
