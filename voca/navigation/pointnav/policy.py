from typing import Any, Dict, Union

import torch
from torch import Tensor

from third_party.vlfm_pointnav.pointnav_policy import (
    WrappedPointNavResNetPolicy as _WrappedPointNavResNetPolicy,
)


class PointNavPolicy:
    """VOCA-facing wrapper around the vendored VLFM PointNav policy."""

    def __init__(self, checkpoint_path: str, device: Union[str, torch.device] = "cuda") -> None:
        self._policy = _WrappedPointNavResNetPolicy(checkpoint_path, device=device)

    def act(
        self,
        observations: Dict[str, Any],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        return self._policy.act(observations, masks, deterministic=deterministic)

    def reset(self) -> None:
        self._policy.reset()
