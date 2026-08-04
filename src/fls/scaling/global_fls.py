from __future__ import annotations

import torch
from torch import nn


class GlobalOutputMultiplier(nn.Module):
    """Wrap a model and multiply logits by a global output multiplier."""

    def __init__(self, model: nn.Module, multiplier: float) -> None:
        super().__init__()
        self.model = model
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * self.model(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, features = self.model.forward_with_features(x)
        return self.multiplier * logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        return self.model.get_block_groups()

