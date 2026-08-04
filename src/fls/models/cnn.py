from __future__ import annotations

import torch
from torch import nn

from fls.models.blocks import conv_block


class SmallCNN(nn.Module):
    """Small CNN exposing early/middle/late/head block groups."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: int = 32, use_bn: bool = True) -> None:
        super().__init__()
        self.early = conv_block(in_channels, width, use_bn)
        self.middle = nn.Sequential(conv_block(width, width * 2, use_bn), conv_block(width * 2, width * 2, use_bn))
        self.late = conv_block(width * 2, width * 4, use_bn)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width * 4, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features: dict[str, torch.Tensor] = {}
        x = self.early(x)
        features["early"] = x
        x = self.middle(x)
        features["middle"] = x
        x = self.late(x)
        features["late"] = x
        x = self.pool(x).flatten(1)
        features["head_input"] = x
        return x, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward_features(x)
        return self.head(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, features = self.forward_features(x)
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        return {"early": [self.early], "middle": [self.middle], "late": [self.late], "head": [self.head]}

