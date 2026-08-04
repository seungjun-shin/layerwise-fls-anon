from __future__ import annotations

import torch
from torch import nn


class PositionOutputMultiplier(nn.Module):
    """Parameter-free multiplier exposed as a hookable position boundary."""

    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * x


class PositionScaledResNet(nn.Module):
    """Apply an output multiplier at an arbitrary depth POSITION.

    The network is treated as a depth-ordered sequence of blocks
    (``base.ordered_blocks()``). After block index ``cut`` the activation is
    multiplied by ``mult``; everything up to and including ``cut`` is the
    upstream region (eligible for learning-rate compensation), the rest plus the
    head is downstream. Parameters are shared with ``base`` (no duplication).
    """

    def __init__(self, base: nn.Module, cut: int, mult: float) -> None:
        super().__init__()
        self.base = base
        self.cut = int(cut)
        self.mult = float(mult)
        self.scale = PositionOutputMultiplier(self.mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and coarse stage features after applying the cut multiplier."""
        features: dict[str, torch.Tensor] = {}
        blocks = self.base.ordered_blocks()
        stage_ends_by_depth = {
            9: {2: "early", 4: "middle", 8: "late"},
            16: {3: "early", 7: "middle", 15: "late"},
        }
        if len(blocks) not in stage_ends_by_depth:
            raise ValueError(f"Unsupported ordered depth for coarse features: {len(blocks)}")
        stage_ends = stage_ends_by_depth[len(blocks)]
        for i, block in enumerate(blocks):
            x = block(x)
            if i == self.cut:
                x = self.scale(x)
            stage = stage_ends.get(i)
            if stage is not None:
                features[stage] = x
        x = self.base.pool(x).flatten(1)
        if hasattr(self.base, "pre_head_norm"):
            x = self.base.pre_head_norm(x)
        logits = self.base.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        """Expose the wrapped model's coarse parameter groups for diagnostics."""
        return self.base.get_block_groups()

    def ordered_blocks(self) -> list[nn.Module]:
        """Expose the wrapped model's ordered body blocks."""
        return self.base.ordered_blocks()


def apply_position_scaling(model: nn.Module, position: int, multiplier: float) -> PositionScaledResNet:
    if not hasattr(model, "ordered_blocks"):
        raise ValueError("Position scaling requires a model exposing ordered_blocks().")
    n = len(model.ordered_blocks())
    if not (0 <= int(position) < n):
        raise ValueError(f"position {position} out of range [0,{n - 1}]")
    return PositionScaledResNet(model, int(position), float(multiplier))
