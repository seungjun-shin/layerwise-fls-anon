from __future__ import annotations

import warnings

import torch
from torch import nn


class OutputScaledBlock(nn.Module):
    """Multiply a block output by a fixed scalar as an operational FLS intervention."""

    def __init__(self, block: nn.Module, multiplier: float) -> None:
        super().__init__()
        self.block = block
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * self.block(x)


class InputScaledBlock(nn.Module):
    """Multiply a block input by a fixed scalar before applying the wrapped block."""

    def __init__(self, block: nn.Module, multiplier: float) -> None:
        super().__init__()
        self.block = block
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.multiplier * x)


def scale_module_initialization(module: nn.Module, scale: float) -> None:
    """Scale weight parameters in a module in-place."""
    if scale == 1.0:
        return
    for param in module.parameters():
        if param.ndim >= 2:
            param.data.mul_(scale)


def _replace_attr(model: nn.Module, name: str, wrapper: nn.Module) -> None:
    setattr(model, name, wrapper)


def apply_blockwise_scaling(model: nn.Module, profile: dict[str, dict], mode: str = "output") -> nn.Module:
    """Apply group-level scaling to direct model attributes.

    For residual networks this intentionally scales group outputs rather than residual branches;
    the induced FLS effect remains empirical and should be measured with diagnostics.
    """
    if "resnet" in model.__class__.__name__.lower():
        warnings.warn("Scaling ResNet groups at group outputs; do not interpret as intrinsic layer-wise FLS.", stacklevel=2)
    for group, values in profile.items():
        if not hasattr(model, group):
            continue
        module = getattr(model, group)
        init_scale = float(values.get("init_scale", 1.0))
        output_multiplier = float(values.get("output_multiplier", 1.0))
        scale_module_initialization(module, init_scale)
        wrapper = InputScaledBlock(module, output_multiplier) if mode == "input" else OutputScaledBlock(module, output_multiplier)
        _replace_attr(model, group, wrapper)
    return model

