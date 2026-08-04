from __future__ import annotations

import warnings

import torch
from torch import nn

BOUNDARY_TO_MODULE = {
    "after_early": "early",
    "after_middle": "middle",
    "after_late": "late",
    "after_pool": "pool",
}


class OutputScaledRegion(nn.Module):
    """Multiply the output at a fixed network boundary."""

    def __init__(self, region: nn.Module, multiplier: float) -> None:
        super().__init__()
        self.region = region
        self.multiplier = float(multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * self.region(x)


def scale_region_initialization(module: nn.Module, scale: float) -> None:
    """Scale weight parameters in a module in-place."""
    if scale == 1.0:
        return
    for param in module.parameters():
        if param.ndim >= 2:
            param.data.mul_(scale)


def apply_location_scaling(model: nn.Module, boundaries: dict[str, dict]) -> nn.Module:
    """Apply output scaling at named representation boundaries.

    A boundary named ``after_late`` means that the output of the model's late
    representation-producing region is multiplied. ``after_pool`` is the final
    feature-to-classifier interface: it scales pooled features before the head.
    These are operational interventions and are not intrinsic layer-wise FLS values.
    """
    if "resnet" in model.__class__.__name__.lower():
        warnings.warn(
            "Scaling is applied at region outputs; do not interpret boundaries as intrinsic layer-wise FLS.",
            stacklevel=2,
        )
    for boundary, values in boundaries.items():
        if boundary not in BOUNDARY_TO_MODULE:
            raise ValueError(f"Unknown scaling boundary: {boundary}")
        module_name = BOUNDARY_TO_MODULE[boundary]
        if not hasattr(model, module_name):
            raise ValueError(f"Model does not expose boundary module: {module_name}")
        module = getattr(model, module_name)
        init_scale = float(values.get("init_scale", 1.0))
        multiplier = float(values.get("output_multiplier", 1.0))
        scale_region_initialization(module, init_scale)
        setattr(model, module_name, OutputScaledRegion(module, multiplier))
    return model
