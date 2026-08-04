from __future__ import annotations

import torch

from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.profiles import build_profile


def build_scaled_model(config: dict) -> torch.nn.Module:
    """Build a model and apply the configured operational scaling wrapper."""
    model = build_model(config)
    fls_cfg = config["fls"]
    mode = fls_cfg["mode"]
    if mode == "global":
        return GlobalOutputMultiplier(
            model,
            fls_cfg.get("global", {}).get("output_multiplier", 1.0),
        )
    if mode == "location":
        return apply_location_scaling(model, fls_cfg.get("boundaries") or {})
    if mode == "position":
        from fls.scaling.position_fls import apply_position_scaling

        return apply_position_scaling(
            model,
            int(fls_cfg.get("position", 0)),
            float(fls_cfg.get("output_multiplier", 1.0)),
        )
    if mode == "blockwise":
        groups = ["early", "middle", "late", "head"]
        profile = fls_cfg.get("groups")
        if profile is None:
            profile = build_profile(
                fls_cfg.get("profile_name", "progressive_increasing"),
                groups,
                fls_cfg.get("base_values"),
                seed=int(config["training"]["seed"]),
            )
            fls_cfg["groups"] = profile
        return apply_blockwise_scaling(model, profile)
    raise ValueError(f"Unknown FLS mode: {mode}")
