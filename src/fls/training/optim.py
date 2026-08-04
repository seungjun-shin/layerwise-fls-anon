from __future__ import annotations

from typing import Any

import torch

REGION_ORDER = ["early", "middle", "late", "head"]
BOUNDARY_ORDER = ["after_early", "after_middle", "after_late", "after_pool"]
BOUNDARY_TO_REGION_INDEX = {
    "after_early": 0,
    "after_middle": 1,
    "after_late": 2,
    # Pooling has no trainable parameters.  A classifier-input multiplier still
    # affects all representation regions while leaving the head at base LR.
    "after_pool": 2,
}


def location_compensation_multipliers(config: dict[str, Any]) -> dict[str, float]:
    """Return boundary multipliers used for location LR compensation.

    By default, compensation follows the actual boundary output multipliers.
    The optional training.location_lr_compensation_multipliers field supports
    LR-only controls where activation scaling remains identity.
    """
    fls_cfg = config.get("fls", {})
    boundaries = fls_cfg.get("boundaries") or {}
    override = (config.get("training") or {}).get("location_lr_compensation_multipliers")
    source = override if isinstance(override, dict) else boundaries
    multipliers: dict[str, float] = {}
    for boundary in BOUNDARY_ORDER:
        value = source.get(boundary, 1.0)
        if isinstance(value, dict):
            value = value.get("output_multiplier", 1.0)
        multipliers[boundary] = float(value)
    return multipliers


def _blockwise_group_lrs(model: torch.nn.Module, config: dict[str, Any], base_lr: float) -> list[dict[str, Any]] | None:
    """Build parameter groups with upstream compensation for block-wise output scaling.

    A multiplier applied after a group scales gradients for that group and all upstream
    groups. The cumulative compensation uses lr / product(active multipliers from the
    current group to the head), while downstream groups keep the base lr unless affected
    by their own later multipliers.
    """
    fls_cfg = config.get("fls", {})
    compensation = config.get("training", {}).get("blockwise_lr_compensation")
    if fls_cfg.get("mode") != "blockwise" or not compensation:
        return None
    if compensation not in {True, "upstream_cumulative"}:
        raise ValueError(f"Unknown blockwise_lr_compensation mode: {compensation}")
    groups = fls_cfg.get("groups") or {}
    multipliers = {
        group: float((groups.get(group) or {}).get("output_multiplier", 1.0))
        for group in REGION_ORDER
    }
    param_groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for index, group in enumerate(REGION_ORDER):
        if not hasattr(model, group):
            continue
        cumulative = 1.0
        for downstream_group in REGION_ORDER[index:]:
            cumulative *= multipliers[downstream_group]
        if cumulative == 0.0:
            raise ValueError("Blockwise lr compensation requires nonzero output multipliers.")
        params = [param for param in getattr(model, group).parameters() if param.requires_grad]
        if not params:
            continue
        assigned.update(id(param) for param in params)
        param_groups.append({"params": params, "lr": base_lr / cumulative, "name": group, "lr_scale": 1.0 / cumulative})
    remaining = [param for param in model.parameters() if param.requires_grad and id(param) not in assigned]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr, "name": "remaining", "lr_scale": 1.0})
    return param_groups


def _location_region_lrs(model: torch.nn.Module, config: dict[str, Any], base_lr: float) -> list[dict[str, Any]] | None:
    """Build parameter groups with upstream compensation for scaling boundaries."""
    fls_cfg = config.get("fls", {})
    compensation = config.get("training", {}).get("location_lr_compensation")
    if fls_cfg.get("mode") != "location" or not compensation:
        return None
    if compensation not in {True, "upstream"}:
        raise ValueError(f"Unknown location_lr_compensation mode: {compensation}")
    multipliers = location_compensation_multipliers(config)
    param_groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for region_index, region in enumerate(REGION_ORDER):
        if not hasattr(model, region):
            continue
        cumulative = 1.0
        for boundary, boundary_index in BOUNDARY_TO_REGION_INDEX.items():
            if region_index <= boundary_index:
                cumulative *= multipliers[boundary]
        if cumulative == 0.0:
            raise ValueError("Location lr compensation requires nonzero output multipliers.")
        params = [param for param in getattr(model, region).parameters() if param.requires_grad]
        if not params:
            continue
        assigned.update(id(param) for param in params)
        param_groups.append({"params": params, "lr": base_lr / cumulative, "name": region, "lr_scale": 1.0 / cumulative})
    remaining = [param for param in model.parameters() if param.requires_grad and id(param) not in assigned]
    if remaining:
        param_groups.append({"params": remaining, "lr": base_lr, "name": "remaining", "lr_scale": 1.0})
    return param_groups


def _position_region_lrs(model: torch.nn.Module, config: dict[str, Any], base_lr: float) -> list[dict[str, Any]] | None:
    """Upstream LR compensation for a depth-POSITION boundary.

    Blocks 0..cut (upstream of the multiplier) get base_lr / mult; the remaining
    blocks plus the head stay at base_lr. Mirrors the location convention where
    only upstream-of-the-boundary params are compensated.
    """
    fls_cfg = config.get("fls", {})
    if fls_cfg.get("mode") != "position":
        return None
    if not config.get("training", {}).get("position_lr_compensation"):
        return None
    base = getattr(model, "base", model)
    blocks = base.ordered_blocks()
    cut = int(fls_cfg.get("position", len(blocks) - 1))
    mult = float(fls_cfg.get("output_multiplier", 1.0))
    # The upstream LR boost normally tracks the activation multiplier. An optional
    # ``lr_compensation_multiplier`` decouples the two so the LR pattern of a given
    # multiplier can be reproduced with identity activations (output_multiplier=1.0),
    # i.e. a per-cut LR-only control. Defaults to ``mult`` (legacy coupled behavior).
    lr_mult = float(fls_cfg.get("lr_compensation_multiplier", mult))
    if lr_mult == 0.0:
        raise ValueError("Position lr compensation requires nonzero multiplier.")
    upstream_ids: set[int] = set()
    for blk in blocks[: cut + 1]:
        upstream_ids.update(id(p) for p in blk.parameters() if p.requires_grad)
    up = [p for p in model.parameters() if p.requires_grad and id(p) in upstream_ids]
    down = [p for p in model.parameters() if p.requires_grad and id(p) not in upstream_ids]
    groups: list[dict[str, Any]] = []
    if up:
        groups.append({"params": up, "lr": base_lr / lr_mult, "name": "upstream", "lr_scale": 1.0 / lr_mult})
    if down:
        groups.append({"params": down, "lr": base_lr, "name": "downstream", "lr_scale": 1.0})
    return groups


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build the configured optimizer."""
    cfg = config["training"]
    lr = float(cfg["lr"])
    if config["fls"].get("mode") == "global" and config["fls"].get("global", {}).get("lr_compensation", False):
        global_cfg = config["fls"]["global"]
        compensation_multiplier = float(
            global_cfg.get("lr_compensation_multiplier", global_cfg.get("output_multiplier", 1.0))
        )
        if compensation_multiplier == 0.0:
            raise ValueError("Global lr compensation requires a nonzero multiplier.")
        lr = lr / compensation_multiplier
    name = cfg.get("optimizer", "sgd").lower()
    params = (_position_region_lrs(model, config, lr) or _location_region_lrs(model, config, lr)
              or _blockwise_group_lrs(model, config, lr) or model.parameters())
    head_lr_multiplier = cfg.get("head_lr_multiplier")
    if head_lr_multiplier is not None and isinstance(params, list):
        factor = float(head_lr_multiplier)
        for group in params:
            if group.get("name") == "head":
                group["lr"] = group["lr"] * factor
                group["lr_scale"] = group.get("lr_scale", 1.0) * factor
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.0)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
    raise ValueError(f"Unknown optimizer: {name}")
