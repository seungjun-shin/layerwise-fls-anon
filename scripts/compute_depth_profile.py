#!/usr/bin/env python
"""Per-block (depth-axis) representation profile for a saved run.

Companion to ``compute_diagnostics.py``. Where that script reports diagnostics
at the four coarse early/middle/late/head groups, this one walks the model's
``ordered_blocks()`` and reports CKA drift from initialization, effective rank,
class alignment, and feature norm at *every* depth block. The result is a
continuous depth profile that characterizes where in the network representations
move under a scaling intervention.

CKA, effective rank, and (cosine-based) class alignment are invariant to the
isotropic activation multiplier inserted by boundary/position scaling, so the
profile is read off the underlying base model's block outputs without distortion;
feature norm is reported for completeness and does scale with the multiplier
downstream of the cut.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fls.data.datasets import build_dataset
from fls.diagnostics.alignment import mean_class_alignment
from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.profiles import build_profile
from fls.training.seed import set_seed
from fls.utils.device import resolve_device

REGIONS = ["early", "middle", "late", "head"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-block depth profile of representation diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Run directory with resolved_config.yaml and checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt", help="Checkpoint filename or absolute path.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=None, help="If set, sample a fixed random subset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Output CSV (default: <run-dir>/depth_profile.csv).")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_scaled_model(config: dict[str, Any]) -> torch.nn.Module:
    """Mirror scripts/compute_diagnostics.py so checkpoints load into the same structure."""
    model = build_model(config)
    fls_cfg = config["fls"]
    mode = fls_cfg["mode"]
    if mode == "global":
        return GlobalOutputMultiplier(model, fls_cfg.get("global", {}).get("output_multiplier", 1.0))
    if mode == "location":
        return apply_location_scaling(model, fls_cfg.get("boundaries") or {})
    if mode == "position":
        from fls.scaling.position_fls import apply_position_scaling

        return apply_position_scaling(model, int(fls_cfg.get("position", 0)), float(fls_cfg.get("output_multiplier", 1.0)))
    if mode == "blockwise":
        profile = fls_cfg.get("groups")
        if profile is None:
            profile = build_profile(
                fls_cfg.get("profile_name", "progressive_increasing"),
                REGIONS,
                fls_cfg.get("base_values"),
                seed=int(config["training"]["seed"]),
            )
            fls_cfg["groups"] = profile
        return apply_blockwise_scaling(model, profile)
    raise ValueError(f"Unknown FLS mode: {mode}")


def _flatten(features: torch.Tensor) -> torch.Tensor:
    return features.detach().float().reshape(features.shape[0], -1)


def linear_cka_samples(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    eps: float = 1e-12,
) -> float:
    """Linear CKA via sample-space Gram matrices (avoids materializing D x D)."""
    x = _flatten(x).to(device)
    y = _flatten(y).to(device)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    k = k - k.mean(dim=0, keepdim=True) - k.mean(dim=1, keepdim=True) + k.mean()
    l = l - l.mean(dim=0, keepdim=True) - l.mean(dim=1, keepdim=True) + l.mean()
    numerator = (k * l).sum()
    denominator = torch.linalg.matrix_norm(k, ord="fro") * torch.linalg.matrix_norm(l, ord="fro")
    return float((numerator / (denominator + eps)).clamp(0.0, 1.0).item())


def safe_alignment(features: torch.Tensor, labels: torch.Tensor) -> float:
    if features.shape[0] == 0 or labels.unique().numel() < 2:
        return 0.0
    return mean_class_alignment(features, labels)


def effective_rank_samples(
    features: torch.Tensor, device: torch.device, eps: float = 1e-12
) -> float:
    """Compute effective rank through the sample Gram matrix.

    The nonzero singular values of the centered feature matrix are the square
    roots of the eigenvalues of its sample Gram matrix.  This equivalent form
    avoids a costly SVD of a very wide activation matrix.
    """

    x = _flatten(features).to(device)
    x = x - x.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(x @ x.T).clamp_min(0.0)
    singular_values = eigenvalues.sqrt()
    probs = singular_values / (singular_values.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def collect_block_features(
    base: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture the activation after every block in ``base.ordered_blocks()``."""
    base.eval()
    blocks = base.ordered_blocks()
    feats: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(blocks))}
    y_clean_list: list[torch.Tensor] = []
    y_noisy_list: list[torch.Tensor] = []
    corrupted_list: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        x = batch["x"].to(device)
        take = min(x.shape[0], max_samples - seen)
        h = x
        for i, blk in enumerate(blocks):
            h = blk(h)
            feats[i].append(h[:take].detach().cpu())
        y_clean = batch.get("y_clean", batch["y"])
        y_clean_list.append(torch.as_tensor(y_clean[:take], dtype=torch.long).cpu())
        y_noisy_list.append(torch.as_tensor(batch["y"][:take], dtype=torch.long).cpu())
        corrupted_list.append(torch.as_tensor(batch["is_corrupted"][:take], dtype=torch.bool).cpu())
        seen += take
        if seen >= max_samples:
            break
    merged = {i: torch.cat(v, dim=0) for i, v in feats.items() if v}
    return merged, torch.cat(y_clean_list), torch.cat(y_noisy_list), torch.cat(corrupted_list)


def _base_of(model: torch.nn.Module) -> torch.nn.Module:
    base = model
    seen: set[int] = set()
    while not hasattr(base, "ordered_blocks"):
        if id(base) in seen:
            break
        seen.add(id(base))
        if hasattr(base, "base"):
            base = getattr(base, "base")
        elif hasattr(base, "model"):
            base = getattr(base, "model")
        else:
            break
    if not hasattr(base, "ordered_blocks"):
        raise SystemExit(f"Model {type(base).__name__} does not expose ordered_blocks(); depth profile unavailable.")
    return base


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    seed = int(config["training"]["seed"])
    device = torch.device(args.device) if args.device else resolve_device(config["training"].get("device", "auto"))

    dataset = build_dataset(config, train=args.split == "train")
    sample_count = min(args.max_samples, len(dataset))
    if args.sample_seed is None:
        indices = list(range(sample_count))
        sample_policy = "first_n"
    else:
        generator = torch.Generator().manual_seed(args.sample_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        sample_policy = "fixed_random"
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialization reference (same seed as training -> identical init).
    set_seed(seed)
    init_base = _base_of(build_scaled_model(config).to(device))
    init_feats, y_clean, y_noisy, corrupted = collect_block_features(init_base, loader, device, args.max_samples)

    # Trained model.
    set_seed(seed)
    model = build_scaled_model(config).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    trained_base = _base_of(model)
    cur_feats, _, _, _ = collect_block_features(trained_base, loader, device, args.max_samples)

    fls_cfg = config.get("fls", {})
    cut = fls_cfg.get("position")
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_path.name,
        "split": args.split,
        "diagnostic_sample_policy": sample_policy,
        "diagnostic_max_samples": args.max_samples,
        "experiment_name": config.get("experiment", {}).get("name"),
        "dataset": config.get("data", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "seed": seed,
        "fls_mode": fls_cfg.get("mode"),
        "cut_position": cut,
        "output_multiplier": fls_cfg.get("output_multiplier"),
        "num_blocks": len(init_feats),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }

    rows: list[dict[str, Any]] = []
    for i in sorted(init_feats):
        cur = cur_feats[i]
        init = init_feats[i]
        clean_alignment = safe_alignment(cur, y_clean)
        noisy_alignment = safe_alignment(cur, y_noisy)
        cka_to_initial = linear_cka_samples(init, cur, device)
        rows.append(
            {
                **metadata,
                "block_index": i,
                "depth_frac": round(i / (len(init_feats) - 1), 4) if len(init_feats) > 1 else 0.0,
                "is_after_cut": (cut is not None and i > int(cut)),
                "num_samples": int(cur.shape[0]),
                "cka_to_initial": cka_to_initial,
                "cka_drift": 1.0 - cka_to_initial,
                "effective_rank": effective_rank_samples(cur, device),
                "feature_norm": float(_flatten(cur).norm(dim=1).mean().item()),
                "clean_label_alignment": clean_alignment,
                "train_label_alignment": noisy_alignment,
                "alignment_gap_train_minus_clean": noisy_alignment - clean_alignment,
            }
        )

    output = Path(args.output) if args.output else run_dir / "depth_profile.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
