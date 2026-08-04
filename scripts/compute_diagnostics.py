#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from fls.data.datasets import build_dataset
from fls.diagnostics.alignment import mean_class_alignment
from fls.diagnostics.effective_rank import effective_rank
from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.profiles import build_profile
from fls.training.seed import set_seed
from fls.utils.device import resolve_device


REGIONS = ["early", "middle", "late", "head"]
GROUPS = REGIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute representation diagnostics for a saved run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing resolved_config.yaml and checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoint_last.pt", help="Checkpoint filename or absolute path.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=None, help="If set, sample a fixed random subset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_scaled_model(config: dict[str, Any]) -> torch.nn.Module:
    model = build_model(config)
    fls_cfg = config["fls"]
    if fls_cfg["mode"] == "global":
        return GlobalOutputMultiplier(model, fls_cfg.get("global", {}).get("output_multiplier", 1.0))
    if fls_cfg["mode"] == "location":
        return apply_location_scaling(model, fls_cfg.get("boundaries") or {})
    if fls_cfg["mode"] == "position":
        from fls.scaling.position_fls import apply_position_scaling

        return apply_position_scaling(
            model,
            int(fls_cfg.get("position", 0)),
            float(fls_cfg.get("output_multiplier", 1.0)),
        )
    if fls_cfg["mode"] == "blockwise":
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
    raise ValueError(f"Unknown FLS mode: {fls_cfg['mode']}")


def _flatten(features: torch.Tensor) -> torch.Tensor:
    return features.detach().float().reshape(features.shape[0], -1)


def linear_cka_samples(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute linear CKA using sample-space Gram matrices to avoid huge feature matrices."""
    x = _flatten(x)
    y = _flatten(y)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    k = k - k.mean(dim=0, keepdim=True) - k.mean(dim=1, keepdim=True) + k.mean()
    l = l - l.mean(dim=0, keepdim=True) - l.mean(dim=1, keepdim=True) + l.mean()
    numerator = (k * l).sum()
    denominator = torch.linalg.matrix_norm(k, ord="fro") * torch.linalg.matrix_norm(l, ord="fro")
    return float((numerator / (denominator + eps)).clamp(0.0, 1.0).item())


@torch.no_grad()
def collect_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    features: dict[str, list[torch.Tensor]] = {group: [] for group in GROUPS}
    y_clean_list: list[torch.Tensor] = []
    y_noisy_list: list[torch.Tensor] = []
    corrupted_list: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        x = batch["x"].to(device)
        logits, batch_features = model.forward_with_features(x)
        batch_features = {**batch_features, "head": logits}
        take = min(x.shape[0], max_samples - seen)
        for group in GROUPS:
            if group in batch_features:
                features[group].append(batch_features[group][:take].detach().cpu())
        y_clean = batch.get("y_clean", batch["y"])
        y_clean_list.append(torch.as_tensor(y_clean[:take], dtype=torch.long).cpu())
        y_noisy_list.append(torch.as_tensor(batch["y"][:take], dtype=torch.long).cpu())
        corrupted_list.append(torch.as_tensor(batch["is_corrupted"][:take], dtype=torch.bool).cpu())
        seen += take
        if seen >= max_samples:
            break
    merged = {group: torch.cat(values, dim=0) for group, values in features.items() if values}
    return merged, torch.cat(y_clean_list), torch.cat(y_noisy_list), torch.cat(corrupted_list)


def safe_alignment(features: torch.Tensor, labels: torch.Tensor) -> float:
    if features.shape[0] == 0 or labels.unique().numel() < 2:
        return 0.0
    return mean_class_alignment(features, labels)


def diagnostic_rows(
    initial_features: dict[str, torch.Tensor],
    current_features: dict[str, torch.Tensor],
    y_clean: torch.Tensor,
    y_noisy: torch.Tensor,
    corrupted: torch.Tensor,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        if group not in initial_features or group not in current_features:
            continue
        current = current_features[group]
        initial = initial_features[group]
        corrupted_current = current[corrupted]
        corrupted_noisy = y_noisy[corrupted]
        corrupted_clean = y_clean[corrupted]
        clean_alignment = safe_alignment(current, y_clean)
        noisy_alignment = safe_alignment(current, y_noisy)
        rows.append(
            {
                **metadata,
                "group": group,
                "num_samples": int(current.shape[0]),
                "num_corrupted": int(corrupted.sum().item()),
                "cka_to_initial": linear_cka_samples(initial, current),
                "cka_drift": 1.0 - linear_cka_samples(initial, current),
                "effective_rank": effective_rank(current),
                "feature_norm": float(_flatten(current).norm(dim=1).mean().item()),
                "clean_label_alignment": clean_alignment,
                "train_label_alignment": noisy_alignment,
                "alignment_gap_train_minus_clean": noisy_alignment - clean_alignment,
                "corrupted_clean_alignment": safe_alignment(corrupted_current, corrupted_clean),
                "corrupted_train_alignment": safe_alignment(corrupted_current, corrupted_noisy),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    set_seed(int(config["training"]["seed"]))
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
    dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    set_seed(int(config["training"]["seed"]))
    initial_model = build_scaled_model(config).to(device)
    initial_features, y_clean, y_noisy, corrupted = collect_features(initial_model, loader, device, args.max_samples)

    set_seed(int(config["training"]["seed"]))
    current_model = build_scaled_model(config).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current_model.load_state_dict(checkpoint["model"])
    current_features, _, _, _ = collect_features(current_model, loader, device, args.max_samples)

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "diagnostic_sample_policy": sample_policy,
        "diagnostic_sample_seed": args.sample_seed,
        "diagnostic_max_samples": args.max_samples,
        "experiment_name": config.get("experiment", {}).get("name"),
        "dataset": config.get("data", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "seed": config.get("training", {}).get("seed"),
        "fls_mode": config.get("fls", {}).get("mode"),
        "profile_name": config.get("fls", {}).get("profile_name"),
        "global_output_multiplier": config.get("fls", {}).get("global", {}).get("output_multiplier"),
        "location_boundaries": config.get("fls", {}).get("boundaries"),
        "label_noise_rate": config.get("label_noise", {}).get("rate"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
    }
    rows = diagnostic_rows(initial_features, current_features, y_clean, y_noisy, corrupted, metadata)
    if not rows:
        raise SystemExit("No diagnostic rows were produced; check model.forward_with_features output.")
    output = Path(args.output) if args.output else run_dir / "diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
