#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fls.data.datasets import build_dataset
from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.profiles import build_profile
from fls.training.losses import build_loss
from fls.training.optim import BOUNDARY_ORDER, BOUNDARY_TO_REGION_INDEX, REGION_ORDER, location_compensation_multipliers
from fls.training.seed import set_seed
from fls.training.trainer import apply_bn_controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-region gradient and update-pressure diagnostics.")
    parser.add_argument("--preset", choices=["main_pair", "compensation", "confound4way"], default="main_pair")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=2001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="outputs/update_pressure_diagnostics")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def close(value: float | None, target: float, tol: float = 1e-12) -> bool:
    return value is not None and abs(value - target) <= tol


def boundary_value(config: dict[str, Any], name: str) -> float | None:
    value = ((config.get("fls") or {}).get("boundaries") or {}).get(name)
    if not isinstance(value, dict):
        return None
    multiplier = value.get("output_multiplier")
    return None if multiplier is None else float(multiplier)


def global_value(config: dict[str, Any]) -> float | None:
    value = ((config.get("fls") or {}).get("global") or {}).get("output_multiplier")
    return None if value is None else float(value)


def position_value(config: dict[str, Any]) -> float | None:
    fls_cfg = config.get("fls") or {}
    if fls_cfg.get("mode") != "position":
        return None
    value = fls_cfg.get("output_multiplier")
    return None if value is None else float(value)


def condition_label(config: dict[str, Any], preset: str = "main_pair") -> str | None:
    exp_name = (config.get("experiment") or {}).get("name", "")
    seed = int((config.get("training") or {}).get("seed", -1))
    lr = float((config.get("training") or {}).get("lr", -1.0))
    noise = float((config.get("label_noise") or {}).get("rate", 0.0))
    if preset in {"compensation", "confound4way"}:
        if (
            (exp_name.startswith("experiment_40_boundary_late_seed_expansion") or exp_name == "val60_core_boundary_cut8")
            and seed in ({0, 1, 2, 3, 4} if preset == "confound4way" else {0, 1, 2})
            and close(lr, 0.04096)
            and (close(boundary_value(config, "after_late"), 0.0625) or close(position_value(config), 0.0625))
            and close(noise, 0.0)
        ):
            return "boundary_upstream_comp" if preset == "confound4way" else "main_upstream_comp"
        if (
            (exp_name.startswith("experiment_63_boundary_late_no_comp_ablation") or exp_name == "val60_core_nocomp_cut8")
            and seed in ({0, 1, 2, 3, 4} if preset == "confound4way" else {0, 1, 2})
            and close(lr, 0.04096)
            and (close(boundary_value(config, "after_late"), 0.0625) or close(position_value(config), 0.0625))
            and close(noise, 0.0)
        ):
            return "boundary_no_comp" if preset == "confound4way" else "no_comp"
        if preset == "compensation":
            if (
                exp_name.startswith("experiment_64_boundary_late_full_comp_ablation")
                and seed in {0, 1, 2}
                and close(lr, 0.65536)
                and close(boundary_value(config, "after_late"), 0.0625)
                and close(noise, 0.0)
            ):
                return "full_comp"
        if preset == "confound4way":
            if (
                (exp_name.startswith("experiment_49_global_matched_seed_expansion") or exp_name == "val60_core_global_c0p25")
                and seed in {0, 1, 2, 3, 4}
                and close(lr, 0.04096)
                and close(global_value(config), 0.25)
                and close(noise, 0.0)
            ):
                return "matched_global"
            if (
                (exp_name.startswith("experiment_75_boundary_late_lr_only_ablation") or exp_name == "val60_core_lronly_cut8")
                and seed in {0, 1, 2, 3, 4}
                and close(lr, 0.04096)
                and (close(boundary_value(config, "after_late"), 1.0) or close(position_value(config), 1.0))
                and close(noise, 0.0)
            ):
                return "lr_only"
        return None
    if (
        (exp_name.startswith("experiment_49_global_matched_seed_expansion") or exp_name == "val60_core_global_c0p25")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and close(global_value(config), 0.25)
        and close(noise, 0.0)
    ):
        return "matched_global"
    if (
        (exp_name.startswith("experiment_40_boundary_late_seed_expansion") or exp_name == "val60_core_boundary_cut8")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and (close(boundary_value(config, "after_late"), 0.0625) or close(position_value(config), 0.0625))
        and close(noise, 0.0)
    ):
        return "late_boundary"
    return None


def completed_preset_dirs(outputs_root: Path, checkpoint: str, preset: str) -> list[Path]:
    run_dirs: list[Path] = []
    for config_path in sorted(outputs_root.glob("**/resolved_config.yaml")):
        run_dir = config_path.parent
        if not (run_dir / "metrics.csv").exists() or not (run_dir / checkpoint).exists():
            continue
        config = load_yaml(config_path)
        if condition_label(config, preset) is not None:
            run_dirs.append(run_dir)
    return run_dirs


def region_lrs(config: dict[str, Any]) -> dict[str, float]:
    base_lr = float((config.get("training") or {}).get("lr", 0.0))
    fls_cfg = config.get("fls") or {}
    if fls_cfg.get("mode") == "global":
        multiplier = float((fls_cfg.get("global") or {}).get("output_multiplier", 1.0))
        if (fls_cfg.get("global") or {}).get("lr_compensation", False):
            base_lr = base_lr / multiplier
        return {region: base_lr for region in REGION_ORDER}
    if fls_cfg.get("mode") == "location" and (config.get("training") or {}).get("location_lr_compensation"):
        multipliers = location_compensation_multipliers(config)
        lrs: dict[str, float] = {}
        for region_index, region in enumerate(REGION_ORDER):
            cumulative = 1.0
            for boundary, boundary_index in BOUNDARY_TO_REGION_INDEX.items():
                if region_index <= boundary_index:
                    cumulative *= multipliers[boundary]
            lrs[region] = base_lr / cumulative
        return lrs
    if fls_cfg.get("mode") == "position" and (config.get("training") or {}).get("position_lr_compensation"):
        cut = int(fls_cfg.get("position", 0))
        multiplier = float(fls_cfg.get("lr_compensation_multiplier", fls_cfg.get("output_multiplier", 1.0)))
        stage_ends = {"early": 2, "middle": 4, "late": 8}
        lrs = {
            region: (base_lr / multiplier if stage_end <= cut else base_lr)
            for region, stage_end in stage_ends.items()
        }
        lrs["head"] = base_lr
        return lrs
    return {region: base_lr for region in REGION_ORDER}


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
                REGION_ORDER,
                fls_cfg.get("base_values"),
                seed=int(config["training"]["seed"]),
            )
            fls_cfg["groups"] = profile
        return apply_blockwise_scaling(model, profile)
    raise ValueError(f"Unknown FLS mode: {fls_cfg['mode']}")


def region_params(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups = model.get_block_groups()
    assigned: set[int] = set()
    output: dict[str, list[torch.nn.Parameter]] = {}
    for region in REGION_ORDER:
        params: list[torch.nn.Parameter] = []
        for module in groups.get(region, []):
            for param in module.parameters():
                if param.requires_grad and id(param) not in assigned:
                    params.append(param)
                    assigned.add(id(param))
        output[region] = params
    return output


def norm_from_tensors(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    total = sum(float(t.detach().float().pow(2).sum().item()) for t in tensors)
    return math.sqrt(total)


def batch_region_stats(
    model: torch.nn.Module,
    params_by_region: dict[str, list[torch.nn.Parameter]],
    lrs_by_region: dict[str, float],
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for region, params in params_by_region.items():
        grads = [param.grad for param in params if param.grad is not None]
        grad_norm = norm_from_tensors(grads)
        param_norm = norm_from_tensors(params)
        lr = lrs_by_region.get(region, 0.0)
        update_norm = lr * grad_norm
        rows[region] = {
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_norm": param_norm,
            "relative_update_norm": update_norm / max(param_norm, 1e-12),
            "effective_lr": lr,
        }
    return rows


def summarize_batches(batch_rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for region in REGION_ORDER:
        keys = ["grad_norm", "update_norm", "param_norm", "relative_update_norm", "effective_lr"]
        output[region] = {}
        for key in keys:
            values = [row[region][key] for row in batch_rows]
            output[region][f"{key}_mean"] = mean(values)
            output[region][f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    head_update = output["head"]["update_norm_mean"]
    head_relative = output["head"]["relative_update_norm_mean"]
    for region in REGION_ORDER:
        output[region]["update_norm_ratio_to_head"] = output[region]["update_norm_mean"] / max(head_update, 1e-12)
        output[region]["relative_update_ratio_to_head"] = output[region]["relative_update_norm_mean"] / max(head_relative, 1e-12)
    return output


def compute_run(run_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    config = load_yaml(run_dir / "resolved_config.yaml")
    set_seed(int(config["training"]["seed"]))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, train=args.split == "train")
    sample_count = min(args.max_samples, len(dataset))
    generator = torch.Generator().manual_seed(args.sample_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_scaled_model(config).to(device)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    apply_bn_controls(model, config)
    loss_fn = build_loss()
    params_by_region = region_params(model)
    lrs_by_region = region_lrs(config)

    batch_rows: list[dict[str, dict[str, float]]] = []
    seen = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = torch.as_tensor(batch["y"], device=device, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        batch_rows.append(batch_region_stats(model, params_by_region, lrs_by_region))
        seen += y.numel()
        if seen >= sample_count:
            break
    summary = summarize_batches(batch_rows)
    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "condition": condition_label(config, args.preset),
        "experiment_name": config.get("experiment", {}).get("name"),
        "dataset": config.get("data", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "seed": config.get("training", {}).get("seed"),
        "split": args.split,
        "sample_seed": args.sample_seed,
        "max_samples": args.max_samples,
        "num_batches": len(batch_rows),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "global_output_multiplier": global_value(config),
        "after_late_output_multiplier": boundary_value(config, "after_late"),
    }
    return [{**metadata, "region": region, **summary[region]} for region in REGION_ORDER]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["region"]))].append(row)
    metrics = [
        "grad_norm_mean",
        "update_norm_mean",
        "relative_update_norm_mean",
        "update_norm_ratio_to_head",
        "relative_update_ratio_to_head",
        "effective_lr_mean",
    ]
    output: list[dict[str, Any]] = []
    for (condition, region), group_rows in sorted(grouped.items()):
        row: dict[str, Any] = {"condition": condition, "region": region, "num_runs": len(group_rows)}
        for metric in metrics:
            values = [float(item[metric]) for item in group_rows]
            row[f"{metric}_across_runs_mean"] = mean(values)
            row[f"{metric}_across_runs_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main() -> None:
    args = parse_args()
    run_dirs = completed_preset_dirs(Path(args.outputs_root), args.checkpoint, args.preset)
    expected = {"main_pair": 10, "compensation": 9, "confound4way": 20}[args.preset]
    if len(run_dirs) != expected:
        raise SystemExit(f"Expected {expected} {args.preset} runs, found {len(run_dirs)}.")
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(compute_run(run_dir, args))
    output_dir = Path(args.output_dir)
    prefix = f"{args.preset}_update_pressure"
    write_csv(output_dir / f"{prefix}.csv", rows)
    write_csv(output_dir / f"{prefix}_summary.csv", aggregate_rows(rows))


if __name__ == "__main__":
    main()
