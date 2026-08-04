#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute diagnostics for boundary-protocol checkpoint reruns.")
    parser.add_argument(
        "--preset",
        choices=["main_pair", "compensation", "confound4way", "representatives", "noise", "all"],
        default="main_pair",
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-suffix", default=None, help="Write diagnostics_<suffix>.csv instead of diagnostics.csv.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def boundary_value(config: dict[str, Any], name: str) -> float | None:
    value = ((config.get("fls") or {}).get("boundaries") or {}).get(name)
    if not isinstance(value, dict):
        return None
    multiplier = value.get("output_multiplier")
    return None if multiplier is None else float(multiplier)


def global_value(config: dict[str, Any]) -> float | None:
    value = ((config.get("fls") or {}).get("global") or {}).get("output_multiplier")
    return None if value is None else float(value)


def close(value: float | None, target: float, tol: float = 1e-12) -> bool:
    return value is not None and abs(value - target) <= tol


def matches_preset(config: dict[str, Any], preset: str) -> bool:
    exp_name = (config.get("experiment") or {}).get("name", "")
    seed = int((config.get("training") or {}).get("seed", -1))
    lr = float((config.get("training") or {}).get("lr", -1.0))
    noise = float((config.get("label_noise") or {}).get("rate", 0.0))
    is_main_global = (
        exp_name.startswith("experiment_49_global_matched_seed_expansion")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and close(global_value(config), 0.25)
        and close(noise, 0.0)
    )
    is_main_boundary = (
        exp_name.startswith("experiment_40_boundary_late_seed_expansion")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 0.0625)
        and close(noise, 0.0)
    )
    is_comp_main = (
        exp_name.startswith("experiment_40_boundary_late_seed_expansion")
        and seed in {0, 1, 2}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 0.0625)
        and close(noise, 0.0)
    )
    is_comp_no = (
        exp_name.startswith("experiment_63_boundary_late_no_comp_ablation")
        and seed in {0, 1, 2}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 0.0625)
        and close(noise, 0.0)
    )
    is_comp_full = (
        exp_name.startswith("experiment_64_boundary_late_full_comp_ablation")
        and seed in {0, 1, 2}
        and close(lr, 0.65536)
        and close(boundary_value(config, "after_late"), 0.0625)
        and close(noise, 0.0)
    )
    is_confound_global = is_main_global
    is_confound_boundary = is_main_boundary
    is_confound_no = (
        exp_name.startswith("experiment_63_boundary_late_no_comp_ablation")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 0.0625)
        and close(noise, 0.0)
    )
    is_confound_lr_only = (
        exp_name.startswith("experiment_75_boundary_late_lr_only_ablation")
        and seed in {0, 1, 2, 3, 4}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 1.0)
        and close(noise, 0.0)
    )
    is_representative = exp_name.startswith("diagnostic_boundary_representative_") and seed == 0 and close(noise, 0.0)
    is_noise_global = (
        exp_name.startswith("experiment_50_noise_global_matched")
        and seed in {0, 1, 2}
        and close(lr, 0.04096)
        and close(global_value(config), 0.25)
        and (close(noise, 0.2) or close(noise, 0.5))
    )
    is_noise_boundary = (
        exp_name.startswith("experiment_51_noise_boundary_late_matched")
        and seed in {0, 1, 2}
        and close(lr, 0.04096)
        and close(boundary_value(config, "after_late"), 0.0625)
        and (close(noise, 0.2) or close(noise, 0.5))
    )
    if preset == "main_pair":
        return is_main_global or is_main_boundary
    if preset == "compensation":
        return is_comp_main or is_comp_no or is_comp_full
    if preset == "confound4way":
        return is_confound_global or is_confound_boundary or is_confound_no or is_confound_lr_only
    if preset == "representatives":
        return is_representative
    if preset == "noise":
        return is_noise_global or is_noise_boundary
    return (
        is_main_global
        or is_main_boundary
        or is_comp_no
        or is_comp_full
        or is_representative
        or is_noise_global
        or is_noise_boundary
    )


def completed_run_dirs(outputs_root: Path, preset: str, checkpoint: str, force: bool) -> list[Path]:
    run_dirs: list[Path] = []
    for config_path in sorted(outputs_root.glob("**/resolved_config.yaml")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        checkpoint_path = run_dir / checkpoint
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        if not force and (run_dir / "diagnostics.csv").exists():
            continue
        config = load_yaml(config_path)
        if matches_preset(config, preset):
            run_dirs.append(run_dir)
    return run_dirs


def run_diagnostics(run_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/compute_diagnostics.py",
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.sample_seed is not None:
        command.extend(["--sample-seed", str(args.sample_seed)])
    if args.output_suffix:
        command.extend(["--output", str(run_dir / f"diagnostics_{args.output_suffix}.csv")])
    if args.device:
        command.extend(["--device", args.device])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    run_dirs = completed_run_dirs(Path(args.outputs_root), args.preset, args.checkpoint, args.force)
    if not run_dirs:
        print(f"No matching checkpoint runs found for preset={args.preset}.")
        return
    for run_dir in run_dirs:
        run_diagnostics(run_dir, args)
        print(run_dir)


if __name__ == "__main__":
    main()
