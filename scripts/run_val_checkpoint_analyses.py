#!/usr/bin/env python
"""Recompute paper diagnostics from validation-selected checkpoints.

This runner covers the core representation tables, diagnostic-subset checks,
VGG supportive diagnostics, practical-regime persistence, update pressure,
test-set Fisher diagnostics, CIFAR-100-C, and SVHN OOD evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/_val_preserve_all_20260801")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def latest_runs(experiment: str, seeds: set[int]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for run_dir in sorted(Path("outputs").glob(f"{experiment}/**/seed_*")):
        if not (run_dir / "checkpoint_best.pt").exists():
            continue
        try:
            seed = int(run_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed in seeds:
            found[seed] = run_dir
    missing = sorted(seeds - set(found))
    if missing:
        raise FileNotFoundError(f"Missing checkpoint seeds for {experiment}: {missing}")
    return found


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def compute_one(
    run_dir: Path,
    output: Path,
    device: str,
    sample_seed: int,
    max_samples: int,
    force: bool,
    log_path: Path,
) -> None:
    if output.exists() and not force:
        return
    run([
        sys.executable,
        "scripts/compute_diagnostics.py",
        "--run-dir", str(run_dir),
        "--checkpoint", "checkpoint_best.pt",
        "--split", "train",
        "--max-samples", str(max_samples),
        "--sample-seed", str(sample_seed),
        "--batch-size", "128",
        "--device", device,
        "--output", str(output),
    ], log_path)


def aggregate(files: list[tuple[str, Path]], output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for condition, path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault((condition, row["group"]), []).append(row)
    metrics = ["cka_drift", "effective_rank", "feature_norm", "clean_label_alignment", "train_label_alignment"]
    rows: list[dict[str, Any]] = []
    for (condition, group), records in sorted(grouped.items()):
        row: dict[str, Any] = {"condition": condition, "group": group, "n": len(records)}
        for metric in metrics:
            values = [float(record[metric]) for record in records]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    diag_root = args.root / "diagnostics"
    log_path = diag_root / "analysis.log"
    diag_root.mkdir(parents=True, exist_ok=True)

    core = {
        "global": "val60_core_global_c0p25",
        "identity": "val60_core_identity",
        "boundary_cut2": "val60_core_boundary_cut2",
        "boundary_cut8": "val60_core_boundary_cut8",
        "lr_only_cut8": "val60_core_lronly_cut8",
        "no_comp_cut8": "val60_core_nocomp_cut8",
    }
    core_files: list[tuple[str, Path]] = []
    for condition, experiment in core.items():
        for seed, run_dir in latest_runs(experiment, set(range(5))).items():
            output = run_dir / "diagnostics_valsel_seed2001.csv"
            compute_one(run_dir, output, args.device, 2001, 512, args.force, log_path)
            core_files.append((condition, output))
    aggregate(core_files, diag_root / "core_representation_summary.csv")

    subset_files: list[tuple[str, Path]] = []
    for condition, experiment in {"global": core["global"], "boundary_cut8": core["boundary_cut8"]}.items():
        for _, run_dir in latest_runs(experiment, set(range(5))).items():
            for sample_seed in [1001, 1002, 1003]:
                output = run_dir / f"diagnostics_valsel_subset{sample_seed}.csv"
                compute_one(run_dir, output, args.device, sample_seed, 512, args.force, log_path)
                subset_files.append((f"{condition}_subset{sample_seed}", output))
    aggregate(subset_files, diag_root / "core_subset_robustness_summary.csv")

    vgg = {
        "vgg_global": "val_vgg_global_c0p25_rep",
        "vgg_boundary": "val_vgg_bnd_cut15_rep",
        "vgg_lr_exact16x": "val_vgg_lr_cut15_16x_rep",
        "vgg_lr_selected1p75x": "val_vgg_lr_cut15_1p75x_rep",
    }
    vgg_files: list[tuple[str, Path]] = []
    for condition, experiment in vgg.items():
        for _, run_dir in latest_runs(experiment, {3, 4, 5}).items():
            output = run_dir / "diagnostics_valsel_seed2001.csv"
            compute_one(run_dir, output, args.device, 2001, 512, args.force, log_path)
            vgg_files.append((condition, output))
    aggregate(vgg_files, diag_root / "vgg_representation_summary.csv")

    selection = json.loads((args.root / "validation_selection.json").read_text(encoding="utf-8"))
    practical_files: list[tuple[str, Path]] = []
    practical_seed0 = {
        "practical_global": selection["practical_global"]["name"],
        "practical_boundary": selection["practical_boundary"]["name"],
    }
    for condition, experiment in practical_seed0.items():
        run_dir = latest_runs(experiment, {0})[0]
        output = run_dir / "diagnostics_valsel_seed2001.csv"
        compute_one(run_dir, output, args.device, 2001, 512, args.force, log_path)
        practical_files.append((condition, output))
    for condition, experiment in {
        "practical_global": "valp_practical_global_locked",
        "practical_boundary": "valp_practical_boundary_locked",
    }.items():
        for _, run_dir in latest_runs(experiment, {1, 2, 3, 4}).items():
            output = run_dir / "diagnostics_valsel_seed2001.csv"
            compute_one(run_dir, output, args.device, 2001, 512, args.force, log_path)
            practical_files.append((condition, output))
    aggregate(practical_files, diag_root / "practical_representation_summary.csv")

    run([
        sys.executable, "scripts/compute_update_pressure_diagnostics.py",
        "--preset", "confound4way",
        "--checkpoint", "checkpoint_best.pt",
        "--split", "train",
        "--max-samples", "256",
        "--sample-seed", "2001",
        "--device", args.device,
        "--output-dir", str(diag_root / "update_pressure"),
    ], log_path)

    run([
        sys.executable, "scripts/compute_discriminating_diagnostic.py",
        "--protocol", "val60",
        "--split", "test",
        "--max-samples", "5000",
        "--device", args.device,
        "--out", str(diag_root / "discriminating_diagnostic.md"),
    ], log_path)

    run([
        sys.executable, "scripts/eval_robustness.py",
        "--experiment", "val60_core_global_c0p25",
        "--experiment", "val60_core_boundary_cut8",
        "--seeds", "0", "1", "2", "3", "4",
        "--device", args.device,
        "--batch-size", "256",
        "--data-root", "data",
        "--no-download",
    ], log_path)

    (diag_root / "COMPLETED").write_text("all validation-selected checkpoint analyses completed\n", encoding="utf-8")
    print(f"Completed validation-selected analyses in {diag_root}")


if __name__ == "__main__":
    main()
