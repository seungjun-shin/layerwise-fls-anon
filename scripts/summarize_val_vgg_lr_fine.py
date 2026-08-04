#!/usr/bin/env python
"""Select the VGG19-BN LR-only comparator using validation metrics only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


LR_CONDITIONS = {
    1.0: "val_vgg_lr_cut15_1x_fine",
    1.25: "val_vgg_lr_cut15_1p25x_fine",
    1.5: "val_vgg_lr_cut15_1p5x_fine",
    1.75: "val_vgg_lr_cut15_1p75x_fine",
    2.0: "val_vgg_lr_cut15_2x_iso",
    4.0: "val_vgg_lr_cut15_4x_iso",
    8.0: "val_vgg_lr_cut15_8x_iso",
    16.0: "val_vgg_lr_cut15_16x_iso",
}
BOUNDARY_EXPERIMENT = "val_vgg_bnd_cut15_iso"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_EPOCHS = 80


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/_summaries/vgg_lr_fine_20260730"),
        type=Path,
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read one complete, finite metrics file."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_EPOCHS:
        raise ValueError(f"Expected {EXPECTED_EPOCHS} rows, found {len(rows)}: {path}")
    if int(float(rows[-1]["epoch"])) != EXPECTED_EPOCHS - 1:
        raise ValueError(f"Incomplete run: {path}")
    for row in rows:
        for field in (
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "val_loss",
            "val_accuracy",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} at epoch {row['epoch']}: {path}")
    return rows


def find_complete_run(outputs_root: Path, experiment: str, seed: int) -> Path:
    """Return the newest complete metrics file for an experiment and seed."""
    candidates = sorted(
        outputs_root.glob(f"{experiment}/*/seed_{seed}/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            read_metrics(path)
        except ValueError:
            continue
        return path
    raise FileNotFoundError(f"No complete run for {experiment}, seed {seed}")


def validation_checkpoint(rows: list[dict[str, str]]) -> dict[str, str]:
    """Select the earliest checkpoint attaining maximum validation accuracy."""
    return max(rows, key=lambda row: float(row["val_accuracy"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float]:
    """Return a two-sided 95% t interval."""
    avg = mean(values)
    half_width = float(stats.t.ppf(0.975, len(values) - 1)) * stdev(values) / math.sqrt(
        len(values)
    )
    return avg - half_width, avg + half_width


def main() -> None:
    """Select an LR-only multiplier by validation, then report its test contrast."""
    args = parse_args()
    per_multiplier: dict[float, dict[int, dict[str, object]]] = {}
    validation_rows: list[dict[str, object]] = []

    for multiplier, experiment in LR_CONDITIONS.items():
        per_multiplier[multiplier] = {}
        for seed in EXPECTED_SEEDS:
            metrics_path = find_complete_run(args.outputs_root, experiment, seed)
            checkpoint = validation_checkpoint(read_metrics(metrics_path))
            per_multiplier[multiplier][seed] = {
                "metrics_csv": str(metrics_path),
                "epoch": int(float(checkpoint["epoch"])),
                "val_accuracy": float(checkpoint["val_accuracy"]),
                "val_loss": float(checkpoint["val_loss"]),
                "test_accuracy": float(checkpoint["test_accuracy"]),
            }

        selected = list(per_multiplier[multiplier].values())
        validation_rows.append(
            {
                "lr_multiplier": multiplier,
                "experiment": experiment,
                "n": len(selected),
                "mean_best_val_accuracy_pct": 100.0
                * mean(float(row["val_accuracy"]) for row in selected),
                "mean_val_loss_at_best_val": mean(
                    float(row["val_loss"]) for row in selected
                ),
            }
        )

    selected_multiplier = max(
        LR_CONDITIONS,
        key=lambda multiplier: (
            mean(
                float(row["val_accuracy"])
                for row in per_multiplier[multiplier].values()
            ),
            -mean(
                float(row["val_loss"]) for row in per_multiplier[multiplier].values()
            ),
            -multiplier,
        ),
    )
    for row in validation_rows:
        row["selected_by_validation"] = (
            float(row["lr_multiplier"]) == selected_multiplier
        )

    boundary: dict[int, dict[str, object]] = {}
    for seed in EXPECTED_SEEDS:
        metrics_path = find_complete_run(args.outputs_root, BOUNDARY_EXPERIMENT, seed)
        checkpoint = validation_checkpoint(read_metrics(metrics_path))
        boundary[seed] = {
            "metrics_csv": str(metrics_path),
            "epoch": int(float(checkpoint["epoch"])),
            "val_accuracy": float(checkpoint["val_accuracy"]),
            "val_loss": float(checkpoint["val_loss"]),
            "test_accuracy": float(checkpoint["test_accuracy"]),
        }

    comparison_rows: list[dict[str, object]] = []
    differences: list[float] = []
    for seed in EXPECTED_SEEDS:
        control = per_multiplier[selected_multiplier][seed]
        boundary_row = boundary[seed]
        difference = 100.0 * (
            float(boundary_row["test_accuracy"]) - float(control["test_accuracy"])
        )
        differences.append(difference)
        comparison_rows.append(
            {
                "seed": seed,
                "selected_lr_multiplier": selected_multiplier,
                "control_best_val_epoch": control["epoch"],
                "control_best_val_accuracy_pct": 100.0
                * float(control["val_accuracy"]),
                "control_test_at_best_val_pct": 100.0
                * float(control["test_accuracy"]),
                "boundary_best_val_epoch": boundary_row["epoch"],
                "boundary_best_val_accuracy_pct": 100.0
                * float(boundary_row["val_accuracy"]),
                "boundary_test_at_best_val_pct": 100.0
                * float(boundary_row["test_accuracy"]),
                "boundary_minus_control_pp": difference,
            }
        )

    low, high = ci95(differences)
    selection = {
        "selected_lr_multiplier": selected_multiplier,
        "selected_experiment": LR_CONDITIONS[selected_multiplier],
        "selection_rule": (
            "highest mean validation accuracy; tie by lower mean validation loss; "
            "then smaller LR multiplier"
        ),
        "selection_seeds": list(EXPECTED_SEEDS),
        "mean_boundary_minus_selected_control_pp": mean(differences),
        "ci95_low_pp": low,
        "ci95_high_pp": high,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_selection.csv", validation_rows)
    write_csv(args.output_dir / "selected_comparison_seed_level.csv", comparison_rows)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
