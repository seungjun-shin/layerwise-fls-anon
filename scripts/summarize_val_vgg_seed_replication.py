#!/usr/bin/env python
"""Summarize selection-free VGG19-BN seed replications."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


CONDITIONS = {
    "boundary_cut15_16x": "val_vgg_bnd_cut15_rep",
    "optimized_lr_only_1.75x": "val_vgg_lr_cut15_1p75x_rep",
    "global_c0.25": "val_vgg_global_c0p25_rep",
    "exact_lr_only_16x": "val_vgg_lr_cut15_16x_rep",
}
EXPECTED_SEEDS = tuple(range(3, 10))
EXPECTED_EPOCHS = 80


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/_summaries/vgg_seed_replication_20260730"),
        type=Path,
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read and validate a complete metrics file."""
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


def sample_summary(values: list[float]) -> dict[str, float]:
    """Return mean, sample SD, and a two-sided 95% t interval."""
    avg = mean(values)
    sd = stdev(values)
    half_width = float(stats.t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(
        len(values)
    )
    return {
        "mean": avg,
        "sd": sd,
        "ci95_low": avg - half_width,
        "ci95_high": avg + half_width,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build seed-level, condition-level, and paired replication summaries."""
    args = parse_args()
    selected: dict[str, dict[int, dict[str, float | int | str]]] = {}
    seed_rows: list[dict[str, object]] = []

    for condition, experiment in CONDITIONS.items():
        selected[condition] = {}
        for seed in EXPECTED_SEEDS:
            metrics_path = find_complete_run(args.outputs_root, experiment, seed)
            rows = read_metrics(metrics_path)
            checkpoint = validation_checkpoint(rows)
            final = rows[-1]
            record: dict[str, float | int | str] = {
                "metrics_csv": str(metrics_path),
                "best_val_epoch": int(float(checkpoint["epoch"])),
                "best_val_accuracy_pct": 100.0 * float(checkpoint["val_accuracy"]),
                "test_accuracy_at_best_val_pct": 100.0
                * float(checkpoint["test_accuracy"]),
                "test_loss_at_best_val": float(checkpoint["test_loss"]),
                "final_val_accuracy_pct": 100.0 * float(final["val_accuracy"]),
                "final_test_accuracy_pct": 100.0 * float(final["test_accuracy"]),
                "final_train_loss": float(final["train_loss"]),
            }
            selected[condition][seed] = record
            seed_rows.append(
                {
                    "condition": condition,
                    "experiment": experiment,
                    "seed": seed,
                    **record,
                }
            )

    condition_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        val_values = [
            float(selected[condition][seed]["best_val_accuracy_pct"])
            for seed in EXPECTED_SEEDS
        ]
        test_values = [
            float(selected[condition][seed]["test_accuracy_at_best_val_pct"])
            for seed in EXPECTED_SEEDS
        ]
        final_values = [
            float(selected[condition][seed]["final_test_accuracy_pct"])
            for seed in EXPECTED_SEEDS
        ]
        val_stats = sample_summary(val_values)
        test_stats = sample_summary(test_values)
        final_stats = sample_summary(final_values)
        condition_rows.append(
            {
                "condition": condition,
                "n": len(EXPECTED_SEEDS),
                "mean_best_val_accuracy_pct": val_stats["mean"],
                "sd_best_val_accuracy_pct": val_stats["sd"],
                "mean_test_accuracy_at_best_val_pct": test_stats["mean"],
                "sd_test_accuracy_at_best_val_pct": test_stats["sd"],
                "test_ci95_low_pct": test_stats["ci95_low"],
                "test_ci95_high_pct": test_stats["ci95_high"],
                "mean_final_test_accuracy_pct": final_stats["mean"],
                "sd_final_test_accuracy_pct": final_stats["sd"],
            }
        )

    comparison_specs = (
        ("primary_boundary_minus_optimized_lr", "optimized_lr_only_1.75x"),
        ("secondary_boundary_minus_global", "global_c0.25"),
        ("diagnostic_boundary_minus_exact_16x", "exact_lr_only_16x"),
    )
    comparison_rows: list[dict[str, object]] = []
    for label, comparator in comparison_specs:
        differences = [
            float(
                selected["boundary_cut15_16x"][seed][
                    "test_accuracy_at_best_val_pct"
                ]
            )
            - float(selected[comparator][seed]["test_accuracy_at_best_val_pct"])
            for seed in EXPECTED_SEEDS
        ]
        diff_stats = sample_summary(differences)
        paired = stats.ttest_rel(
            [
                float(
                    selected["boundary_cut15_16x"][seed][
                        "test_accuracy_at_best_val_pct"
                    ]
                )
                for seed in EXPECTED_SEEDS
            ],
            [
                float(selected[comparator][seed]["test_accuracy_at_best_val_pct"])
                for seed in EXPECTED_SEEDS
            ],
        )
        comparison_rows.append(
            {
                "comparison": label,
                "comparator": comparator,
                "n": len(EXPECTED_SEEDS),
                "mean_boundary_minus_comparator_pp": diff_stats["mean"],
                "sd_difference_pp": diff_stats["sd"],
                "ci95_low_pp": diff_stats["ci95_low"],
                "ci95_high_pp": diff_stats["ci95_high"],
                "paired_t": float(paired.statistic),
                "paired_p": float(paired.pvalue),
            }
        )

    finite_status = {
        condition: all(
            math.isfinite(float(selected[condition][seed]["final_train_loss"]))
            for seed in EXPECTED_SEEDS
        )
        for condition in CONDITIONS
    }
    summary = {
        "seed_role": "selection-free seed replications after comparator selection",
        "seeds": list(EXPECTED_SEEDS),
        "optimized_lr_multiplier": 1.75,
        "checkpoint_selection": "maximum validation accuracy within each run",
        "finite_status": finite_status,
        "primary_comparison": comparison_rows[0],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_level.csv", seed_rows)
    write_csv(args.output_dir / "condition_summary.csv", condition_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
