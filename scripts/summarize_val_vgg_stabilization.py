#!/usr/bin/env python
"""Summarize the validation-controlled VGG19-BN stabilization sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


CONDITIONS = {
    "global_c0.25": "val_vgg_global_c0p25_iso",
    "boundary_cut15_16x": "val_vgg_bnd_cut15_iso",
    "lr_only_cut15_2x": "val_vgg_lr_cut15_2x_iso",
    "lr_only_cut15_4x": "val_vgg_lr_cut15_4x_iso",
    "lr_only_cut15_8x": "val_vgg_lr_cut15_8x_iso",
    "lr_only_cut15_16x": "val_vgg_lr_cut15_16x_iso",
}
LR_ONLY_CONDITIONS = tuple(name for name in CONDITIONS if name.startswith("lr_only_"))
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_EPOCHS = 80


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/_summaries/vgg_stabilization_20260730"),
        type=Path,
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read and validate one metrics file."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_EPOCHS:
        raise ValueError(f"Expected {EXPECTED_EPOCHS} rows, found {len(rows)}: {path}")
    if int(float(rows[-1]["epoch"])) != EXPECTED_EPOCHS - 1:
        raise ValueError(f"Incomplete run: {path}")
    numeric_fields = (
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "val_loss",
        "val_accuracy",
    )
    for row in rows:
        for field in numeric_fields:
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
    errors: list[str] = []
    for path in candidates:
        try:
            read_metrics(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        return path
    detail = "\n".join(errors) if errors else "no metrics files found"
    raise FileNotFoundError(f"No complete run for {experiment}, seed {seed}:\n{detail}")


def select_by_validation(rows: list[dict[str, str]]) -> dict[str, str]:
    """Select the earliest epoch attaining the maximum validation accuracy."""
    return max(rows, key=lambda row: float(row["val_accuracy"]))


def sample_summary(values: list[float]) -> dict[str, float]:
    """Return mean, sample SD, and a two-sided 95% t interval."""
    n = len(values)
    avg = mean(values)
    sd = stdev(values)
    half_width = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n": float(n),
        "mean": avg,
        "sd": sd,
        "ci95_low": avg - half_width,
        "ci95_high": avg + half_width,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build seed-level, condition-level, and paired-comparison summaries."""
    args = parse_args()
    seed_rows: list[dict[str, object]] = []
    selected_test: dict[str, dict[int, float]] = {}
    selected_val: dict[str, dict[int, float]] = {}

    for condition, experiment in CONDITIONS.items():
        selected_test[condition] = {}
        selected_val[condition] = {}
        for seed in EXPECTED_SEEDS:
            metrics_path = find_complete_run(args.outputs_root, experiment, seed)
            rows = read_metrics(metrics_path)
            best = select_by_validation(rows)
            final = rows[-1]
            val_accuracy = 100.0 * float(best["val_accuracy"])
            test_accuracy = 100.0 * float(best["test_accuracy"])
            selected_val[condition][seed] = val_accuracy
            selected_test[condition][seed] = test_accuracy
            seed_rows.append(
                {
                    "condition": condition,
                    "experiment": experiment,
                    "seed": seed,
                    "run_dir": str(metrics_path.parent),
                    "status": "complete_finite",
                    "best_val_epoch": int(float(best["epoch"])),
                    "best_val_accuracy_pct": val_accuracy,
                    "test_accuracy_at_best_val_pct": test_accuracy,
                    "test_loss_at_best_val": float(best["test_loss"]),
                    "final_val_accuracy_pct": 100.0 * float(final["val_accuracy"]),
                    "final_test_accuracy_pct": 100.0 * float(final["test_accuracy"]),
                    "final_train_loss": float(final["train_loss"]),
                }
            )

    condition_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        val_stats = sample_summary(list(selected_val[condition].values()))
        test_stats = sample_summary(list(selected_test[condition].values()))
        condition_rows.append(
            {
                "condition": condition,
                "n": int(test_stats["n"]),
                "mean_best_val_accuracy_pct": val_stats["mean"],
                "sd_best_val_accuracy_pct": val_stats["sd"],
                "mean_test_accuracy_at_best_val_pct": test_stats["mean"],
                "sd_test_accuracy_at_best_val_pct": test_stats["sd"],
                "test_ci95_low_pct": test_stats["ci95_low"],
                "test_ci95_high_pct": test_stats["ci95_high"],
            }
        )

    best_lr_only = max(
        LR_ONLY_CONDITIONS,
        key=lambda condition: mean(selected_val[condition].values()),
    )
    comparison_order = (*LR_ONLY_CONDITIONS, best_lr_only)
    comparison_labels = (*LR_ONLY_CONDITIONS, "validation_selected_lr_only")
    comparison_rows: list[dict[str, object]] = []
    for label, comparator in zip(comparison_labels, comparison_order):
        differences = [
            selected_test["boundary_cut15_16x"][seed] - selected_test[comparator][seed]
            for seed in EXPECTED_SEEDS
        ]
        diff_stats = sample_summary(differences)
        test = stats.ttest_rel(
            [selected_test["boundary_cut15_16x"][seed] for seed in EXPECTED_SEEDS],
            [selected_test[comparator][seed] for seed in EXPECTED_SEEDS],
        )
        comparison_rows.append(
            {
                "comparison": label,
                "actual_comparator": comparator,
                "comparator_selected_by": (
                    "mean_validation_accuracy_across_seeds"
                    if label == "validation_selected_lr_only"
                    else "fixed_in_advance"
                ),
                "n": int(diff_stats["n"]),
                "mean_boundary_minus_control_pp": diff_stats["mean"],
                "sd_difference_pp": diff_stats["sd"],
                "ci95_low_pp": diff_stats["ci95_low"],
                "ci95_high_pp": diff_stats["ci95_high"],
                "paired_t": float(test.statistic),
                "paired_p": float(test.pvalue),
            }
        )

    write_csv(
        args.output_dir / "seed_level.csv",
        list(seed_rows[0]),
        seed_rows,
    )
    write_csv(
        args.output_dir / "condition_summary.csv",
        list(condition_rows[0]),
        condition_rows,
    )
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    print(f"Validation-selected LR-only comparator: {best_lr_only}")
    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
