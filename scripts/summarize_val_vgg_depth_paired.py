#!/usr/bin/env python
"""Summarize the validation-controlled paired VGG19-BN depth sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


CUTS = (1, 3, 5, 7, 9, 11, 13, 15)
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_EPOCHS = 80


def experiment_name(cut: int, condition: str) -> str:
    """Return the experiment name for one cut and condition."""
    if cut == 15:
        return (
            "val_vgg_bnd_cut15_iso"
            if condition == "boundary"
            else "val_vgg_lr_cut15_16x_iso"
        )
    prefix = "bnd" if condition == "boundary" else "lr"
    suffix = "" if condition == "boundary" else "_16x"
    return f"val_vgg_depth_{prefix}_cut{cut}{suffix}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/_summaries/vgg_depth_paired_20260731"),
        type=Path,
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read and validate one complete metrics file."""
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
    """Select the earliest epoch attaining maximum validation accuracy."""
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
    """Build seed-level and per-cut paired summaries."""
    args = parse_args()
    selected: dict[tuple[int, str, int], dict[str, float | int | str]] = {}
    seed_rows: list[dict[str, object]] = []

    for cut in CUTS:
        for condition in ("boundary", "lr_only_exact_16x"):
            experiment = experiment_name(cut, condition)
            for seed in EXPECTED_SEEDS:
                metrics_path = find_complete_run(args.outputs_root, experiment, seed)
                rows = read_metrics(metrics_path)
                checkpoint = validation_checkpoint(rows)
                final = rows[-1]
                record: dict[str, float | int | str] = {
                    "metrics_csv": str(metrics_path),
                    "best_val_epoch": int(float(checkpoint["epoch"])),
                    "best_val_accuracy_pct": 100.0
                    * float(checkpoint["val_accuracy"]),
                    "test_accuracy_at_best_val_pct": 100.0
                    * float(checkpoint["test_accuracy"]),
                    "test_loss_at_best_val": float(checkpoint["test_loss"]),
                    "final_test_accuracy_pct": 100.0 * float(final["test_accuracy"]),
                    "final_train_loss": float(final["train_loss"]),
                }
                selected[cut, condition, seed] = record
                seed_rows.append(
                    {
                        "cut": cut,
                        "normalized_depth": cut / 15.0,
                        "condition": condition,
                        "experiment": experiment,
                        "seed": seed,
                        **record,
                    }
                )

    cut_rows: list[dict[str, object]] = []
    boundary_means: list[float] = []
    lr_means: list[float] = []
    margin_means: list[float] = []
    for cut in CUTS:
        boundary = [
            float(selected[cut, "boundary", seed]["test_accuracy_at_best_val_pct"])
            for seed in EXPECTED_SEEDS
        ]
        lr_only = [
            float(
                selected[cut, "lr_only_exact_16x", seed][
                    "test_accuracy_at_best_val_pct"
                ]
            )
            for seed in EXPECTED_SEEDS
        ]
        differences = [boundary[i] - lr_only[i] for i in range(len(EXPECTED_SEEDS))]
        boundary_stats = sample_summary(boundary)
        lr_stats = sample_summary(lr_only)
        diff_stats = sample_summary(differences)
        paired = stats.ttest_rel(boundary, lr_only)
        boundary_means.append(boundary_stats["mean"])
        lr_means.append(lr_stats["mean"])
        margin_means.append(diff_stats["mean"])
        cut_rows.append(
            {
                "cut": cut,
                "normalized_depth": cut / 15.0,
                "n": len(EXPECTED_SEEDS),
                "boundary_mean_test_at_best_val_pct": boundary_stats["mean"],
                "boundary_sd_pct": boundary_stats["sd"],
                "lr_only_mean_test_at_best_val_pct": lr_stats["mean"],
                "lr_only_sd_pct": lr_stats["sd"],
                "boundary_minus_lr_only_pp": diff_stats["mean"],
                "margin_sd_pp": diff_stats["sd"],
                "margin_ci95_low_pp": diff_stats["ci95_low"],
                "margin_ci95_high_pp": diff_stats["ci95_high"],
                "paired_t_nominal": float(paired.statistic),
                "paired_p_nominal_uncorrected": float(paired.pvalue),
            }
        )

    cut_axis = list(CUTS)
    rho_boundary = stats.spearmanr(cut_axis, boundary_means)
    rho_lr = stats.spearmanr(cut_axis, lr_means)
    rho_margin = stats.spearmanr(cut_axis, margin_means)
    nonfinal_cuts = CUTS[:-1]
    final_minus_nonfinal: list[float] = []
    nonfinal_seed_means: list[float] = []
    for seed in EXPECTED_SEEDS:
        nonfinal_margins = [
            float(
                selected[cut, "boundary", seed][
                    "test_accuracy_at_best_val_pct"
                ]
            )
            - float(
                selected[cut, "lr_only_exact_16x", seed][
                    "test_accuracy_at_best_val_pct"
                ]
            )
            for cut in nonfinal_cuts
        ]
        final_margin = float(
            selected[15, "boundary", seed]["test_accuracy_at_best_val_pct"]
        ) - float(
            selected[15, "lr_only_exact_16x", seed][
                "test_accuracy_at_best_val_pct"
            ]
        )
        nonfinal_seed_means.append(mean(nonfinal_margins))
        final_minus_nonfinal.append(final_margin - mean(nonfinal_margins))
    interaction_stats = sample_summary(final_minus_nonfinal)
    summary = {
        "checkpoint_selection": "maximum validation accuracy within each run",
        "seeds": list(EXPECTED_SEEDS),
        "cuts": list(CUTS),
        "activation_multiplier": 0.0625,
        "exact_lr_only_upstream_multiplier": 16.0,
        "spearman_descriptive": {
            "boundary_accuracy_rho": float(rho_boundary.statistic),
            "boundary_accuracy_p_nominal": float(rho_boundary.pvalue),
            "lr_only_accuracy_rho": float(rho_lr.statistic),
            "lr_only_accuracy_p_nominal": float(rho_lr.pvalue),
            "margin_rho": float(rho_margin.statistic),
            "margin_p_nominal": float(rho_margin.pvalue),
        },
        "final_localization_descriptive": {
            "mean_nonfinal_margin_pp": mean(nonfinal_seed_means),
            "final_margin_pp": margin_means[-1],
            "final_minus_mean_nonfinal_pp": interaction_stats["mean"],
            "ci95_low_pp": interaction_stats["ci95_low"],
            "ci95_high_pp": interaction_stats["ci95_high"],
        },
        "interpretation_rule": (
            "Per-cut intervals and p-values are descriptive and uncorrected; "
            "the selection-free final-cut replication is the confirmatory VGG result."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "seed_level.csv", seed_rows)
    write_csv(args.output_dir / "cut_summary.csv", cut_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
