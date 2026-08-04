#!/usr/bin/env python
"""Summarize the validation-controlled ViT-CIFAR experiment package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy import stats


CONDITIONS = {
    "baseline": "val_vit25_baseline",
    "global_c0p25": "val_vit25_global_c0p25",
    "global_c0p0625": "val_vit25_global_c0p0625",
    "boundary_late_c0p0625": "val_vit25_boundary_late_c0p0625",
    "lr_only_late_16x": "val_vit25_lr_only_late_16x",
}
SEEDS = tuple(range(5))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_complete_run(root: Path, experiment: str, seed: int) -> dict[str, Any]:
    """Read the latest complete run and select its earliest best-validation epoch."""
    candidates = sorted(root.glob(f"{experiment}/**/seed_{seed}/metrics.csv"), reverse=True)
    for metrics_path in candidates:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or max(int(float(row["epoch"])) for row in rows) < 79:
            continue
        best_val = max(float(row["val_accuracy"]) for row in rows)
        selected = next(row for row in rows if float(row["val_accuracy"]) == best_val)
        return {
            "experiment": experiment,
            "seed": seed,
            "metrics_csv": str(metrics_path),
            "best_val_epoch": int(float(selected["epoch"])),
            "best_val_accuracy_pct": 100.0 * float(selected["val_accuracy"]),
            "val_loss_at_best_val": float(selected["val_loss"]),
            "test_accuracy_at_best_val_pct": 100.0 * float(selected["test_accuracy"]),
            "test_loss_at_best_val": float(selected["test_loss"]),
            "final_test_accuracy_pct": 100.0 * float(rows[-1]["test_accuracy"]),
        }
    raise FileNotFoundError(f"No complete metrics for {experiment}, seed {seed}")


def mean_sd(values: list[float]) -> dict[str, float | int]:
    """Return count, mean, and sample standard deviation."""
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def paired_summary(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize paired test-accuracy differences in percentage points."""
    right_by_seed = {int(row["seed"]): row for row in right}
    diffs = [
        float(row["test_accuracy_at_best_val_pct"])
        - float(right_by_seed[int(row["seed"])]["test_accuracy_at_best_val_pct"])
        for row in left
    ]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else 0.0
    p = float(stats.ttest_1samp(diffs, popmean=0.0).pvalue) if n > 1 and sd > 0 else 1.0
    return {
        "n": n,
        "mean_pp": mean,
        "sd_pp": sd,
        "ci95_low_pp": mean - half,
        "ci95_high_pp": mean + half,
        "paired_p": p,
    }


def main() -> None:
    """Collect all runs and write seed-level and aggregate summaries."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        condition: [read_complete_run(args.outputs_root, experiment, seed) for seed in SEEDS]
        for condition, experiment in CONDITIONS.items()
    }

    seed_path = args.output_dir / "seed_level.csv"
    fields = ["condition", *next(iter(records.values()))[0].keys()]
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, rows in records.items():
            for row in rows:
                writer.writerow({"condition": condition, **row})

    condition_summary = {
        condition: {
            "validation_accuracy_pct": mean_sd([float(row["best_val_accuracy_pct"]) for row in rows]),
            "test_accuracy_at_best_validation_pct": mean_sd(
                [float(row["test_accuracy_at_best_val_pct"]) for row in rows]
            ),
        }
        for condition, rows in records.items()
    }
    contrasts = {
        "legacy_boundary_minus_global_c0p25": paired_summary(
            records["boundary_late_c0p0625"], records["global_c0p25"]
        ),
        "activation_boundary_minus_exact_lr_only": paired_summary(
            records["boundary_late_c0p0625"], records["lr_only_late_16x"]
        ),
        "boundary_minus_baseline": paired_summary(
            records["boundary_late_c0p0625"], records["baseline"]
        ),
        "lr_only_minus_baseline": paired_summary(records["lr_only_late_16x"], records["baseline"]),
    }
    summary = {
        "protocol": "fixed 10% CIFAR-100 train/validation split; untouched test set",
        "checkpoint_selection": "earliest epoch attaining maximum validation accuracy",
        "seeds": list(SEEDS),
        "test_metrics_used_for_selection": False,
        "conditions_fixed_before_evaluation": True,
        "condition_summary": condition_summary,
        "paired_contrasts": contrasts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Validation-controlled ViT-CIFAR summary",
        "",
        "All checkpoints were selected using the fixed validation split; test metrics were report-only.",
        "",
        "| Condition | Test accuracy at validation-selected checkpoint |",
        "|---|---:|",
    ]
    for condition, values in condition_summary.items():
        metric = values["test_accuracy_at_best_validation_pct"]
        lines.append(f"| {condition} | {metric['mean']:.2f} +/- {metric['sd']:.2f} |")
    lines.extend(["", "## Paired contrasts", ""])
    for name, values in contrasts.items():
        lines.append(
            f"- {name}: {values['mean_pp']:+.2f} pp "
            f"[95% CI {values['ci95_low_pp']:+.2f}, {values['ci95_high_pp']:+.2f}], "
            f"p={values['paired_p']:.4g}"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {seed_path} and aggregate summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
