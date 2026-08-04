#!/usr/bin/env python
"""Summarize the IVC P0-3 decomposition and P0-4 robustness evaluations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scipy import stats

CORRUPTION_FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
PRIMARY_ROBUSTNESS_ENDPOINTS = (
    "mCA",
    "msp_auroc",
    "energy_auroc",
    "mahalanobis_auroc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--norm-direction-dir",
        type=Path,
        default=Path("outputs/ivc_p0_3_norm_direction_20260804"),
    )
    parser.add_argument("--robustness-root", type=Path, default=Path("outputs/robustness"))
    parser.add_argument(
        "--robustness-act-experiment",
        action="append",
        dest="robustness_act_experiments",
        help="ACT robustness result directory name; repeat for sharded evaluations.",
    )
    parser.add_argument(
        "--robustness-lr-experiment",
        action="append",
        dest="robustness_lr_experiments",
        help="LR-only robustness result directory name; repeat for sharded evaluations.",
    )
    parser.add_argument(
        "--robustness-seeds", type=int, nargs="+", default=list(range(5))
    )
    parser.add_argument("--skip-norm-direction", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ivc_p0_aux_summary_20260804")
    )
    return parser.parse_args()


def interval(values: list[float]) -> dict[str, float]:
    """Return mean, SD, and a two-sided 95% t interval."""
    mean = fmean(values)
    sd = stdev(values)
    half = float(stats.t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def exact_sign_flip_p(values: list[float]) -> float:
    """Return the exhaustive two-sided paired sign-flip p-value."""
    observed = abs(fmean(values))
    exceedances = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(fmean(sign * value for sign, value in zip(signs, values, strict=True)))
        exceedances += int(permuted >= observed - 1e-15)
        total += 1
    return exceedances / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values keyed like the input."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_norm_direction(source: Path, output: Path) -> dict[str, Any]:
    """Merge the two GPU shards and summarize every numeric metric."""
    rows = read_csv(source / "seeds_5_9.csv") + read_csv(source / "seeds_10_14.csv")
    rows.sort(key=lambda row: int(row["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "norm_direction_seed_level.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {"num_seeds": len(rows), "split": "validation"}
    for key in rows[0]:
        if key in {"seed", "num_examples"}:
            continue
        summary[key] = interval([float(row[key]) for row in rows])
    return summary


def read_seed_result(root: Path, experiments: list[str], seed: int) -> dict[str, Any]:
    """Read exactly one sharded robustness result for a seed."""
    candidates = [root / experiment / f"seed_{seed}.json" for experiment in experiments]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise ValueError(
            f"Expected one result for seed {seed}, found {len(existing)}: {existing}"
        )
    return json.loads(existing[0].read_text(encoding="utf-8"))


def corruption_family_mean(row: dict[str, Any], family: str) -> float:
    values = row["corruption"]["per_corruption_mean"]
    return fmean(float(values[name]) for name in CORRUPTION_FAMILIES[family])


def summarize_robustness(
    root: Path,
    output: Path,
    act_experiments: list[str] | None = None,
    lr_experiments: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Compute paired ACT-minus-LR-only robustness and OOD statistics."""
    output.mkdir(parents=True, exist_ok=True)
    act_experiments = act_experiments or ["val60_core_boundary_cut8"]
    lr_experiments = lr_experiments or ["val60_core_lronly_cut8"]
    seeds = seeds if seeds is not None else list(range(5))
    act = [read_seed_result(root, act_experiments, seed) for seed in seeds]
    lr = [read_seed_result(root, lr_experiments, seed) for seed in seeds]
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "mCA": lambda row: float(row["corruption"]["mCA"]),
        "msp_auroc": lambda row: float(row["ood"]["msp_auroc"]),
        "energy_auroc": lambda row: float(row["ood"]["energy_auroc"]),
        "mahalanobis_auroc": lambda row: float(row["ood"]["mahalanobis_auroc"]),
    }
    for severity in range(1, 6):
        extractors[f"severity_{severity}_accuracy"] = (
            lambda row, severity=severity: float(
                row["corruption"]["per_severity_mean"][str(severity)]
            )
        )
    for family in CORRUPTION_FAMILIES:
        extractors[f"family_{family}_accuracy"] = (
            lambda row, family=family: corruption_family_mean(row, family)
        )
    for corruption in (
        name for names in CORRUPTION_FAMILIES.values() for name in names
    ):
        extractors[f"corruption_{corruption}_accuracy"] = (
            lambda row, corruption=corruption: float(
                row["corruption"]["per_corruption_mean"][corruption]
            )
        )
    seed_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"num_paired_seeds": len(act)}
    for index, seed in enumerate(seeds):
        row: dict[str, Any] = {"seed": seed}
        for metric, extractor in extractors.items():
            row[f"act_{metric}"] = extractor(act[index])
            row[f"lr_only_{metric}"] = extractor(lr[index])
            row[f"act_minus_lr_{metric}"] = row[f"act_{metric}"] - row[f"lr_only_{metric}"]
        seed_rows.append(row)
    for metric in extractors:
        values = [float(row[f"act_minus_lr_{metric}"]) for row in seed_rows]
        metric_summary = interval(values)
        act_values = [float(row[f"act_{metric}"]) for row in seed_rows]
        lr_values = [float(row[f"lr_only_{metric}"]) for row in seed_rows]
        metric_summary["act_mean"] = fmean(act_values)
        metric_summary["lr_only_mean"] = fmean(lr_values)
        metric_summary["paired_t_p"] = float(stats.ttest_rel(act_values, lr_values).pvalue)
        metric_summary["exact_sign_flip_p"] = exact_sign_flip_p(values)
        summary[metric] = metric_summary
    t_p_values = {
        metric: float(summary[metric]["paired_t_p"])
        for metric in PRIMARY_ROBUSTNESS_ENDPOINTS
    }
    sign_p_values = {
        metric: float(summary[metric]["exact_sign_flip_p"])
        for metric in PRIMARY_ROBUSTNESS_ENDPOINTS
    }
    adjusted_t = holm_adjust(t_p_values)
    adjusted_sign = holm_adjust(sign_p_values)
    for metric in PRIMARY_ROBUSTNESS_ENDPOINTS:
        summary[metric]["holm_paired_t_p"] = adjusted_t[metric]
        summary[metric]["holm_exact_sign_flip_p"] = adjusted_sign[metric]
    with (output / "robustness_seed_level.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if not args.skip_norm_direction:
        summary["p0_3_norm_direction"] = summarize_norm_direction(
            args.norm_direction_dir, args.output_dir
        )
    summary["p0_4_exact_control_robustness"] = summarize_robustness(
        args.robustness_root,
        args.output_dir,
        args.robustness_act_experiments,
        args.robustness_lr_experiments,
        args.robustness_seeds,
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
