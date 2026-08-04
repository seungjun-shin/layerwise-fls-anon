#!/usr/bin/env python
"""Frozen paired statistics for WP2 accuracy and WP4 diagnostics."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"


def arithmetic_mean(values: list[float]) -> float:
    """Return the arithmetic mean without importing the shadowed stdlib module."""
    return math.fsum(values) / len(values)


def sample_stdev(values: list[float]) -> float:
    """Return the Bessel-corrected sample standard deviation."""
    center = arithmetic_mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-table",
        default=str(WORKSPACE / "seed_tables" / "practical_main_confirmatory_seed_level.csv"),
    )
    parser.add_argument(
        "--mechanism-table",
        default=str(WORKSPACE / "analysis" / "mechanism" / "mechanism_seed_level.csv"),
    )
    return parser.parse_args()


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    if not differences or not all(math.isfinite(value) for value in differences):
        raise ValueError("Exact sign-flip test requires non-empty finite paired differences.")
    observed = abs(arithmetic_mean(differences))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(
            arithmetic_mean(
                [sign * value for sign, value in zip(signs, differences, strict=True)]
            )
        )
        count += statistic >= observed - 1e-15
        total += 1
    return count / total


def paired_summary(differences: list[float]) -> dict[str, Any]:
    if not differences or not all(math.isfinite(value) for value in differences):
        raise ValueError("Paired summary requires non-empty finite differences.")
    n = len(differences)
    average = arithmetic_mean(differences)
    sd = sample_stdev(differences) if n > 1 else float("nan")
    critical = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
    half_width = critical * sd / math.sqrt(n) if n > 1 else float("nan")
    try:
        wilcoxon_p = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n": n,
        "mean_difference": average,
        "median_difference": float(np.median(differences)),
        "sd_difference": sd,
        "ci95_low": average - half_width,
        "ci95_high": average + half_width,
        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
        "wilcoxon_sensitivity_p": wilcoxon_p,
        "paired_standardized_effect": average / sd if sd and math.isfinite(sd) else float("nan"),
    }


def rows_by_condition(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        condition = row["condition"]
        seed = int(row["seed"])
        if seed not in set(range(5, 15)):
            raise ValueError(f"Unexpected non-confirmatory seed {seed} in {path}.")
        condition_rows = output.setdefault(condition, {})
        if seed in condition_rows:
            raise ValueError(f"Duplicate ({condition}, seed={seed}) row in {path}.")
        condition_rows[seed] = row
    return output


def finite_float(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing/non-numeric {field} for {context}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for {context}.")
    return value


def comparison_rows(seed_table: Path) -> list[dict[str, Any]]:
    grouped = rows_by_condition(seed_table)
    expected_conditions = {"P-REF", "P-ACT-FINAL", "P-LR-FINAL"}
    if set(grouped) != expected_conditions:
        raise ValueError(
            f"Accuracy table conditions {sorted(grouped)} do not match {sorted(expected_conditions)}."
        )
    comparisons = [
        ("H1", "P-ACT-FINAL", "P-LR-FINAL"),
        ("secondary", "P-ACT-FINAL", "P-REF"),
        ("secondary", "P-LR-FINAL", "P-REF"),
    ]
    output: list[dict[str, Any]] = []
    for tier, left, right in comparisons:
        seeds = sorted(set(grouped[left]) & set(grouped[right]))
        differences = [
            finite_float(
                grouped[left][seed], "test_accuracy_percent", f"{left}, seed={seed}"
            )
            - finite_float(
                grouped[right][seed], "test_accuracy_percent", f"{right}, seed={seed}"
            )
            for seed in seeds
        ]
        output.append(
            {
                "tier": tier,
                "comparison": f"{left} - {right}",
                "metric": "test_accuracy_percentage_points",
                "planned_n": 10,
                "completed_paired_n": len(seeds),
                "excluded_n": 10 - len(seeds),
                "seeds": ";".join(map(str, seeds)),
                **paired_summary(differences),
            }
        )
    return output


def holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in pvalues):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def mechanism_rows(path: Path) -> list[dict[str, Any]]:
    grouped = rows_by_condition(path)
    expected_conditions = {"P-REF", "P-ACT-FINAL", "P-LR-FINAL"}
    if set(grouped) != expected_conditions:
        raise ValueError(
            f"Mechanism table conditions {sorted(grouped)} do not match {sorted(expected_conditions)}."
        )
    metrics = [
        "feature_l2_norm",
        "classifier_input_gradient_l2_norm",
        "logit_l2_norm",
        "classifier_weight_effective_rank",
    ]
    output: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    for metric in metrics:
        seeds = sorted(set(grouped["P-ACT-FINAL"]) & set(grouped["P-LR-FINAL"]))
        differences = [
            finite_float(grouped["P-ACT-FINAL"][seed], metric, f"P-ACT-FINAL, seed={seed}")
            - finite_float(grouped["P-LR-FINAL"][seed], metric, f"P-LR-FINAL, seed={seed}")
            for seed in seeds
        ]
        row = {
            "tier": "H2",
            "comparison": "P-ACT-FINAL - P-LR-FINAL",
            "metric": metric,
            "planned_n": 10,
            "completed_paired_n": len(seeds),
            "excluded_n": 10 - len(seeds),
            "seeds": ";".join(map(str, seeds)),
            **paired_summary(differences),
        }
        output.append(row)
        raw_pvalues.append(float(row["exact_sign_flip_p"]))
    for row, adjusted in zip(output, holm_adjust(raw_pvalues), strict=True):
        row["holm_adjusted_p"] = adjusted
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    accuracy = comparison_rows(Path(args.seed_table))
    diagnostics = mechanism_rows(Path(args.mechanism_table))
    write_csv(WORKSPACE / "tables" / "practical_main_summary.csv", accuracy)
    write_csv(WORKSPACE / "analysis" / "statistics" / "mechanism_holm_summary.csv", diagnostics)
    print(WORKSPACE / "tables" / "practical_main_summary.csv")
    print(WORKSPACE / "analysis" / "statistics" / "mechanism_holm_summary.csv")


if __name__ == "__main__":
    main()
