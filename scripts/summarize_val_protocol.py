#!/usr/bin/env python
"""Summarize the final validation-controlled ResNet depth protocol.

The script reads only raw ``metrics.csv`` files.  For every run it selects the
earliest epoch attaining the maximum held-out validation accuracy and reports
the test endpoint at that epoch.  It produces the seed-level audit table, the
depth-profile plot input, and paired selection/confirmation summaries used by
the main manuscript and Supplementary Tables I--II.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy import stats


CUTS = tuple(range(2, 9))
SELECTION_SEEDS = tuple(range(5))
CONFIRMATION_SEEDS = tuple(range(5, 10))


def parse_args() -> argparse.Namespace:
    """Parse input and output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"))
    return parser.parse_args()


def read_complete(path: Path) -> list[dict[str, str]]:
    """Read one complete 80-epoch metrics trajectory."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80 or int(float(rows[-1]["epoch"])) != 79:
        raise ValueError(f"Incomplete 80-epoch run: {path}")
    return rows


def latest_metrics(outputs_root: Path, experiment: str, seed: int) -> Path:
    """Return the newest complete metrics file for an experiment and seed."""

    candidates = sorted(
        outputs_root.glob(f"{experiment}/*/seed_{seed}/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            read_complete(path)
        except ValueError:
            continue
        return path
    raise FileNotFoundError(f"No complete run for {experiment}, seed {seed}")


def selected_record(
    outputs_root: Path,
    experiment: str,
    seed: int,
    condition: str,
    block: str,
) -> dict[str, Any]:
    """Return one validation-selected report-only test record."""

    path = latest_metrics(outputs_root, experiment, seed)
    rows = read_complete(path)
    maximum = max(float(row["val_accuracy"]) for row in rows)
    selected = next(row for row in rows if float(row["val_accuracy"]) == maximum)
    final = rows[-1]
    return {
        "block": block,
        "condition": condition,
        "experiment": experiment,
        "cut": "",
        "seed": seed,
        "selected_epoch": int(float(selected["epoch"])),
        "validation_accuracy_percent": 100.0 * float(selected["val_accuracy"]),
        "test_accuracy_percent": 100.0 * float(selected["test_accuracy"]),
        "final_test_accuracy_percent": 100.0 * float(final["test_accuracy"]),
        "metrics_csv": str(path),
    }


def experiment_for_cut(role: str, cut: int) -> str:
    """Return the final experiment name for one selection-block cut."""

    if cut == 8:
        return "val_bnd_cut8_iso" if role == "act" else "val_lr_cut8_iso"
    return f"val_{'bnd' if role == 'act' else 'lr'}_cut{cut}"


def exact_sign_flip(values: list[float]) -> float:
    """Return the exhaustive two-sided paired sign-flip p-value."""

    observed = abs(mean(values))
    permutations = (
        abs(mean(sign * value for sign, value in zip(signs, values, strict=True)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    all_values = list(permutations)
    return sum(value >= observed - 1e-12 for value in all_values) / len(all_values)


def paired_summary(
    rows: list[dict[str, Any]],
    block: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    """Summarize a named paired contrast within one design block."""

    selected = [row for row in rows if row["block"] == block]
    by_condition = {
        condition: {
            int(row["seed"]): float(row["test_accuracy_percent"])
            for row in selected
            if row["condition"] == condition
        }
        for condition in (left, right)
    }
    seeds = sorted(set(by_condition[left]) & set(by_condition[right]))
    differences = [by_condition[left][seed] - by_condition[right][seed] for seed in seeds]
    if len(differences) < 2:
        raise RuntimeError(f"Incomplete contrast {block}: {left} - {right}")
    center = mean(differences)
    spread = stdev(differences)
    half = float(stats.t.ppf(0.975, len(differences) - 1)) * spread / math.sqrt(len(differences))
    return {
        "block": block,
        "contrast": f"{left}_minus_{right}",
        "n": len(differences),
        "seeds": ";".join(map(str, seeds)),
        "mean_difference_pp": center,
        "sd_difference_pp": spread,
        "ci95_low_pp": center - half,
        "ci95_high_pp": center + half,
        "paired_t_p": float(stats.ttest_1samp(differences, 0.0).pvalue),
        "exact_sign_flip_p": exact_sign_flip(differences),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries with deterministic column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build all validation-protocol audit and plot tables."""

    args = parse_args()
    rows: list[dict[str, Any]] = []
    for cut in CUTS:
        for role, condition in (("act", "act"), ("lr", "exact_lr_only")):
            experiment = experiment_for_cut(role, cut)
            for seed in SELECTION_SEEDS:
                record = selected_record(
                    args.outputs_root, experiment, seed, condition, "selection"
                )
                record["cut"] = cut
                rows.append(record)
    for seed in SELECTION_SEEDS:
        rows.append(selected_record(args.outputs_root, "val_G", seed, "global", "selection"))
    for seed in CONFIRMATION_SEEDS:
        rows.append(selected_record(
            args.outputs_root, "val_G", seed, "global", "confirmation"
        ))
        rows.append(selected_record(
            args.outputs_root, "val_bnd_cut8_conf", seed, "act", "confirmation"
        ))
        rows.append(selected_record(
            args.outputs_root, "val_lr_cut8_conf", seed, "exact_lr_only", "confirmation"
        ))
        rows.append(selected_record(
            args.outputs_root, "val_lr_cut6_conf", seed, "selected_per_cut_lr_only", "confirmation"
        ))

    rows.sort(
        key=lambda row: (
            row["block"],
            int(row["cut"]) if row["cut"] != "" else 99,
            row["seed"],
            row["condition"],
        )
    )
    write_csv(args.output_dir / "validation_protocol_seed_level.csv", rows)

    depth_rows: list[dict[str, Any]] = []
    global_values = [
        float(row["test_accuracy_percent"])
        for row in rows
        if row["block"] == "selection" and row["condition"] == "global"
    ]
    for cut in CUTS:
        selected = [row for row in rows if row["block"] == "selection" and row.get("cut") == cut]
        act = [float(row["test_accuracy_percent"]) for row in selected if row["condition"] == "act"]
        lr = [float(row["test_accuracy_percent"]) for row in selected if row["condition"] == "exact_lr_only"]
        depth_rows.append(
            {
                "cut": cut,
                "boundary_mean": mean(act),
                "boundary_sd": stdev(act),
                "lr_only_mean": mean(lr),
                "lr_only_sd": stdev(lr),
                "global_mean": mean(global_values),
                "global_sd": stdev(global_values),
            }
        )
    write_csv(args.output_dir / "depth_profile_val_plot.csv", depth_rows)

    matched_rows: list[dict[str, Any]] = []
    for seed in (*SELECTION_SEEDS, *CONFIRMATION_SEEDS):
        block = "selection" if seed in SELECTION_SEEDS else "confirmation"
        candidates = [row for row in rows if row["block"] == block and row["seed"] == seed]
        condition_rows = {
            "G": next(row for row in candidates if row["condition"] == "global"),
            "L": next(
                row for row in candidates
                if row["condition"] == "act" and (row.get("cut") in {8, ""})
            ),
            "Rsame": next(
                row for row in candidates
                if row["condition"] == "exact_lr_only" and (row.get("cut") in {8, ""})
            ),
            "Rbest": next(
                row for row in candidates
                if (
                    row["condition"] == "selected_per_cut_lr_only"
                    or (row["condition"] == "exact_lr_only" and row.get("cut") == 6)
                )
            ),
        }
        for label, record in condition_rows.items():
            matched_rows.append(
                {
                    "seed": seed,
                    "block": block,
                    "condition": label,
                    "test_accuracy_percent": record["test_accuracy_percent"],
                    "validation_accuracy_percent": record["validation_accuracy_percent"],
                    "selected_epoch": record["selected_epoch"],
                    "metrics_csv": record["metrics_csv"],
                }
            )
    write_csv(args.output_dir / "matched_final_boundary_seed_plot.csv", matched_rows)

    summaries = [
        paired_summary(rows, "selection", "act", "exact_lr_only"),
        paired_summary(rows, "selection", "act", "global"),
        paired_summary(rows, "confirmation", "act", "exact_lr_only"),
        paired_summary(rows, "confirmation", "act", "selected_per_cut_lr_only"),
    ]
    write_csv(args.output_dir / "validation_protocol_paired_summary.csv", summaries)
    print(args.output_dir / "validation_protocol_seed_level.csv")
    print(args.output_dir / "depth_profile_val_plot.csv")
    print(args.output_dir / "matched_final_boundary_seed_plot.csv")
    print(args.output_dir / "validation_protocol_paired_summary.csv")


if __name__ == "__main__":
    main()
