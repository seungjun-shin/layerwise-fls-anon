#!/usr/bin/env python
"""Aggregate the frozen ten-pair controlled cells used in Main Table 1.

Seeds 5--9 reuse the validation-selected controlled runs reported before the
extension.  Seeds 10--14 come from the frozen M7 extension.  Checkpoints are
selected by the earliest epoch attaining maximum validation accuracy; the test
accuracy from that epoch is used only for reporting.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
PAPER_WORKSPACE = ROOT / "paper_reproduction"
EXPECTED_SEEDS = set(range(5, 15))


def arithmetic_mean(values: Any) -> float:
    """Return a stable arithmetic mean without importing ``statistics``."""

    sequence = list(values)
    return math.fsum(sequence) / len(sequence)


def sample_stdev(values: Any) -> float:
    """Return the Bessel-corrected sample standard deviation."""

    sequence = list(values)
    center = arithmetic_mean(sequence)
    return math.sqrt(
        math.fsum((value - center) ** 2 for value in sequence) / (len(sequence) - 1)
    )


def parse_args() -> argparse.Namespace:
    """Parse the external raw-metric root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing the four paper_table1_* families.",
    )
    return parser.parse_args()


def validation_selected_row(path: Path) -> dict[str, str]:
    """Return the earliest row attaining maximum validation accuracy."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty metrics file: {path}")
    maximum = max(float(row["val_accuracy"]) for row in rows)
    return min(
        (row for row in rows if float(row["val_accuracy"]) == maximum),
        key=lambda row: int(row["epoch"]),
    )


def extension_rows(raw_root: Path, family: str, role: str) -> list[dict[str, Any]]:
    """Load and validate extension seeds 10--14 for one architecture and arm."""

    experiment = f"paper_table1_{family}_controlled_{role}"
    paths = sorted((raw_root / experiment).glob("*/seed_*/metrics.csv"))
    output: list[dict[str, Any]] = []
    for path in paths:
        selected = validation_selected_row(path)
        seed = int(selected["seed"])
        if seed not in range(10, 15):
            continue
        output.append(
            {
                "architecture": "ResNet18-CIFAR" if family == "resnet" else "VGG19-BN",
                "seed": seed,
                "role": role,
                "source_block": "M7 extension",
                "test_accuracy_percent": 100.0 * float(selected["test_accuracy"]),
                "validation_accuracy_percent": 100.0 * float(selected["val_accuracy"]),
                "checkpoint_epoch": int(selected["epoch"]),
                "source_metrics": str(path),
            }
        )
    found = {int(row["seed"]) for row in output}
    if found != set(range(10, 15)):
        raise RuntimeError(f"{experiment}: expected seeds 10--14, found {sorted(found)}")
    return output


def legacy_resnet_rows() -> list[dict[str, Any]]:
    """Load ResNet confirmation seeds 5--9 directly from raw run metrics."""

    families = {
        "act": ROOT / "outputs/val_bnd_cut8_conf",
        "lr": ROOT / "outputs/val_lr_cut8_conf",
    }
    output: list[dict[str, Any]] = []
    for role, directory in families.items():
        for path in sorted(directory.glob("*/seed_*/metrics.csv")):
            selected = validation_selected_row(path)
            seed = int(selected["seed"])
            if seed not in range(5, 10):
                continue
            output.append(
                {
                    "architecture": "ResNet18-CIFAR",
                    "seed": seed,
                    "role": role,
                    "source_block": "held-out confirmation",
                    "test_accuracy_percent": 100.0 * float(selected["test_accuracy"]),
                    "validation_accuracy_percent": 100.0 * float(selected["val_accuracy"]),
                    "checkpoint_epoch": int(selected["epoch"]),
                    "source_metrics": str(path),
                }
            )
    return output


def legacy_vgg_rows() -> list[dict[str, Any]]:
    """Load the frozen VGG seed 5--9 ACT and exact-LR runs."""

    families = {
        "act": ROOT / "outputs/val_vgg_bnd_cut15_rep",
        "lr": ROOT / "outputs/val_vgg_lr_cut15_16x_rep",
    }
    output: list[dict[str, Any]] = []
    for role, directory in families.items():
        for path in sorted(directory.glob("*/seed_*/metrics.csv")):
            selected = validation_selected_row(path)
            seed = int(selected["seed"])
            if seed not in range(5, 10):
                continue
            output.append(
                {
                    "architecture": "VGG19-BN",
                    "seed": seed,
                    "role": role,
                    "source_block": "original held-out",
                    "test_accuracy_percent": 100.0 * float(selected["test_accuracy"]),
                    "validation_accuracy_percent": 100.0 * float(selected["val_accuracy"]),
                    "checkpoint_epoch": int(selected["epoch"]),
                    "source_metrics": str(path),
                }
            )
    return output


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Return the exact two-sided sign-flip p-value on the paired mean."""

    observed = abs(arithmetic_mean(differences))
    statistics = [
        abs(
            arithmetic_mean(
                [sign * value for sign, value in zip(signs, differences, strict=True)]
            )
        )
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-12 for value in statistics) / len(statistics)


def validate_pairs(rows: list[dict[str, Any]]) -> None:
    """Require exactly one ACT and LR observation for each expected seed."""

    keys = [(row["architecture"], int(row["seed"]), row["role"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate architecture/seed/role observations")
    for architecture in ("ResNet18-CIFAR", "VGG19-BN"):
        for role in ("act", "lr"):
            seeds = {
                int(row["seed"])
                for row in rows
                if row["architecture"] == architecture and row["role"] == role
            }
            if seeds != EXPECTED_SEEDS:
                raise RuntimeError(
                    f"{architecture}/{role}: expected seeds 5--14, found {sorted(seeds)}"
                )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one paired summary per architecture."""

    output: list[dict[str, Any]] = []
    for architecture in ("ResNet18-CIFAR", "VGG19-BN"):
        selected = [row for row in rows if row["architecture"] == architecture]
        by_role = {
            role: {
                int(row["seed"]): float(row["test_accuracy_percent"])
                for row in selected
                if row["role"] == role
            }
            for role in ("act", "lr")
        }
        differences = [by_role["act"][seed] - by_role["lr"][seed] for seed in range(5, 15)]
        average = arithmetic_mean(differences)
        sd = sample_stdev(differences)
        half_width = float(t.ppf(0.975, len(differences) - 1)) * sd / math.sqrt(len(differences))
        output.append(
            {
                "architecture": architecture,
                "n": len(differences),
                "seeds": ";".join(map(str, range(5, 15))),
                "mean_act_percent": arithmetic_mean(by_role["act"].values()),
                "sd_act_percent": sample_stdev(by_role["act"].values()),
                "mean_lr_percent": arithmetic_mean(by_role["lr"].values()),
                "sd_lr_percent": sample_stdev(by_role["lr"].values()),
                "mean_difference_pp": average,
                "sd_difference_pp": sd,
                "ci95_low_pp": average - half_width,
                "ci95_high_pp": average + half_width,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "all_differences_same_sign": len({value > 0 for value in differences}) == 1,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with stable field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build seed-level and summary artifacts."""

    args = parse_args()
    rows = legacy_resnet_rows() + legacy_vgg_rows()
    for family in ("resnet", "vgg"):
        for role in ("act", "lr"):
            rows.extend(extension_rows(args.raw_root, family, role))
    rows.sort(key=lambda row: (row["architecture"], int(row["seed"]), row["role"]))
    validate_pairs(rows)
    summary = summarize(rows)
    destination = PAPER_WORKSPACE / "analysis/statistics"
    write_csv(destination / "table1_controlled_n10_seed_level.csv", rows)
    write_csv(destination / "table1_controlled_n10_summary.csv", summary)
    print(destination / "table1_controlled_n10_seed_level.csv")
    print(destination / "table1_controlled_n10_summary.csv")


if __name__ == "__main__":
    main()
