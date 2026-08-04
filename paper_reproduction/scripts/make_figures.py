#!/usr/bin/env python
"""Create versioned paper figures from frozen seed-level CSV artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy",
        default=str(WORKSPACE / "seed_tables" / "practical_main_confirmatory_seed_level.csv"),
    )
    parser.add_argument(
        "--mechanism",
        default=str(WORKSPACE / "analysis" / "mechanism" / "mechanism_seed_level.csv"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[int, dict[str, str]]]:
    output: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        output.setdefault(row["condition"], {})[int(row["seed"])] = row
    return output


def save_versioned(fig: plt.Figure, stem: str, timestamp: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".pdf", ".png"):
        for name in (stem, f"{stem}_{timestamp}"):
            path = WORKSPACE / "figures" / f"{name}{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
    return paths


def accuracy_figure(rows: list[dict[str, str]], timestamp: str) -> list[Path]:
    data = grouped(rows)
    order = ["P-REF", "P-LR-FINAL", "P-ACT-FINAL"]
    labels = ["Reference", "LR-only", "Activation"]
    seeds = sorted(set.intersection(*(set(data[name]) for name in order)))
    values = np.asarray(
        [[float(data[name][seed]["test_accuracy_percent"]) for name in order] for seed in seeds]
    )
    fig, axis = plt.subplots(figsize=(3.5, 3.0))
    x = np.arange(len(order))
    for row in values:
        axis.plot(x, row, color="#8a8a8a", alpha=0.48, linewidth=0.8, marker="o", markersize=3)
    means = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    half = stats.t.ppf(0.975, len(seeds) - 1) * sem
    axis.errorbar(
        x,
        means,
        yerr=half,
        fmt="D",
        color="#a51c30",
        linewidth=1.6,
        capsize=3,
        markersize=5,
        label="Mean and 95% t CI",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Test top-1 accuracy (%)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths = save_versioned(fig, "practical_main_paired_accuracy", timestamp)
    plt.close(fig)
    return paths


def mechanism_figure(rows: list[dict[str, str]], timestamp: str) -> list[Path]:
    data = grouped(rows)
    metrics = [
        ("feature_l2_norm", "Feature L2 norm"),
        ("classifier_input_gradient_l2_norm", "Classifier-input grad L2"),
        ("logit_l2_norm", "Logit L2 norm"),
        ("classifier_weight_effective_rank", "Classifier effective rank"),
    ]
    seeds = sorted(set(data["P-ACT-FINAL"]) & set(data["P-LR-FINAL"]))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        values = np.asarray(
            [
                [
                    float(data["P-LR-FINAL"][seed][metric]),
                    float(data["P-ACT-FINAL"][seed][metric]),
                ]
                for seed in seeds
            ]
        )
        for row in values:
            axis.plot([0, 1], row, color="#777777", alpha=0.52, linewidth=0.8, marker="o", markersize=3)
        means = values.mean(axis=0)
        axis.plot([0, 1], means, color="#a51c30", marker="D", linewidth=1.8, markersize=5)
        axis.set_xticks([0, 1], ["LR-only", "Activation"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    paths = save_versioned(fig, "practical_main_mechanism", timestamp)
    plt.close(fig)
    return paths


def append_manifest(paths: list[Path], sources: list[Path]) -> None:
    """Register only immutable timestamped figures, not mutable latest aliases."""
    manifest = WORKSPACE / "manifests" / "artifact_manifest.csv"
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for path in paths:
            if re.search(r"_\d{8}_\d{6}$", path.stem) is None:
                continue
            writer.writerow(
                [
                    path.stem,
                    str(path.relative_to(ROOT)),
                    "figure",
                    ";".join(str(source.relative_to(ROOT)) for source in sources),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    str(Path(__file__).relative_to(ROOT)),
                    "Individual seeds visible; mean interval is a 95% t interval.",
                ]
            )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    accuracy_path = Path(args.accuracy)
    mechanism_path = Path(args.mechanism)
    outputs = accuracy_figure(read_rows(accuracy_path), timestamp)
    outputs += mechanism_figure(read_rows(mechanism_path), timestamp)
    append_manifest(outputs, [accuracy_path, mechanism_path])
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
