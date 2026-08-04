#!/usr/bin/env python
"""Depth-wise representation profile figure.

Reads the per-block diagnostics produced by compute_depth_profile.py
(outputs/depth_profiles/<exp>__seed<seed>.csv) and plots, as a function of
block depth (0=stem ... 8=final block), three representation diagnostics --
CKA drift from initialization, effective rank, and clean-label class
alignment -- for the global-FLS reference and two boundary interventions
(after-early cut2, after-late cut8). Mean over seeds; shaded band is sample
std (ddof=1). This resolves the coarse early/middle/late diagnostics into a
continuous depth axis, showing that the after-late intervention amplifies the
late-stage signature (more drift, lower rank, stronger alignment) precisely in
the deep blocks. Output: paper/figures/depth_repr_profile.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03, "savefig.dpi": 300})

CONDITIONS = [
    ("global", "Global FLS reference", "#7f7f7f", "-o"),
    ("cut2", "After-early (cut 2)", "#1f77b4", "-s"),
    ("cut8", "After-late (cut 8)", "#2ca02c", "-^"),
]
METRICS = [
    ("cka_drift", "CKA drift from initialization"),
    ("effective_rank", "Effective rank"),
    ("clean_label_alignment", "Class-conditional alignment"),
]


DATA_DIR = Path("outputs/_summaries/validation_depth_repr_20260803")


def aggregate_rows() -> list[dict[str, float | int | str]]:
    """Aggregate validation-selected per-seed diagnostics by condition and block."""
    aggregate: list[dict[str, float | int | str]] = []
    for condition, _, _, _ in CONDITIONS:
        paths = sorted(DATA_DIR.glob(f"{condition}_seed*.csv"))
        if len(paths) != 5:
            raise RuntimeError(
                f"Expected five validation-selected files for {condition}, found {len(paths)}"
            )
        seed_rows = [list(csv.DictReader(path.open())) for path in paths]
        for block in range(9):
            matching = [
                row for rows in seed_rows for row in rows
                if int(row["block_index"]) == block
            ]
            if len(matching) != 5:
                raise RuntimeError(
                    f"Expected five rows for {condition} block {block}, found {len(matching)}"
                )
            summary: dict[str, float | int | str] = {
                "condition": condition,
                "block": block,
            }
            for metric, _ in METRICS:
                values = np.array([float(row[metric]) for row in matching])
                summary[metric] = float(values.mean())
                summary[f"{metric}_sd"] = float(values.std(ddof=1))
            aggregate.append(summary)
    return aggregate


def main() -> None:
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    rows = aggregate_rows()

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), constrained_layout=True)
    for panel_index, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
        for name, label, color, fmt in CONDITIONS:
            selected = [r for r in rows if r["condition"] == name]
            blocks = [int(r["block"]) for r in selected]
            means = np.array([float(r[metric]) for r in selected])
            sds = np.array([float(r[f"{metric}_sd"]) for r in selected])
            ax.plot(blocks, means, fmt, color=color, ms=4, lw=1.7, label=label)
            ax.fill_between(blocks, means - sds, means + sds, color=color, alpha=0.15)
        ax.set_xlabel("Block index (shallow $\\to$ deep)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(ord('a') + panel_index)}) {ylabel}")
        ax.set_xticks(range(0, 9))
        ax.grid(True, color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9,
                   handlelength=1.8, labelspacing=0.25)
    out = "paper/figures/depth_repr_profile.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    for name, label, _, _ in CONDITIONS:
        selected = [r for r in rows if r["condition"] == name and int(r["block"]) == 8][0]
        print("  " + label + " @block8: " + " ".join(
            f"{metric}={float(selected[metric]):.3f}" for metric, _ in METRICS))


if __name__ == "__main__":
    main()
