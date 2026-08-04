#!/usr/bin/env python
"""Update-pressure figure: grouped bars of the relative update-to-head ratio per
network region for the matched global vs after-late boundary pair (seeds 0-4).
Source: outputs/_val_preserve_all_20260801/diagnostics/update_pressure/confound4way_update_pressure_summary.csv
Output: paper/figures/update_pressure_ratios.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "savefig.dpi": 300})

SRC = "outputs/_val_preserve_all_20260801/diagnostics/update_pressure/confound4way_update_pressure_summary.csv"
REGIONS = ["early", "middle", "late"]
SERIES = [("matched_global", "Global FLS reference", "#4C78A8"),
          ("boundary_upstream_comp", "After-late boundary", "#F58518")]


def main():
    rows = list(csv.DictReader(open(SRC)))
    data = {(r["condition"], r["region"]):
            (float(r["relative_update_ratio_to_head_across_runs_mean"]),
             float(r["relative_update_ratio_to_head_across_runs_std"]))
            for r in rows}

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = np.arange(len(REGIONS))
    w = 0.38
    hatches = ["///", "..."]
    for i, (cond, label, color) in enumerate(SERIES):
        means = [data[(cond, reg)][0] for reg in REGIONS]
        stds = [data[(cond, reg)][1] for reg in REGIONS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
               color=color, label=label, edgecolor="#333333", linewidth=0.45,
               hatch=hatches[i])
    ax.axhline(1.0, color="#555555", ls="--", lw=1.0)
    ax.text(2.45, 1.25, "head $=1$", ha="right", va="bottom",
            fontsize=6.8, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in REGIONS])
    ax.set_xlabel("Network region")
    ax.set_ylabel("Relative update / head")
    ax.legend(loc="upper left", frameon=True, handlelength=1.8,
              labelspacing=0.25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35)
    out = "paper/figures/update_pressure_ratios.pdf"
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
