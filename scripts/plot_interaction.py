#!/usr/bin/env python
"""Early-late interaction heatmap (after-early x after-late boundary multipliers).
Regenerated standalone with boundary-multiplier axis labels (the make_plots path
keys on a different column name). Source: paper/tables/profile_comparison.csv,
experiment_41 early-late interaction. Output: paper/figures/early_late_interaction_heatmap.pdf
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def _pow2(v: float) -> str:
    e = round(math.log2(v))
    return f"$2^{{{e}}}$"


def main():
    rows = [r for r in csv.DictReader(open("paper/tables/profile_comparison.csv"))
            if "early_late" in r["experiment_name"].lower()]
    earlys = sorted({float(r["after_early_multiplier"]) for r in rows})
    lates = sorted({float(r["after_late_multiplier"]) for r in rows})
    grid = np.full((len(earlys), len(lates)), np.nan)
    for r in rows:
        i = earlys.index(float(r["after_early_multiplier"]))
        j = lates.index(float(r["after_late_multiplier"]))
        grid[i, j] = float(r["mean_best_test_accuracy"]) * 100.0

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(lates)))
    ax.set_xticklabels([_pow2(v) for v in lates])
    ax.set_yticks(range(len(earlys)))
    ax.set_yticklabels([_pow2(v) for v in earlys])
    ax.set_xlabel("After-late boundary multiplier $c$", fontsize=12)
    ax.set_ylabel("After-early boundary multiplier $c$", fontsize=12)
    ax.set_title("Early–late boundary interaction", fontsize=12)
    best = np.nanmax(grid)
    for i in range(len(earlys)):
        for j in range(len(lates)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if v == best else "normal",
                        color="white" if v < 60.0 else "black")
    cb = fig.colorbar(im, ax=ax, label="Mean test-selected accuracy (%)")
    cb.ax.tick_params(labelsize=10)
    out = "paper/figures/early_late_interaction_heatmap.pdf"
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}  (best cell {best:.2f})")


if __name__ == "__main__":
    main()
