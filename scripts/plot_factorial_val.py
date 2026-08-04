#!/usr/bin/env python
"""Validation-controlled 3x3 factorial heatmap (Fig. 11).

Reads paper/tables/factorial_val_summary.csv: for each (m, a) cell, the test
accuracy at the validation-selected epoch, averaged over seeds 0-4. Rows are the
upstream-LR-compensation multiplier m (upstream LR = base/m), columns the
after-late activation multiplier a. Output: paper/figures/factorial_grid.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03, "savefig.dpi": 300})

MS = [1.0, 0.25, 0.0625]   # rows: upstream LR boost = 1/m -> 1x, 4x, 16x
AS = [1.0, 0.25, 0.0625]   # cols: activation multiplier

grid = {}
for r in csv.DictReader(open("paper/tables/factorial_val_summary.csv")):
    grid[(float(r["m"]), float(r["a"]))] = float(r["test_at_best_val"]) * 100.0

Z = [[grid.get((m, a), float("nan")) for a in AS] for m in MS]

fig, ax = plt.subplots(figsize=(3.45, 2.75))
im = ax.imshow(Z, cmap="viridis", aspect="auto")
for i, m in enumerate(MS):
    for j, a in enumerate(AS):
        v = Z[i][j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 57 else "black", fontsize=8, fontweight="bold")
ax.set_xticks(range(len(AS)))
ax.set_xticklabels([f"{a:g}" for a in AS])
ax.set_yticks(range(len(MS)))
ax.set_yticklabels([f"{m:g}\n({int(round(1/m))}× LR)" for m in MS])
ax.set_xlabel("Activation multiplier $a$")
ax.set_ylabel("LR-compensation multiplier $m$  (upstream LR $= \\eta/m$)")
colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
colorbar.set_label("Validation-selected test accuracy (%)", fontsize=7)
colorbar.ax.tick_params(labelsize=7)
fig.tight_layout(pad=0.35)
out = "paper/figures/factorial_grid.pdf"
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
# marginal means for the manuscript
row_means = [sum(Z[i]) / len(AS) for i in range(len(MS))]
col_means = [sum(Z[i][j] for i in range(len(MS))) / len(MS) for j in range(len(AS))]
print("  row means (over a):", [f"{x:.2f}" for x in row_means])
print("  col means (over m):", [f"{x:.2f}" for x in col_means])
print(f"  best cell (m=0.0625,a=0.0625) = {grid[(0.0625,0.0625)]*100 if False else Z[2][2]:.2f}")
