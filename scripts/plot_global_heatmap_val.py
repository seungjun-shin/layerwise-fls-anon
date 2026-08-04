#!/usr/bin/env python
"""Validation-controlled global-FLS heatmap (Fig 2).

Reads the seed-0 validation grid summary (paper/tables/global_fls_val_summary.csv):
each cell shows the TEST accuracy at the epoch that maximizes held-out validation
accuracy. The boxed reference cell is selected by VALIDATION accuracy (argmax over
the grid), so the operating point is chosen entirely on held-out data.
Output: paper/figures/global_fls_heatmap_experiment_42_global_reference_rerun_cifar100.pdf
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 7.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

SRC = "paper/tables/global_fls_val_summary.csv"
K = 6.4e-4


def p2(v):
    return f"$2^{{{round(math.log2(v))}}}$"


def lrtick(v):
    return f"$2^{{{round(math.log2(v / K))}}}k$"


df = pd.read_csv(SRC)
# cell display value: test accuracy at the validation-selected epoch
disp = df.pivot_table(index="learning_rate", columns="global_output_multiplier",
                      values="test_at_best_val", aggfunc="mean")
# selection landscape: validation accuracy
valp = df.pivot_table(index="learning_rate", columns="global_output_multiplier",
                      values="val_accuracy", aggfunc="mean")

fig, ax = plt.subplots(figsize=(3.55, 2.85))
image = ax.imshow(disp.values, aspect="auto", origin="lower", cmap="viridis")
ax.set_xticks(range(len(disp.columns)))
ax.set_xticklabels([p2(float(v)) for v in disp.columns], rotation=0)
ax.set_yticks(range(len(disp.index)))
ax.set_yticklabels([lrtick(float(v)) for v in disp.index])
ax.set_xlabel("Global output multiplier $c$")
ax.set_ylabel(r"Base learning rate ($k=6.4{\times}10^{-4}$)")
for i, lr in enumerate(disp.index):
    for j, c in enumerate(disp.columns):
        val = disp.loc[lr, c]
        if pd.notna(val):
            ax.text(j, i, f"{float(val) * 100:.1f}", ha="center", va="center",
                    fontsize=6.2, color="white" if float(val) < 0.36 else "black")

# box the cell selected by VALIDATION accuracy (argmax over the grid)
imax = valp.values.argmax()
isel, jsel = divmod(imax, valp.shape[1])
ax.add_patch(mpatches.Rectangle((jsel - 0.5, isel - 0.5), 1, 1, fill=False,
                                edgecolor="#D62728", linewidth=2.2, zorder=5))
cb = fig.colorbar(image, ax=ax, label="Validation-selected test accuracy (%)")
cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{100.0 * v:.0f}"))
cb.ax.tick_params(labelsize=7)
cb.set_label("Validation-selected test accuracy (%)", fontsize=7.5)
fig.tight_layout(pad=0.25)
out = "paper/figures/global_fls_heatmap_experiment_42_global_reference_rerun_cifar100.pdf"
Path("paper/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(out)
selc, sellr = disp.columns[jsel], disp.index[isel]
print(f"wrote {out}")
print(f"  val-selected cell: c={selc} base_lr={sellr} (eff={sellr/selc:.5f})  "
      f"val={valp.iloc[isel, jsel]*100:.2f}  test={disp.iloc[isel, jsel]*100:.2f}")
