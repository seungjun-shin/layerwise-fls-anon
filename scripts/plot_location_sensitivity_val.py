#!/usr/bin/env python
"""Validation-controlled boundary-location sensitivity (Fig 3).

Reads paper/tables/location_calib_val_summary.csv (seed-0 validation grid): for each
boundary position (after-early/middle/late) and multiplier c, the TEST accuracy and
loss at the epoch maximizing held-out validation accuracy. The after-late operating
point used in the paper is the multiplier maximizing validation accuracy.
Output: paper/figures/location_sensitivity_curves.pdf
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

POS = {2: "after-early", 4: "after-middle", 8: "after-late"}
COL = {2: "#4C78A8", 4: "#F58518", 8: "#2E7D32"}
MARKER = {2: "o", 4: "s", 8: "^"}
LINESTYLE = {2: "-", 4: "--", 8: "-."}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

df = pd.read_csv("paper/tables/location_calib_val_summary.csv")
# validation-selected global FLS reference (argmax-val cell of the global grid)
g = pd.read_csv("paper/tables/global_fls_val_summary.csv")
gref = 100.0 * g.loc[g["val_accuracy"].idxmax(), "test_at_best_val"]

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
has_std = "std_test_at_best_val" in df.columns
for pos in (2, 4, 8):
    sub = df[df["position"].eq(pos)].sort_values("output_multiplier")
    yerr = 100.0 * sub["std_test_at_best_val"] if has_std else None
    axes[0].errorbar(sub["output_multiplier"], 100.0 * sub["test_at_best_val"], yerr=yerr,
                     marker=MARKER[pos], linestyle=LINESTYLE[pos], lw=1.7,
                     color=COL[pos], capsize=2.5, markersize=4.5, label=POS[pos])
    axes[1].plot(sub["output_multiplier"], sub["test_loss_at_best_val"],
                 marker=MARKER[pos], linestyle=LINESTYLE[pos], lw=1.7,
                 color=COL[pos], markersize=4.5, label=POS[pos])
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Boundary multiplier $c$")
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("Validation-selected test accuracy (%)")
axes[1].set_ylabel("Validation-selected test loss")
axes[0].set_title("(a) Accuracy")
axes[1].set_title("(b) Loss")
axes[0].axhline(gref, color="#4C78A8", ls="--", lw=1.2, label=f"Global FLS reference ({gref:.1f}%)")
axes[0].legend(loc="lower left", ncol=2, columnspacing=0.8, handlelength=2.0)
axes[1].legend(loc="upper left", handlelength=2.4)
fig.tight_layout(pad=0.4, w_pad=1.2)
out = Path("paper/figures/location_sensitivity_curves.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}  (global FLS ref = {gref:.2f}%)")
for pos in (2, 4, 8):
    sub = df[df["position"].eq(pos)]
    bv = sub.loc[sub["val_accuracy"].idxmax()]
    print(f"  {POS[pos]}: val-selected c={bv['output_multiplier']}  test={100*bv['test_at_best_val']:.2f}%")
