#!/usr/bin/env python
"""Iso-accuracy representation separation figure (new key figure).

At MATCHED test accuracy, the after-late boundary intervention vs the LR-only control:
per-block paired Delta (boundary - LR-only) in effective rank and class alignment, for
ResNet18-CIFAR (cut 8, target 55%) and VGG19-BN (cut 15, target 58%). Late blocks separate
(rank down, alignment up) while early/mid blocks do not -- a representation difference that
is NOT explained by accuracy (the matched gap is ~0). Per-panel y-axes: the EFFECT SIZE is
architecture-dependent (much larger on ResNet); only the DIRECTION reproduces.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "savefig.dpi": 300})
DATA = (
    Path("paper/tables/iso_accuracy_resnet_plot.csv"),
    Path("paper/tables/iso_accuracy_vgg_plot.csv"),
)
PANELS = [("ResNet18-CIFAR", "ResNet18-CIFAR (cut 8)", "+0.10 pp"),
          ("VGG19-BN", "VGG19-BN (cut 15)", "+0.09 pp")]


fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.7), constrained_layout=True)
all_rows = [row for path in DATA for row in csv.DictReader(path.open())]
for col, (arch, title, gap) in enumerate(PANELS):
    for row, (metric, ylab) in enumerate([("effective_rank", r"$\Delta$ effective rank"),
                                          ("class_alignment", r"$\Delta$ class-conditional alignment")]):
        ax = axes[row][col]
        selected = [r for r in all_rows if r["architecture"] == arch and r["metric"] == metric]
        fracs = [float(r["block_depth"]) for r in selected]
        mus = [float(r["mean_difference"]) for r in selected]
        color = "#C0392B" if metric == "effective_rank" else "#2C3E50"
        ax.plot(fracs, mus, "-o", color=color, ms=4, lw=1.6)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.grid(True, color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        if row == 0:
            ax.set_title(f"{title}\nvalidation-matched test-accuracy difference {gap}")
        if row == 1:
            ax.set_xlabel("Block depth (shallow $\\to$ deep)")
        if col == 0:
            ax.set_ylabel(ylab)
out = "paper/figures/iso_acc_repr.pdf"
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
