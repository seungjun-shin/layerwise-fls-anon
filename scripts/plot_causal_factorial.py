#!/usr/bin/env python
"""Causal mechanism figure: 3x3 factorial heatmap of upstream-LR multiplier (m; upstream
LR = base/m) x activation multiplier (a), best test accuracy. Shows the LR main effect
(rows), the weak/non-monotone activation main effect (cols), and the LR x activation
interaction (activation reduction collapses at m=1.0 but peaks at m=0.0625).

Reads outputs/imp_analysis/causal_factorial.csv. Output: paper/figures/factorial_grid.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 11, "savefig.bbox": "tight", "savefig.dpi": 200})

MS = [1.0, 0.25, 0.0625]      # rows: upstream LR boost = 1/m  -> 1x, 4x, 16x
AS = [1.0, 0.25, 0.0625]      # cols: activation multiplier


def main():
    grid = {}
    with open("outputs/imp_analysis/causal_factorial.csv") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith(("m_lrcomp", "#")):
                continue
            m, a, acc = float(row[0]), float(row[1]), float(row[2])
            grid[(m, a)] = acc

    Z = [[grid.get((m, a), float("nan")) for a in AS] for m in MS]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(Z, cmap="viridis", aspect="auto")
    for i, m in enumerate(MS):
        for j, a in enumerate(AS):
            v = Z[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 57 else "black", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(AS)))
    ax.set_xticklabels([f"{a:g}" for a in AS])
    ax.set_yticks(range(len(MS)))
    ax.set_yticklabels([f"{a:g}\n({int(round(1/m))}× LR)" for m, a in [(m, m) for m in MS]])
    ax.set_xlabel("activation multiplier $a$")
    ax.set_ylabel("LR-comp multiplier $m$  (upstream LR $= \\eta/m$)")
    ax.set_title("LR × activation factorial (best test acc, %)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="best test acc (%)")
    out = "paper/figures/factorial_grid.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
