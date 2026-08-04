#!/usr/bin/env python
"""Continuous depth profile, validation-controlled. Boundary scaling vs the
per-cut learning-rate-only control, validation-selected test accuracy (test acc at the
argmax-validation epoch), seeds 0-4, cuts 2-8. Boundary rho=0.93, LR-only rho=0.89;
the final boundary has the largest positive boundary-minus-LR-only margin.
Output: paper/figures/depth_profile.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "savefig.dpi": 300})
CUTS = [2, 3, 4, 5, 6, 7, 8]
COARSE = {2: "after-early", 4: "after-middle", 8: "after-late"}
REGION = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
C_LRO = "#8E44AD"
DATA = Path("paper/tables/depth_profile_val_plot.csv")


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
    for k, i in enumerate(o): r[i] = k
    return r


def spear(x, y):
    n = len(x); d2 = sum((a-b)**2 for a, b in zip(rank(x), rank(y))); return 1-6*d2/(n*(n*n-1))


def main():
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(DATA.open()))
    xb = xl = [int(r["cut"]) for r in rows]
    mb = [float(r["boundary_mean"]) for r in rows]
    sb = [float(r["boundary_sd"]) for r in rows]
    ml = [float(r["lr_only_mean"]) for r in rows]
    sl = [float(r["lr_only_sd"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.errorbar(xb, mb, yerr=sb, fmt="-o", color="#555555", ms=4, lw=1.8, capsize=3, zorder=3,
                label=r"Boundary scaling ($c=0.0625$)")
    ax.errorbar(xl, ml, yerr=sl, fmt="--s", color=C_LRO, ms=4, lw=1.6, capsize=3, zorder=2,
                label=r"Exact LR-only ($16\eta$)")
    # Five-seed validation-selected global reference from the same summary.
    GREF = float(rows[0]["global_mean"])
    ax.axhline(GREF, color="#2C3E50", ls=":", lw=1.4)
    ax.text(xb[0] + 0.08, GREF + 0.08, f"Global FLS reference = {GREF:.2f}", color="#2C3E50",
            va="bottom", ha="left", fontsize=6.5)
    for c in CUTS:
        if c in COARSE and c in xb:
            ax.scatter([c], [mb[xb.index(c)]], s=48, color=REGION[c], edgecolor="white",
                       linewidth=0.8, zorder=6)
    ax.set_xlabel("Cut position (shallow $\\to$ deep)")
    ax.set_ylabel("Validation-selected test accuracy (%)")
    ax.set_xticks(CUTS)
    series_legend = ax.legend(loc="upper left", frameon=True, borderaxespad=0.3,
                              handlelength=2.1, labelspacing=0.3)
    ax.add_artist(series_legend)
    named_cut_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               markerfacecolor=REGION[c], markeredgecolor="white",
               markeredgewidth=0.6, label=COARSE[c])
        for c in (2, 4, 8)
    ]
    ax.legend(handles=named_cut_handles, loc="lower right", ncol=3, frameon=True,
              fontsize=5.8, borderaxespad=0.35, borderpad=0.35,
              handletextpad=0.25, columnspacing=0.55)
    ax.grid(True, color="#DDDDDD", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35)
    out = "paper/figures/depth_profile.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"boundary rho={spear(xb,mb):.3f}  LR-only rho={spear(xl,ml):.3f}")
    bmap = dict(zip(xl, ml))
    for c, m in zip(xb, mb):
        print(f"  cut{c}: bnd {m:.2f}  lr {bmap.get(c, float('nan')):.2f}  margin {m-bmap.get(c, float('nan')):+.2f}")


if __name__ == "__main__":
    main()
