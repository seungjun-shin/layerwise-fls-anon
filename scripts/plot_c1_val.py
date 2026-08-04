#!/usr/bin/env python
"""Validation-controlled decoupled upstream-LR-only sweep (Fig. 10).

Reads paper/tables/factorial_val_summary.csv, the a=1.0 (identity-activation)
column: representation regions trained at k*eta (k = 1/m), head at eta, no
activation scaling. Plots test accuracy at the validation-selected epoch vs k,
over seeds 0-4, against the after-late boundary and global FLS references.
Output: paper/figures/c1_decoupled_lr_only.pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "savefig.dpi": 300})

KS = [4, 8, 16, 32, 64]   # upstream LR multiplier = 1/m
pts = {}
with open("paper/tables/factorial_val_summary.csv", newline="", encoding="utf-8") as handle:
    for r in csv.DictReader(handle):
        if abs(float(r["a"]) - 1.0) < 1e-9:
            k = round(1.0 / float(r["m"]))
            pts[k] = (
                float(r["test_at_best_val"]) * 100.0,
                float(r["std_test"]) * 100.0,
            )

ks = [k for k in KS if k in pts]
ys = [pts[k][0] for k in ks]
es = [pts[k][1] for k in ks]

# References are derived from their validation-controlled summary tables.
with open("paper/tables/global_fls_val_summary.csv", newline="", encoding="utf-8") as handle:
    global_rows = list(csv.DictReader(handle))
global_selected = max(global_rows, key=lambda row: float(row["val_accuracy"]))
global_reference = 100.0 * float(global_selected["test_at_best_val"])

with open("paper/tables/location_calib_val_summary.csv", newline="", encoding="utf-8") as handle:
    late_rows = [row for row in csv.DictReader(handle) if int(row["position"]) == 8]
late_selected = max(late_rows, key=lambda row: float(row["val_accuracy"]))
after_late_reference = 100.0 * float(late_selected["test_at_best_val"])
after_late_std = 100.0 * float(late_selected["std_test_at_best_val"])

kbest = max(ks, key=lambda k: pts[k][0])
ybest = pts[kbest][0]

fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.axhspan(
    after_late_reference - after_late_std,
    after_late_reference + after_late_std,
    color="#E67E22",
    alpha=0.12,
    zorder=0,
)
ax.axhline(
    after_late_reference,
    color="#E67E22",
    lw=2.0,
    label=f"After-late boundary ({after_late_reference:.2f})",
)
ax.axhline(
    global_reference,
    color="#4C78A8",
    ls="--",
    lw=1.4,
    label=f"Global FLS reference ({global_reference:.2f})",
)
ax.errorbar(range(len(ks)), ys, yerr=es, fmt="-o", color="#222222", ms=6, lw=1.8,
            capsize=4, zorder=3, label="LR-only (identity activations)")
jb = ks.index(kbest)
ax.annotate(f"tuned optimum\n$k={kbest}$ ({ybest:.2f})", xy=(jb, ybest), xytext=(jb + 0.55, ybest + 0.65),
            ha="left", fontsize=6.8, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"$\\eta\\times{k}$" for k in ks])
ax.set_xlabel("Upstream LR multiplier (identity activations)")
ax.set_ylabel("Validation-selected test accuracy (%)")
ax.grid(True, axis="y", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="lower center", framealpha=0.9, handlelength=2.0,
          labelspacing=0.25)
fig.tight_layout(pad=0.35)
out = "paper/figures/c1_decoupled_lr_only.pdf"
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
for k in ks:
    print(f"  k={k}: {pts[k][0]:.2f}+-{pts[k][1]:.2f}")
print(
    f"  tuned optimum k={kbest} ({ybest:.2f}); "
    f"after-late={after_late_reference:.2f}; global={global_reference:.2f}"
)
