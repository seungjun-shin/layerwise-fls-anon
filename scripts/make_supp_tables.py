#!/usr/bin/env python
"""Emit LaTeX supplementary tables for the NEW experiments (built from raw outputs/):
  S-VGG : VGG19-BN depth profile (best test acc per seed + mean+-sd) + Spearman
  S-TINY: Tiny-ImageNet depth profile + late dose-response (per seed + mean+-sd)
  S-DW  : depth-wise per-block representation profile (drift/rank/align, mean+-sd over 5 seeds)
Self-contained re-analysis; prints LaTeX to stdout.
"""
from __future__ import annotations
import csv, glob, statistics as st
from collections import defaultdict
from scipy.stats import spearmanr

MAXEP = 80


def best_by_seed(name):
    d = {}
    for m in glob.glob(f"outputs/{name}/**/seed_*/metrics.csv", recursive=True):
        r = list(csv.DictReader(open(m)))
        if not r or max(int(float(x["epoch"])) for x in r) < MAXEP - 1:
            continue
        s = int(m.split("seed_")[1].split("/")[0])
        d[s] = max(float(x["test_accuracy"]) for x in r) * 100
    return d


def fmt_row(label, d, seeds):
    cells = []
    for s in seeds:
        cells.append(f"{d[s]:.2f}" if s in d else "--")
    vals = [d[s] for s in seeds if s in d]
    ms = f"{st.mean(vals):.2f}$\\pm${st.stdev(vals):.2f}" if len(vals) > 1 else (f"{vals[0]:.2f}" if vals else "--")
    return f"{label} & " + " & ".join(cells) + f" & {ms} \\\\"


def depth_table(title, label, baseline, tmpl, cuts, maxb, seeds=(0, 1, 2)):
    print(f"\\begin{{table}}[t]\n\\centering\n\\caption{{{title}}}\n\\label{{{label}}}\n\\small")
    print("\\begin{tabular}{@{}l" + "c" * len(seeds) + "c@{}}\n\\toprule")
    print("Cut (depth) & " + " & ".join(f"seed {s}" for s in seeds) + " & mean$\\pm$s.d. \\\\\n\\midrule")
    b = best_by_seed(baseline)
    if b:
        print(fmt_row("baseline (no scaling)", b, seeds))
    raw_cuts, means = [], []
    for c in cuts:
        d = best_by_seed(tmpl.format(c))
        if not d:
            continue
        print(fmt_row(f"cut {c} ({c/maxb:.2f})", d, seeds))
        raw_cuts.append(c); means.append(st.mean(d.values() if False else [d[s] for s in seeds if s in d]))
    print("\\bottomrule\n\\end{tabular}")
    if len(raw_cuts) > 2:
        rho, p = spearmanr(raw_cuts, means)
        print(f"\\\\[2pt]\\footnotesize Spearman $\\rho={rho:.2f}$ ($p={p:.3g}$) between cut depth and mean best accuracy.")
    print("\\end{table}\n")


print("% ===== AUTO-GENERATED supplementary tables (new experiments) =====\n")

depth_table("VGG19-BN-CIFAR depth profile on CIFAR-100: best test accuracy (\\%) by scaling cut, "
            "fixed $c=0.0625$ with upstream learning-rate compensation, seeds 0--2.",
            "tab:supp_vgg_depth", "imp_vgg_pos_baseline", "imp_vgg_pos_cut{}",
            [1, 3, 5, 7, 9, 11, 13, 15], 15)

depth_table("Tiny-ImageNet (ResNet18-CIFAR) depth profile: best test accuracy (\\%) by scaling cut, "
            "fixed $c=0.0625$ with upstream learning-rate compensation, seeds 0--2.",
            "tab:supp_tiny_depth", "tiny_pos_baseline", "tiny_pos_cut{}",
            [2, 3, 4, 5, 6, 7, 8], 8)

# Tiny dose-response table
print("\\begin{table}[t]\n\\centering\n\\caption{Tiny-ImageNet late-boundary (position 8) dose-response: "
      "best test accuracy (\\%) by multiplier $c$, seeds 0--2.}\n\\label{tab:supp_tiny_dose}\n\\small")
print("\\begin{tabular}{@{}lccc c@{}}\n\\toprule\n$c$ & seed 0 & seed 1 & seed 2 & mean$\\pm$s.d. \\\\\n\\midrule")
dose = [("0.03125", "tiny_dose_c0p03125"), ("0.0625", "tiny_pos_cut8"), ("0.125", "tiny_dose_c0p125"),
        ("0.25", "tiny_dose_c0p25"), ("0.5", "tiny_dose_c0p5")]
for c, nm in dose:
    d = best_by_seed(nm)
    if d:
        print(fmt_row(c, d, (0, 1, 2)))
print("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# Depth-wise representation profile (per block, mean+-sd over 5 seeds)
def dw_load(name):
    by_block = defaultdict(list)
    for path in glob.glob(f"outputs/depth_profiles/{name}__seed*.csv"):
        for row in csv.DictReader(open(path)):
            by_block[int(row["block_index"])].append(row)
    return by_block

conds = [("imp_pos_baseline", "baseline"), ("imp_pos_cut2", "after-early (cut 2)"), ("imp_pos_cut8", "after-late (cut 8)")]
print("\\begin{table*}[t]\n\\centering\n\\caption{Depth-wise representation profile on CIFAR-100 (ResNet18-CIFAR): "
      "per-block CKA drift $D_l$, effective rank $r_{\\mathrm{eff}}$, and clean-label alignment $A$, "
      "mean$\\pm$s.d.\\ over seeds 0--4.}\n\\label{tab:supp_depthwise}\n\\small")
print("\\begin{tabular}{@{}l" + "ccc" * 3 + "@{}}\n\\toprule")
print(" & \\multicolumn{3}{c}{Drift $D_l$} & \\multicolumn{3}{c}{Eff.\\ rank $r_{\\mathrm{eff}}$} "
      "& \\multicolumn{3}{c}{Align.\\ $A$} \\\\")
print("Block & base & cut2 & cut8 & base & cut2 & cut8 & base & cut2 & cut8 \\\\\n\\midrule")
loaded = {c: dw_load(c) for c, _ in conds}
nb = max(max(loaded[c].keys()) for c, _ in conds) + 1
def cell(c, b, metric):
    rows = loaded[c].get(b, [])
    if not rows:
        return "--"
    v = [float(r[metric]) for r in rows]
    return f"{st.mean(v):.3f}" if metric != "effective_rank" else f"{st.mean(v):.0f}"
for b in range(nb):
    cols = []
    for metric in ["cka_drift", "effective_rank", "clean_label_alignment"]:
        cols += [cell(c, b, metric) for c, _ in conds]
    print(f"{b} & " + " & ".join(cols) + " \\\\")
print("\\bottomrule\n\\end{tabular}\n\\end{table*}")
