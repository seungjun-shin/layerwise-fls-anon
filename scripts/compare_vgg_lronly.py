#!/usr/bin/env python
"""VGG19-BN generality check for M1: does the per-cut activation margin appear only at
the deepest cut (as on ResNet), or is the whole VGG depth profile an LR-fraction effect?

Compares the VGG boundary depth profile (imp_vgg_pos_cut{c}: activation 0.0625, 16eta
upstream) against the per-cut LR-only control (imp_vgg_pos_lronly_cut{c}: identity
activation, SAME 16eta upstream). 16-block axis, seeds 0-2.
"""
from __future__ import annotations

import csv
import glob
import math
import statistics as st

CUTS = [1, 3, 5, 7, 9, 11, 13, 15]
BASELINE = "imp_vgg_pos_baseline"


def best_accs(name: str) -> list[float]:
    out = []
    for m in glob.glob(f"outputs/{name}/**/seed_*/metrics.csv", recursive=True):
        try:
            r = list(csv.DictReader(open(m)))
        except OSError:
            continue
        if not r or max(int(float(x["epoch"])) for x in r) < 79:
            continue
        out.append(max(float(x["test_accuracy"]) for x in r) * 100.0)
    return out


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> None:
    base = best_accs(BASELINE)
    bm = st.mean(base) if base else float("nan")
    print(f"VGG19-BN baseline (no scaling): {bm:.2f}  (n={len(base)})\n")
    print(f"{'cut':>3} | {'boundary (n)':>16} | {'LR-only (n)':>16} | {'act.margin':>10} | {'Δvanilla(bnd)':>12}")
    print("-" * 72)
    xb, mb, xl, ml, margins = [], [], [], [], []
    for c in CUTS:
        b = best_accs(f"imp_vgg_pos_cut{c}")
        l = best_accs(f"imp_vgg_pos_lronly_cut{c}")
        bs = f"{st.mean(b):.2f}±{(st.stdev(b) if len(b) > 1 else 0):.2f} ({len(b)})" if b else "—"
        ls = f"{st.mean(l):.2f}±{(st.stdev(l) if len(l) > 1 else 0):.2f} ({len(l)})" if l else "—"
        mg = f"{st.mean(b) - st.mean(l):+.2f}" if b and l else "—"
        dv = f"{st.mean(b) - bm:+.2f}" if b else "—"
        if b:
            xb.append(c); mb.append(st.mean(b))
        if l:
            xl.append(c); ml.append(st.mean(l))
        if b and l:
            margins.append(st.mean(b) - st.mean(l))
        print(f"{c:>3} | {bs:>16} | {ls:>16} | {mg:>10} | {dv:>12}")
    print()
    print(f"boundary curve Spearman rho(depth,acc) = {spearman(xb, mb):.3f}  (n_cuts={len(xb)})")
    print(f"LR-only  curve Spearman rho(depth,acc) = {spearman(xl, ml):.3f}  (n_cuts={len(xl)})")
    if margins:
        print(f"activation margin: mean {st.mean(margins):+.2f} pp, "
              f"range [{min(margins):+.2f},{max(margins):+.2f}] over {len(margins)} cuts")
        print("  -> if margin ~0 except at the deepest cut, same phenomenon as ResNet.")


if __name__ == "__main__":
    main()
