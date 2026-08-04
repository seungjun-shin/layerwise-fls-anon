#!/usr/bin/env python
"""VGG iso-accuracy representation control (pillar 2 generality).

Compares VGG boundary cut15 (activation 0.0625) against the HEALTHIEST tuned LR-only
candidate (identity activation, lower upstream LR multiplier) at matched accuracy. The
naive 16x same-cut LR-only collapses on VGG (54.96, below the matching range), so we use
the LR-only multiplier candidate that reaches the highest accuracy as the fair control.

Reports, at each matched-accuracy target: the accuracy gap and the per-block paired Delta
(boundary - LR-only) for drift / eff.rank / align, focusing on late blocks (>=13 of 16).
"""
from __future__ import annotations

import csv
import glob
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

BOUNDARY = "val_vgg_bnd_cut15_iso"
CANDIDATES = ["val_vgg_lr_cut15_2x_iso"]
SEEDS = [0, 1, 2]
TARGETS = [58.0]
LATE_BLOCK = 13
OUTDIR = Path("outputs/depth_profiles_isoacc_vgg_val")
PLOT_TABLE = Path("paper/tables/iso_accuracy_vgg_plot.csv")
T975 = {2: 4.303, 1: 12.706}


def ckpt_name(t):
    return f"checkpoint_atacc_{t:g}.pt"


def latest_run(exp, seed, t):
    cands = sorted(Path("outputs", exp).glob(f"*/seed_{seed}/{ckpt_name(t)}"))
    return cands[-1].parent if cands else None


def best_acc_by_seed(exp):
    out = {}
    for m in glob.glob(f"outputs/{exp}/**/seed_*/metrics.csv", recursive=True):
        mm = re.search(r"/seed_(\d+)/", m)
        if not mm:
            continue
        r = list(csv.DictReader(open(m)))
        if not r or max(int(float(x["epoch"])) for x in r) < 79:
            continue
        out[int(mm.group(1))] = max(float(x["test_accuracy"]) for x in r) * 100.0
    return out


def ensure_diag(exp, t):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        out = OUTDIR / f"{exp}__seed{s}__acc{t:g}.csv"
        if out.exists():
            continue
        rd = latest_run(exp, s, t)
        if rd is None:
            continue
        subprocess.run([sys.executable, "scripts/compute_depth_profile.py", "--run-dir", str(rd),
                        "--checkpoint", ckpt_name(t), "--split", "train", "--max-samples", "256",
                        "--output", str(out)], check=True)


def load_blocks(exp, t):
    bb = {}
    for s in SEEDS:
        p = OUTDIR / f"{exp}__seed{s}__acc{t:g}.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            b = int(row["block_index"])
            d = bb.setdefault(b, {"drift": [], "erank": [], "align": []})
            d["drift"].append(float(row["cka_drift"]))
            d["erank"].append(float(row["effective_rank"]))
            d["align"].append(float(row["clean_label_alignment"]))
    return bb


def paired(b, l):
    n = min(len(b), len(l))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diffs = [b[i] - l[i] for i in range(n)]
    mu, sd = st.mean(diffs), st.stdev(diffs)
    return mu, mu - T975.get(n - 1, 4.303) * sd / math.sqrt(n), mu + T975.get(n - 1, 4.303) * sd / math.sqrt(n), n


def main():
    plot_rows = []
    # 1) report candidate accuracies, pick healthiest LR-only
    print("LR-only candidate best accuracies (VGG cut15):")
    cand_acc = {}
    for c in CANDIDATES:
        a = list(best_acc_by_seed(c).values())
        cand_acc[c] = st.mean(a) if a else float("-inf")
        print(f"  {c}: {('%.2f' % cand_acc[c]) if a else '—'}  (n={len(a)})")
    ba = list(best_acc_by_seed(BOUNDARY).values())
    print(f"  {BOUNDARY}: {('%.2f' % st.mean(ba)) if ba else '—'}  (n={len(ba)})")
    lronly = max(CANDIDATES, key=lambda c: cand_acc[c])
    print(f"\n==> healthiest LR-only control: {lronly} ({cand_acc[lronly]:.2f})\n")

    for t in TARGETS:
        ensure_diag(BOUNDARY, t)
        ensure_diag(lronly, t)
        bacc = [a for s in SEEDS if (rd := latest_run(BOUNDARY, s, t)) for a in [_acc(rd, t)] if a]
        lacc = [a for s in SEEDS if (rd := latest_run(lronly, s, t)) for a in [_acc(rd, t)] if a]
        gap = f"{st.mean(bacc) - st.mean(lacc):+.2f}" if bacc and lacc else "—"
        print("=" * 80)
        print(f"TARGET {t:g}%  matched acc: boundary {st.mean(bacc):.2f} | LR-only {st.mean(lacc):.2f} | gap {gap} pp"
              if bacc and lacc else f"TARGET {t:g}%  (incomplete)")
        bb, lb = load_blocks(BOUNDARY, t), load_blocks(lronly, t)
        if not (bb and lb):
            print("  diagnostics incomplete")
            continue
        for b in sorted(set(bb) & set(lb)):
            if b < LATE_BLOCK:
                continue
            dd, de, da = (paired(bb[b][k], lb[b][k]) for k in ("drift", "erank", "align"))
            if any(result[3] != len(SEEDS) for result in (dd, de, da)):
                raise RuntimeError(f"Incomplete three-seed VGG diagnostics at block {b}")
            sep = (dd[1] > 0) + (de[2] < 0) + (da[1] > 0)
            print(f"  blk {b}: Δdrift {dd[0]:+.3f}[{dd[1]:+.3f},{dd[2]:+.3f}]  "
                  f"Δrank {de[0]:+.2f}[{de[1]:+.2f},{de[2]:+.2f}]  "
                  f"Δalign {da[0]:+.3f}[{da[1]:+.3f},{da[2]:+.3f}]  -> {sep}/3")
            for metric, result in (("effective_rank", de), ("class_alignment", da)):
                plot_rows.append(
                    {
                        "architecture": "VGG19-BN",
                        "metric": metric,
                        "block_depth": b / 15.0,
                        "mean_difference": result[0],
                        "ci95_halfwidth": (result[2] - result[1]) / 2.0,
                    }
                )
    if not plot_rows:
        raise RuntimeError("No complete VGG iso-accuracy rows were produced")
    PLOT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with PLOT_TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    print(PLOT_TABLE)


def _acc(run_dir, t):
    import torch
    try:
        ck = torch.load(run_dir / ckpt_name(t), map_location="cpu", weights_only=False)
    except Exception:
        return None
    m = ck.get("metrics") if isinstance(ck, dict) else None
    return float(m["test_accuracy"]) * 100.0 if m and "test_accuracy" in m else None


if __name__ == "__main__":
    main()
