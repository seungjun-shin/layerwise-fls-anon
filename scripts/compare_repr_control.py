#!/usr/bin/env python
"""Representation-signature control (M1, decisive test).

Compares the after-late BOUNDARY intervention (imp_pos_cut8: activation 0.0625,
upstream 16eta) against an iso-LR LR-ONLY control (imp_pos_lronly_cut8_ckpt:
identity activation, the SAME upstream 16eta pattern). The two share the exact
learning-rate pattern; only the boundary activation multiplier differs.

Question: at (near-)identical accuracy, does the representation trajectory differ?
  - per-block paired Delta (boundary - LR-only) for drift D_l, effective rank r_eff,
    clean-label alignment A, with 95% CI over seeds 0-4.
  - the best-accuracy gap between the two conditions (iso-accuracy check).

Verdict: if late blocks show boundary consistently higher drift / lower rank /
higher alignment with CI excluding 0, boundary scaling exposes representation
dynamics that LR reweighting alone does not -> a distinct intervention. If the
signatures coincide, boundary scaling reduces to layer-wise LR in this regime.
"""
from __future__ import annotations

import csv
import glob
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

BOUNDARY = "imp_pos_cut8"
LRONLY = "imp_pos_lronly_cut8_ckpt"
CKPT = "checkpoint_best.pt"
SEEDS = [0, 1, 2, 3, 4]
DIAG_DIR = Path("outputs/depth_profiles")
T975 = {4: 2.776, 3: 3.182, 2: 4.303}  # df -> two-sided 95%


def latest_run_with_ckpt(experiment: str, seed: int) -> Path | None:
    cands = sorted(Path("outputs", experiment).glob(f"*/seed_{seed}/{CKPT}"))
    return cands[-1].parent if cands else None


def ensure_diagnostics(experiment: str) -> None:
    """Compute per-block diagnostics for any seed missing its CSV (same settings as the boundary side)."""
    for s in SEEDS:
        out = DIAG_DIR / f"{experiment}__seed{s}.csv"
        if out.exists():
            continue
        run_dir = latest_run_with_ckpt(experiment, s)
        if run_dir is None:
            print(f"[warn] {experiment} seed {s}: no checkpoint yet", flush=True)
            continue
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/compute_depth_profile.py",
               "--run-dir", str(run_dir), "--checkpoint", CKPT,
               "--split", "train", "--max-samples", "256", "--output", str(out)]
        print(f"[diag] {experiment} seed {s} -> {out}", flush=True)
        subprocess.run(cmd, check=True)


def load_blocks(experiment: str) -> dict[int, dict[str, list[float]]]:
    """block_index -> metric -> list over seeds."""
    by_block: dict[int, dict[str, list[float]]] = {}
    for s in SEEDS:
        p = DIAG_DIR / f"{experiment}__seed{s}.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            b = int(row["block_index"])
            d = by_block.setdefault(b, {"drift": [], "erank": [], "align": []})
            d["drift"].append(float(row["cka_drift"]))
            d["erank"].append(float(row["effective_rank"]))
            d["align"].append(float(row["clean_label_alignment"]))
    return by_block


def best_acc_by_seed(experiment: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in glob.glob(f"outputs/{experiment}/**/seed_*/metrics.csv", recursive=True):
        import re
        mm = re.search(r"/seed_(\d+)/", m)
        if not mm:
            continue
        r = list(csv.DictReader(open(m)))
        if not r or max(int(float(x["epoch"])) for x in r) < 79:
            continue
        out[int(mm.group(1))] = max(float(x["test_accuracy"]) for x in r) * 100.0
    return out


def paired(bvals: list[float], lvals: list[float]):
    """paired mean diff (b - l) + 95% CI; inputs aligned by seed order."""
    n = min(len(bvals), len(lvals))
    if n < 2:
        return (float("nan"),) * 3 + (n,)
    diffs = [bvals[i] - lvals[i] for i in range(n)]
    mu = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    t = T975.get(n - 1, 2.776)
    return mu, mu - t * se, mu + t * se, n


def main() -> None:
    ensure_diagnostics(LRONLY)

    bb = load_blocks(BOUNDARY)
    lb = load_blocks(LRONLY)

    # accuracy (iso-accuracy check)
    ba, la = best_acc_by_seed(BOUNDARY), best_acc_by_seed(LRONLY)
    common = sorted(set(ba) & set(la))
    if common:
        bacc = [ba[s] for s in common]
        lacc = [la[s] for s in common]
        mu, lo, hi, n = paired(bacc, lacc)
        print(f"ACCURACY  boundary {st.mean(bacc):.2f}  vs  LR-only {st.mean(lacc):.2f}  "
              f"| paired Δ {mu:+.2f} pp [{lo:+.2f},{hi:+.2f}] (n={n})")
    else:
        print("ACCURACY  (LR-only cut8 runs not complete yet)")

    print("\nPer-block representation signature: paired Δ (boundary − LR-only), 95% CI over seeds")
    print(f"{'blk':>3} {'depth':>5} | {'Δ drift D_l':>22} | {'Δ eff.rank':>22} | {'Δ align A':>22}")
    print("-" * 86)
    verdict_blocks = []
    for b in sorted(set(bb) & set(lb)):
        dd = paired(bb[b]["drift"], lb[b]["drift"])
        de = paired(bb[b]["erank"], lb[b]["erank"])
        da = paired(bb[b]["align"], lb[b]["align"])
        frac = b / (max(bb) if max(bb) else 1)

        def fmt(x):
            return f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]"
        print(f"{b:>3} {frac:>5.2f} | {fmt(dd):>22} | {fmt(de):>22} | {fmt(da):>22}")
        # late-block expected signature: higher drift, lower rank, higher align (CI excludes 0)
        if b >= 6:
            sep = (dd[1] > 0) + (de[2] < 0) + (da[1] > 0)
            verdict_blocks.append((b, sep))

    print("\nLate-block (>=6) separation count (of 3 metrics with CI excluding 0 in the expected direction):")
    for b, sep in verdict_blocks:
        print(f"  block {b}: {sep}/3")
    if verdict_blocks:
        deep = [s for b, s in verdict_blocks if b == max(bb)]
        print("\nVERDICT:", "representation SEPARATES (distinct intervention)"
              if deep and deep[0] >= 2 else
              "representation does NOT clearly separate at the deepest block — inspect CIs")


if __name__ == "__main__":
    main()
