#!/usr/bin/env python
"""Iso-accuracy representation control (decisive test for the representation pillar).

Compares the after-late BOUNDARY (iso_boundary_cut8: activation 0.0625, 16eta upstream)
against the LR-ONLY control (iso_lronly_cut8: identity activation, SAME 16eta upstream)
at checkpoints where BOTH reached the same test accuracy (55/57/59%, saved during
training the first epoch each target was crossed).

The end-of-training comparison is confounded: boundary ends ~1.9pp more accurate, and a
better-fit model naturally has lower-rank / higher-aligned late representations. Matching
on accuracy removes that confound. If the late-block signature (higher drift, lower rank,
higher align) still separates AT MATCHED ACCURACY -> representation is an axis independent
of accuracy: a genuine second pillar. If it collapses -> representation is downstream of
accuracy and should be reported as characterization only.

Run AFTER configs/jobs_iso_acc_cut8.txt completes.
"""
from __future__ import annotations

import csv
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

BOUNDARY = "iso_boundary_cut8"
LRONLY = "iso_lronly_cut8"
SEEDS = [0, 1, 2, 3, 4]
TARGETS = [55.0, 57.0, 59.0]
OUTDIR = Path("outputs/depth_profiles_isoacc")
T975 = {4: 2.776, 3: 3.182, 2: 4.303, 1: 12.706}


def ckpt_name(t: float) -> str:
    return f"checkpoint_atacc_{t:g}.pt"


def latest_run(experiment: str, seed: int, t: float) -> Path | None:
    cands = sorted(Path("outputs", experiment).glob(f"*/seed_{seed}/{ckpt_name(t)}"))
    return cands[-1].parent if cands else None


def acc_at_checkpoint(run_dir: Path, t: float) -> float | None:
    """Read the test accuracy stored when the matched-accuracy checkpoint was saved."""
    import torch
    try:
        ck = torch.load(run_dir / ckpt_name(t), map_location="cpu", weights_only=False)
    except Exception:
        return None
    m = ck.get("metrics") if isinstance(ck, dict) else None
    if m and "test_accuracy" in m:
        return float(m["test_accuracy"]) * 100.0
    return None


def ensure_diag(experiment: str, t: float) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        out = OUTDIR / f"{experiment}__seed{s}__acc{t:g}.csv"
        if out.exists():
            continue
        rd = latest_run(experiment, s, t)
        if rd is None:
            print(f"[warn] {experiment} seed {s} acc {t:g}: no checkpoint", flush=True)
            continue
        cmd = [sys.executable, "scripts/compute_depth_profile.py", "--run-dir", str(rd),
               "--checkpoint", ckpt_name(t), "--split", "train", "--max-samples", "256",
               "--output", str(out)]
        print(f"[diag] {experiment} seed {s} acc {t:g}", flush=True)
        subprocess.run(cmd, check=True)


def load_blocks(experiment: str, t: float) -> dict[int, dict[str, list[float]]]:
    by_block: dict[int, dict[str, list[float]]] = {}
    for s in SEEDS:
        p = OUTDIR / f"{experiment}__seed{s}__acc{t:g}.csv"
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            b = int(row["block_index"])
            d = by_block.setdefault(b, {"drift": [], "erank": [], "align": []})
            d["drift"].append(float(row["cka_drift"]))
            d["erank"].append(float(row["effective_rank"]))
            d["align"].append(float(row["clean_label_alignment"]))
    return by_block


def paired(b: list[float], l: list[float]):
    n = min(len(b), len(l))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    diffs = [b[i] - l[i] for i in range(n)]
    mu, sd = st.mean(diffs), st.stdev(diffs)
    t = T975.get(n - 1, 2.776)
    se = sd / math.sqrt(n)
    return mu, mu - t * se, mu + t * se, n


def main() -> None:
    for t in TARGETS:
        ensure_diag(BOUNDARY, t)
        ensure_diag(LRONLY, t)

    for t in TARGETS:
        print("=" * 90)
        # report the actual matched accuracies (confirm they are close)
        ba = [a for s in SEEDS if (rd := latest_run(BOUNDARY, s, t)) and (a := acc_at_checkpoint(rd, t)) is not None]
        la = [a for s in SEEDS if (rd := latest_run(LRONLY, s, t)) and (a := acc_at_checkpoint(rd, t)) is not None]
        bstr = f"{st.mean(ba):.2f}" if ba else "—"
        lstr = f"{st.mean(la):.2f}" if la else "—"
        gap = f"{st.mean(ba) - st.mean(la):+.2f}" if ba and la else "—"
        print(f"TARGET {t:g}%  matched acc: boundary {bstr}  LR-only {lstr}  (gap {gap} pp)")

        bb, lb = load_blocks(BOUNDARY, t), load_blocks(LRONLY, t)
        if not (bb and lb):
            print("  (diagnostics incomplete)")
            continue
        print(f"  {'blk':>3} | {'Δ drift':>20} | {'Δ eff.rank':>20} | {'Δ align':>20}")
        late = []
        for b in sorted(set(bb) & set(lb)):
            dd, de, da = (paired(bb[b][k], lb[b][k]) for k in ("drift", "erank", "align"))

            def f(x):
                return f"{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]"
            print(f"  {b:>3} | {f(dd):>20} | {f(de):>20} | {f(da):>20}")
            if b >= 6:
                late.append((b, (dd[1] > 0) + (de[2] < 0) + (da[1] > 0)))
        for b, sep in late:
            print(f"    late block {b}: {sep}/3 metrics separate (CI excludes 0, expected direction)")


if __name__ == "__main__":
    main()
