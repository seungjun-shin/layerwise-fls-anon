#!/usr/bin/env python
"""Generate paper figures from raw outputs.

Currently produces the C1 decoupling figure: the tuned upstream-LR-only sweep
(identity activations, representation regions at k*eta) against the matched global
reference and the late-boundary intervention. Saves to paper/figures/.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "savefig.bbox": "tight", "savefig.dpi": 200})

OUT = Path("outputs")
FIG = Path("paper/figures")
SEEDS = [0, 1, 2, 3, 4]


def best_acc(metrics: Path) -> float | None:
    try:
        with metrics.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    if not rows:
        return None
    accs = [float(r.get("test_accuracy", 0) or 0) for r in rows]
    if max(int(float(r.get("epoch", -1) or -1)) for r in rows) < 79:
        return None
    return max(accs) * 100.0


def agg(name: str) -> tuple[float, float]:
    vals = []
    for s in SEEDS:
        cands = sorted(OUT.glob(f"{name}/**/seed_{s}/metrics.csv"))
        for mp in reversed(cands):
            v = best_acc(mp)
            if v is not None:
                vals.append(v)
                break
    if not vals:
        return (float("nan"), 0.0)
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ks = [4, 8, 16, 32, 64]
    lr_only = [agg(f"experiment_88_lr_only_decoupled_k{k}") for k in ks]
    g_mean, g_std = agg("experiment_49_global_matched_seed_expansion")
    l_mean, l_std = agg("experiment_40_boundary_late_seed_expansion_after_late")

    xs = list(range(len(ks)))
    means = [m for m, _ in lr_only]
    stds = [s for _, s in lr_only]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # Late-boundary and global reference bands.
    ax.axhspan(l_mean - l_std, l_mean + l_std, color="tab:orange", alpha=0.12)
    ax.axhline(l_mean, color="tab:orange", lw=2, label=f"After-late boundary ({l_mean:.2f})")
    ax.axhspan(g_mean - g_std, g_mean + g_std, color="tab:blue", alpha=0.12)
    ax.axhline(g_mean, color="tab:blue", lw=2, ls="--", label=f"Matched global FLS ({g_mean:.2f})")
    # Tuned LR-only sweep.
    ax.errorbar(xs, means, yerr=stds, marker="o", color="black", capsize=3,
                lw=2, label="LR-only (identity acts.)")
    best_i = max(range(len(means)), key=lambda i: means[i])
    ax.annotate(f"tuned optimum\n$k$={ks[best_i]} ({means[best_i]:.2f})",
                xy=(xs[best_i], means[best_i]), xytext=(xs[best_i], means[best_i] - 1.6),
                ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$\\eta\\times{k}$" for k in ks])
    ax.set_xlabel("Upstream learning-rate multiplier (identity activations)")
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title("Decoupled upstream-LR-only control")
    ax.legend(loc="lower center", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    path = FIG / "c1_decoupled_lr_only.pdf"
    fig.savefig(path)
    print(f"wrote {path}  (global={g_mean:.2f}, late={l_mean:.2f}, lr-only tuned={means[best_i]:.2f} at k={ks[best_i]})")


if __name__ == "__main__":
    main()
