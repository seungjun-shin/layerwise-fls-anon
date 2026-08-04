#!/usr/bin/env python
"""LR-envelope figure/table: best-achievable accuracy per boundary over base_lr x c,
answering "is the late advantage just a favorable base LR?".

For each boundary we marginalize over base LR (take the best base LR at each c) and
plot the resulting LR-robust dose-response, with the global FLS reference as a line.
Only the late boundary clears the global scalar; early/middle cannot, even at their
own best LR -> the advantage is not a base-LR artifact.

Reads ground truth from each run's resolved_config.yaml (dir names are unreliable).
Sources: outputs/experiment_39_* (boundaries) + outputs/experiment_42_* (global).
Outputs: paper/figures/lr_envelope.pdf, paper/tables/lr_envelope.csv
"""
from __future__ import annotations

import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 11, "savefig.bbox": "tight", "savefig.dpi": 200})

BOUNDARIES = ["after_early", "after_middle", "after_late"]
COLOR = {"after_early": "#4F9D4A", "after_middle": "#3C78D8", "after_late": "#7E57C2"}
LABEL = {"after_early": "after-early", "after_middle": "after-middle", "after_late": "after-late"}


def best_acc(metrics: str):
    try:
        r = list(csv.DictReader(open(metrics)))
    except OSError:
        return None
    if not r or max(int(float(x["epoch"])) for x in r) < 79:
        return None
    return max(float(x["test_accuracy"]) for x in r) * 100.0


def scan_location(boundary: str):
    cells = defaultdict(list)  # (lr, c) -> [acc]
    for cfgp in glob.glob(f"outputs/experiment_39_boundary_sensitivity_lrcomp_{boundary}*/**/seed_*/resolved_config.yaml", recursive=True):
        cfg = yaml.safe_load(open(cfgp))
        if cfg.get("fls", {}).get("mode") != "location":
            continue
        b = cfg["fls"].get("boundaries", {})
        c = float((b.get(boundary) or {}).get("output_multiplier", 1.0))
        others = [o for o in BOUNDARIES if o != boundary]
        if any(float((b.get(o) or {}).get("output_multiplier", 1.0)) != 1.0 for o in others):
            continue
        lr = float(cfg["training"]["lr"])
        a = best_acc(cfgp.replace("resolved_config.yaml", "metrics.csv"))
        if a is not None:
            cells[(lr, c)].append(a)
    return cells


def scan_global():
    cells = defaultdict(list)
    for cfgp in glob.glob("outputs/experiment_42_global_reference_rerun/**/seed_*/resolved_config.yaml", recursive=True):
        cfg = yaml.safe_load(open(cfgp))
        g = cfg.get("fls", {}).get("global", {})
        c = float(g.get("output_multiplier", 1.0))
        lr = float(cfg["training"]["lr"])
        a = best_acc(cfgp.replace("resolved_config.yaml", "metrics.csv"))
        if a is not None:
            cells[(lr, c)].append(a)
    return cells


def envelope_over_c(cells):
    """For each c, best mean-acc over base LR. Return sorted [(c, acc, lr)]."""
    by_c = defaultdict(list)
    for (lr, c), accs in cells.items():
        by_c[c].append((st.mean(accs), lr))
    out = []
    for c, lst in by_c.items():
        acc, lr = max(lst)
        out.append((c, round(acc, 2), lr))
    return sorted(out)


def main():
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)

    g = scan_global()
    g_env = max((st.mean(v) for v in g.values()), default=float("nan"))

    rows = []
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for b in BOUNDARIES:
        cells = scan_location(b)
        if not cells:
            continue
        env = envelope_over_c(cells)
        cs = [c for c, _, _ in env]
        accs = [a for _, a, _ in env]
        ax.plot(cs, accs, "-o", color=COLOR[b], label=LABEL[b], ms=4, lw=1.8)
        # overall envelope point
        bc, ba, blr = max(env, key=lambda t: t[1])
        ax.scatter([bc], [ba], s=110, facecolor="none", edgecolor=COLOR[b], lw=1.8, zorder=5)
        rows.append({"boundary": b, "envelope_acc": ba, "best_c": bc, "best_base_lr": blr,
                     "beats_global": ba > g_env})

    ax.axhline(g_env, color="#C0392B", ls="--", lw=1.4)
    ax.text(ax.get_xlim()[1], g_env, f"  global FLS ref = {g_env:.2f}", color="#C0392B",
            va="center", ha="left", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"boundary multiplier $c$ (log scale)")
    ax.set_ylabel("best test accuracy (%), best base LR per $c$")
    ax.set_title("LR-robust dose-response: only the late boundary clears the global scalar")
    ax.legend(loc="lower center", fontsize=9, frameon=False)
    ax.grid(True, color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    out = "paper/figures/lr_envelope.pdf"
    fig.savefig(out)
    print(f"wrote {out}")

    rows.append({"boundary": "global_ref", "envelope_acc": round(g_env, 2),
                 "best_c": "", "best_base_lr": "", "beats_global": ""})
    with open("paper/tables/lr_envelope.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["boundary", "envelope_acc", "best_c", "best_base_lr", "beats_global"])
        w.writeheader()
        w.writerows(rows)
    print("wrote paper/tables/lr_envelope.csv")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
