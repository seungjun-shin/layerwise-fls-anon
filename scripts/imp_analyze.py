#!/usr/bin/env python
"""Aggregate the experiment runs (dose-response, causal, VGG) into summary CSVs.

Reads metrics.csv (accuracy) and diagnostics.csv (NC metrics: effective rank,
CKA drift, clean-label alignment) from each run dir and writes:
  outputs/imp_analysis/accuracy_summary.csv
  outputs/imp_analysis/nc_summary.csv

No hand-entered numbers; everything is read from run outputs.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

OUT = Path("outputs")
ANALYSIS = OUT / "imp_analysis"
SEEDS = [0, 1, 2, 3, 4]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_final_acc(metrics: Path, min_epoch: int = 79) -> tuple[float, float] | None:
    rows = _rows(metrics)
    if not rows:
        return None
    accs = [float(r.get("test_accuracy", 0) or 0) for r in rows]
    epochs = [int(float(r.get("epoch", -1) or -1)) for r in rows]
    if max(epochs) < min_epoch:
        return None
    final = accs[epochs.index(max(epochs))]
    return max(accs) * 100.0, final * 100.0


def latest_seed_dir(name: str, seed: int) -> Path | None:
    cands = sorted(OUT.glob(f"{name}/**/seed_{seed}"))
    return cands[-1] if cands else None


def agg_accuracy(name: str, seeds=SEEDS) -> dict | None:
    best, final = [], []
    for s in seeds:
        d = latest_seed_dir(name, s)
        if d is None:
            continue
        bf = best_final_acc(d / "metrics.csv")
        if bf is None:
            continue
        best.append(bf[0])
        final.append(bf[1])
    if not best:
        return None
    return {
        "experiment": name,
        "n": len(best),
        "best_mean": round(st.mean(best), 3),
        "best_std": round(st.stdev(best), 3) if len(best) > 1 else 0.0,
        "final_mean": round(st.mean(final), 3),
        "final_std": round(st.stdev(final), 3) if len(final) > 1 else 0.0,
    }


def agg_nc(name: str, seeds=SEEDS) -> dict | None:
    keys = ["effective_rank", "cka_drift", "clean_label_alignment"]
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for s in seeds:
        d = latest_seed_dir(name, s)
        if d is None:
            continue
        rows = _rows(d / "diagnostics.csv")
        if not rows:
            continue
        row = rows[-1]
        for k in keys:
            if row.get(k) not in (None, ""):
                acc[k].append(float(row[k]))
    if not any(acc.values()):
        return None
    out = {"experiment": name, "n": max(len(v) for v in acc.values())}
    for k in keys:
        v = acc[k]
        out[f"{k}_mean"] = round(st.mean(v), 4) if v else ""
        out[f"{k}_std"] = round(st.stdev(v), 4) if len(v) > 1 else 0.0
    return out


def discover(prefix: str) -> list[str]:
    names = set()
    for p in OUT.glob(f"{prefix}*"):
        if p.is_dir():
            names.add(p.name)
    return sorted(names)


def write_csv(path: Path, rows: list[dict]) -> None:
    rows = [r for r in rows if r]
    if not rows:
        print(f"[analyze] no rows for {path.name}")
        return
    cols = list({k for r in rows for k in r})
    # stable column order: experiment, n, then the rest
    head = [c for c in ["experiment", "n"] if c in cols]
    head += [c for c in cols if c not in head]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[analyze] wrote {path} ({len(rows)} rows)")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    names = []
    for pre in ["imp_dose_after_late_", "imp_dose_after_early_", "imp_dose_after_middle_",
                "imp_causal_", "imp_vgg_"]:
        names += discover(pre)
    names = sorted(set(names))
    print(f"[analyze] {len(names)} experiments found")
    write_csv(ANALYSIS / "accuracy_summary.csv", [agg_accuracy(n) for n in names])
    write_csv(ANALYSIS / "nc_summary.csv", [agg_nc(n) for n in names])


if __name__ == "__main__":
    main()
