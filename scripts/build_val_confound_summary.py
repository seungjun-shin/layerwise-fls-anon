#!/usr/bin/env python
"""Aggregate the validation-controlled confound + factorial runs (val_cf_*) over seeds.

For each cell, per seed select the epoch maximizing validation accuracy and take the
test accuracy/loss there; report the mean over seeds (and std of test accuracy).

Writes:
  paper/tables/factorial_val_summary.csv   (m, a, n, val, test, test_loss, std)
  paper/tables/confound_val_summary.csv    (condition, n, val, test_best, test_final?, std)
"""
from __future__ import annotations
import csv, glob, re, statistics as st
from pathlib import Path

Path("paper/tables").mkdir(parents=True, exist_ok=True)


def unsan(s):
    return float(s.replace('p', '.'))


def argmax_val(metrics_path):
    rows = [r for r in csv.DictReader(open(metrics_path))
            if r.get("val_accuracy") not in (None, "", "None")]
    if not rows or max(int(float(r["epoch"])) for r in rows) < 79:
        return None
    best = max(rows, key=lambda r: float(r["val_accuracy"]))
    final = max(rows, key=lambda r: int(float(r["epoch"])))
    return (float(best["val_accuracy"]), float(best["test_accuracy"]),
            float(best["test_loss"]), float(final["test_accuracy"]))


def aggregate(cell_dir):
    per_seed = {}
    for m in glob.glob(f"{cell_dir}/**/seed_*/metrics.csv", recursive=True):
        sm = re.search(r"/seed_(\d+)/", m)
        res = argmax_val(m)
        if sm and res is not None:
            per_seed[int(sm.group(1))] = res
    if not per_seed:
        return None
    v = list(per_seed.values())
    n = len(v)
    tests = [x[1] for x in v]
    return dict(n=n, val=st.mean(x[0] for x in v), test=st.mean(tests),
                tloss=st.mean(x[2] for x in v), tfinal=st.mean(x[3] for x in v),
                std=st.stdev(tests) if n > 1 else 0.0, seeds=sorted(per_seed))


# ---- factorial 3x3 ----
fact = []
for d in sorted(glob.glob("outputs/val_cf_m*_a*")):
    mm = re.match(r"val_cf_m([0-9p]+)_a([0-9p]+)$", Path(d).name)
    if not mm:
        continue
    a = aggregate(d)
    if a:
        fact.append((unsan(mm.group(1)), unsan(mm.group(2)), a))

with open("paper/tables/factorial_val_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m", "a", "n", "val_accuracy", "test_at_best_val", "test_loss", "std_test"])
    for m, a, ag in sorted(fact, key=lambda x: (-x[0], -x[1])):
        w.writerow([m, a, ag["n"], f"{ag['val']:.4f}", f"{ag['test']:.4f}",
                    f"{ag['tloss']:.4f}", f"{ag['std']:.4f}"])
print(f"factorial: {len(fact)} cells -> paper/tables/factorial_val_summary.csv")
for m, a, ag in sorted(fact, key=lambda x: (-x[0], -x[1])):
    print(f"  m={m} a={a}: n={ag['n']} val={ag['val']*100:.2f} test={ag['test']*100:.2f}+-{ag['std']*100:.2f}")

# ---- confound table rows ----
# map condition label -> (cell dir, body LR desc, head LR desc, act)
ROWS = [
    ("Boundary no-comp",   "val_cf_nocomp",        "eta",   "eta",   "0.0625"),
    ("Full high-LR",       "val_cf_highlr",        "16eta", "16eta", "1.0"),
    ("Full compensation",  "val_cf_fullcomp",      "16eta", "16eta", "0.0625"),
    ("LR-only (fixed 16x)","val_cf_m0p0625_a1p0",  "16eta", "eta",   "1.0"),
    ("LR-only (tuned 8x)", "val_cf_m0p125_a1p0",   "8eta",  "eta",   "1.0"),
    ("After-late boundary","val_cf_m0p0625_a0p0625","16eta","eta",   "0.0625"),
]
conf = []
for label, name, body, head, act in ROWS:
    a = aggregate(f"outputs/{name}")
    if a:
        conf.append((label, body, head, act, a))

with open("paper/tables/confound_val_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "body_lr", "head_lr", "activation", "n",
                "val_accuracy", "test_at_best_val", "test_final", "std_test"])
    for label, body, head, act, ag in conf:
        w.writerow([label, body, head, act, ag["n"], f"{ag['val']:.4f}",
                    f"{ag['test']:.4f}", f"{ag['tfinal']:.4f}", f"{ag['std']:.4f}"])
print(f"\nconfound: {len(conf)} rows -> paper/tables/confound_val_summary.csv")
for label, body, head, act, ag in conf:
    print(f"  {label:22s}: n={ag['n']} test@val={ag['test']*100:.2f}+-{ag['std']*100:.2f} "
          f"(final {ag['tfinal']*100:.2f})")
