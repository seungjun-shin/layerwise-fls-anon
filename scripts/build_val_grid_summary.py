#!/usr/bin/env python
"""Build validation-based grid summaries from the regrid runs, averaged over seeds.

For every cell and every available seed, select the epoch maximizing held-out
validation accuracy and record the TEST accuracy/loss at that epoch (the
validation-controlled numbers used throughout Table 1). Per cell, report the MEAN
over seeds (and the std of test accuracy). Cell coordinates are parsed from the
per-cell experiment.name (the logged learning_rate column is the EFFECTIVE rate
base/c, so the base rate is taken from the name).

Writes:
  paper/tables/global_fls_val_summary.csv     (c, base_lr, n, val_acc, test_at_val, ...)
  paper/tables/location_calib_val_summary.csv (position, multiplier, n, val_acc, ...)
"""
from __future__ import annotations
import csv, glob, re, statistics as st
from pathlib import Path

Path("paper/tables").mkdir(parents=True, exist_ok=True)


def unsan(s: str) -> float:
    return float(s.replace('p', '.'))


def argmax_val(metrics_path: str):
    rows = [r for r in csv.DictReader(open(metrics_path))
            if r.get("val_accuracy") not in (None, "", "None")]
    if not rows or max(int(float(r["epoch"])) for r in rows) < 79:
        return None  # missing val column or incomplete run
    best = max(rows, key=lambda r: float(r["val_accuracy"]))
    return (float(best["val_accuracy"]), float(best["test_accuracy"]),
            float(best["test_loss"]), int(float(best["epoch"])))


def aggregate(cell_dir: str):
    """Mean over all available seeds of (val, test, test_loss) at the val-selected epoch."""
    per_seed = {}
    for m in glob.glob(f"{cell_dir}/**/seed_*/metrics.csv", recursive=True):
        sm = re.search(r"/seed_(\d+)/", m)
        if not sm:
            continue
        res = argmax_val(m)
        if res is not None:
            per_seed[int(sm.group(1))] = res  # last complete run per seed wins
    if not per_seed:
        return None
    vals = [v[0] for v in per_seed.values()]
    tests = [v[1] for v in per_seed.values()]
    losses = [v[2] for v in per_seed.values()]
    n = len(per_seed)
    return (n, st.mean(vals), st.mean(tests), st.mean(losses),
            st.stdev(tests) if n > 1 else 0.0, sorted(per_seed))


# ---- global grid ----
gg = []
for d in sorted(glob.glob("outputs/val_gg_c*")):
    m = re.match(r"val_gg_c([0-9p]+)_lr([0-9p]+)$", Path(d).name)
    if not m:
        continue
    agg = aggregate(d)
    if agg is None:
        continue
    n, val, test, tloss, tstd, seeds = agg
    gg.append((Path(d).name, unsan(m.group(1)), unsan(m.group(2)), n, val, test, tloss, tstd))

with open("paper/tables/global_fls_val_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name", "global_output_multiplier", "learning_rate", "n",
                "val_accuracy", "test_at_best_val", "test_loss_at_best_val", "std_test_at_best_val"])
    for r in sorted(gg, key=lambda x: (x[2], x[1])):
        w.writerow(r)
print(f"global grid: {len(gg)} cells (seeds/cell: {sorted(set(x[3] for x in gg))}) "
      f"-> paper/tables/global_fls_val_summary.csv")
if gg:
    bv = max(gg, key=lambda x: x[4])
    print(f"  argmax-VAL cell: c={bv[1]} base_lr={bv[2]} (eff={bv[2]/bv[1]:.5f})  n={bv[3]}  "
          f"val={bv[4]*100:.2f} test={bv[5]*100:.2f}+-{bv[7]*100:.2f}")

# ---- boundary calibration ----
bc = []
for d in sorted(glob.glob("outputs/val_bcal_p*")):
    m = re.match(r"val_bcal_p(\d+)_m([0-9p]+)$", Path(d).name)
    if not m:
        continue
    agg = aggregate(d)
    if agg is None:
        continue
    n, val, test, tloss, tstd, seeds = agg
    bc.append((Path(d).name, int(m.group(1)), unsan(m.group(2)), n, val, test, tloss, tstd))

with open("paper/tables/location_calib_val_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment_name", "position", "output_multiplier", "n",
                "val_accuracy", "test_at_best_val", "test_loss_at_best_val", "std_test_at_best_val"])
    for r in sorted(bc, key=lambda x: (x[1], x[2])):
        w.writerow(r)
print(f"boundary calib: {len(bc)} cells (seeds/cell: {sorted(set(x[3] for x in bc))}) "
      f"-> paper/tables/location_calib_val_summary.csv")
for pos in sorted(set(r[1] for r in bc)):
    sub = [r for r in bc if r[1] == pos]
    bv = max(sub, key=lambda x: x[4])
    print(f"  pos {pos}: argmax-VAL mult={bv[2]}  n={bv[3]}  val={bv[4]*100:.2f} "
          f"test={bv[5]*100:.2f}+-{bv[7]*100:.2f}")
