#!/usr/bin/env python
"""Generate the validation-controlled confound + factorial job file (seeds 0-4).

Group 1 (location compensation on, head at base eta): the 3x3 upstream-LR(m) x
activation(a) factorial plus two extra LR-only points (m=0.5, 0.125) for the
decoupled LR-only sweep (Fig. 10). Built on imp_dose_base via per-cell overrides.

Group 2 (compensation off, uniform LR): no-comp, full-comp, full-high-LR controls,
from dedicated clean configs.

All runs use a 10% validation split (val_seed 0); checkpoints are not saved
(selection reads val_accuracy from metrics.csv)."""

def san(v):
    return str(v).replace('.', 'p')

SEEDS = [0, 1, 2, 3, 4]
lines = []

# ---- Group 1: factorial 3x3 + k-sweep extras (location comp on, head = eta) ----
ms_fact = [1.0, 0.25, 0.0625]
as_fact = [1.0, 0.25, 0.0625]
cells = [(m, a) for m in ms_fact for a in as_fact]
cells += [(0.5, 1.0), (0.125, 1.0)]  # k=2, k=8 LR-only points for Fig. 10
for m, a in cells:
    name = f"val_cf_m{san(m)}_a{san(a)}"
    for s in SEEDS:
        lines.append(
            f"configs/imp_dose_base.yaml training.seed={s} training.max_epochs=80 "
            f"training.location_lr_compensation_multipliers.after_late={m} "
            f"fls.boundaries.after_late.output_multiplier={a} "
            f"data.val_fraction=0.1 data.val_seed=0 training.save_checkpoints=false "
            f"experiment.name={name}")
n1 = len(lines)

# ---- Group 2: uniform-LR controls (compensation off) ----
for cfg, name in [("val_cf_nocomp", "val_cf_nocomp"),
                  ("val_cf_fullcomp", "val_cf_fullcomp"),
                  ("val_cf_highlr", "val_cf_highlr")]:
    for s in SEEDS:
        lines.append(
            f"configs/{cfg}.yaml training.seed={s} training.max_epochs=80 "
            f"data.val_fraction=0.1 data.val_seed=0 training.save_checkpoints=false "
            f"experiment.name={name}")
n2 = len(lines) - n1

with open("configs/jobs_val_confound.txt", "w") as f:
    f.write('\n'.join(lines) + '\n')

# uniqueness is per (name, seed)
import re
pairs = [(re.search(r'experiment\.name=(\S+)', l).group(1),
          re.search(r'training\.seed=(\d+)', l).group(1)) for l in lines]
print(f"wrote configs/jobs_val_confound.txt: {n1} group-1 + {n2} group-2 = {len(lines)} jobs")
print(f"distinct cells: {len(set(p[0] for p in pairs))}  unique (name,seed): {len(set(pairs))}/{len(pairs)}")
