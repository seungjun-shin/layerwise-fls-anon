#!/usr/bin/env python
"""Regenerate the validation grid job files with UNIQUE experiment.name per cell.

The original jobs_val_global_grid.txt / jobs_val_boundary_calib.txt shared one
experiment.name across all cells, so gpu_pool_run.py's already_done() check (which
globs outputs/<name>/**/seed_<seed>/metrics.csv) skipped every cell after the first
of each seed completed. This rebuilds them with a per-cell name so resume and
per-job logs work correctly. seed 0 only (operating-point selection)."""
import re, sys

# seeds to emit (default seed 0); cell name is seed-independent so already_done()
# resumes per-seed via the seed_<seed> subdir.
SEEDS = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else {'0'}
OUT = sys.argv[2] if len(sys.argv) > 2 else 'configs/jobs_val_regrid_seed0.txt'

def field(line, key):
    m = re.search(rf'{re.escape(key)}=(\S+)', line)
    return m.group(1) if m else None

def san(v):
    return v.replace('.', 'p')

out = []
for line in open('configs/jobs_val_global_grid.txt'):
    s = line.strip()
    if not s or s.startswith('#') or field(s, 'training.seed') not in SEEDS:
        continue
    name = f"val_gg_c{san(field(s, 'fls.global.output_multiplier'))}_lr{san(field(s, 'training.lr'))}"
    out.append(re.sub(r'experiment\.name=\S+', f'experiment.name={name}', s))
ng = len(out)

for line in open('configs/jobs_val_boundary_calib.txt'):
    s = line.strip()
    if not s or s.startswith('#') or field(s, 'training.seed') not in SEEDS:
        continue
    name = f"val_bcal_p{field(s, 'fls.position')}_m{san(field(s, 'fls.output_multiplier'))}"
    out.append(re.sub(r'experiment\.name=\S+', f'experiment.name={name}', s))
nb = len(out) - ng

with open(OUT, 'w') as f:
    f.write('\n'.join(out) + '\n')

# uniqueness is per (name, seed) — names repeat across seeds by design
pairs = [(field(l, 'experiment.name'), field(l, 'training.seed')) for l in out]
print(f'wrote {OUT}: {ng} global + {nb} boundary = {len(out)} jobs (seeds {sorted(SEEDS)})')
print(f'unique (name,seed): {len(set(pairs))}/{len(pairs)}')
