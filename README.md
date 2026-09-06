# Disentangling Activation Scaling From Layer-Wise Learning-Rate Reweighting at the Final Representation Boundary

Code for the paper "Disentangling Activation Scaling From Layer-Wise Learning-Rate
Reweighting at the Final Representation Boundary".

The library implements depth-resolved feature-learning-strength (FLS) interventions:
a global output multiplier with learning-rate compensation, representation-boundary
scaling at the after-early / after-middle / after-late boundaries, a continuous
depth-position scaling cut, and the diagnostics and robustness evaluations used in the paper.

## Setup

```bash
bash setup.sh
# or:
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
pytest
```

Python >= 3.10. A CUDA GPU is needed for full training; CPU is fine for the smoke test.

## Layout

```
src/fls/        models, scaling interventions, training (with upstream LR compensation), diagnostics
scripts/        training, sweeps, evaluation, plotting, aggregation
configs/        run configs and job lists
tests/          unit tests
outputs/        generated training artifacts (ignored by git)
results/        generated aggregate tables and plots (ignored by git)
```

## Running

Single run:

```bash
python scripts/train.py --config configs/imp_dose_base.yaml \
    training.seed=0 fls.boundaries.after_late.output_multiplier=0.0625
```

Batched runs over GPUs from a job list:

```bash
python scripts/gpu_pool_run.py --jobs configs/jobs_wave1_dose.txt --gpus 0,1
```

## Reproducing the paper

All reported checkpoint-selected accuracies use the test endpoint at the
earliest checkpoint attaining the maximum held-out validation accuracy. The
iso-accuracy analysis uses the first checkpoint crossing the stated held-out
validation-accuracy target. The test set is used only for reporting.

`scripts/reproduce_paper.sh` is the canonical entry point. It creates the job
manifests, resumes completed training runs, evaluates frozen checkpoints, and
runs the corresponding aggregation or plotting code. Run it from the repository
root:

```bash
scripts/reproduce_paper.sh list
scripts/reproduce_paper.sh --dry-run main-table-1
scripts/reproduce_paper.sh main-table-1
```

The default execution pool is GPUs `0,1,2,3`. Override the pool, Python
interpreter, or diagnostic device as needed:

```bash
GPU_LIST=0,1 PYTHON_BIN=./.venv/bin/python ANALYSIS_DEVICE=cuda:0 \
    scripts/reproduce_paper.sh supp-table-XIII
```

### Main manuscript: one command per result

| Result | Evidence | Command | Detailed supplement |
|---|---|---|---|
| Table 1 | Controlled and practical ACT minus exact LR-only comparisons | `scripts/reproduce_paper.sh main-table-1` | XIII--XV, XX--XXI |
| Table 2 | Validation-controlled after-late, exact same-cut, selected LR-only, and global comparisons | `scripts/reproduce_paper.sh main-table-2` | I--II, VI |
| Table 3 | Fresh-seed practical ResNet comparison | `scripts/reproduce_paper.sh main-table-3` | XIV |
| Table 4 | Practical final-interface diagnostics | `scripts/reproduce_paper.sh main-table-4` | XVI--XVII |
| Table 5 | Controlled late-stage and iso-accuracy representation diagnostics | `scripts/reproduce_paper.sh main-table-5` | III, XXV |
| Table 6 | Confound controls | `scripts/reproduce_paper.sh main-table-6` | IV--VI |

Figure 1 is the conceptual protocol diagram and has no experimental run target.

| Result | Evidence | Command |
|---|---|---|
| Figure 2 | Global FLS calibration | `scripts/reproduce_paper.sh main-figure-2` |
| Figure 3 | One-boundary location calibration | `scripts/reproduce_paper.sh main-figure-3` |
| Figure 4 | ResNet ACT and exact LR-only depth profiles | `scripts/reproduce_paper.sh main-figure-4` |
| Figure 5 | ResNet, VGG19-BN, and Tiny-ImageNet depth profiles | `scripts/reproduce_paper.sh main-figure-5` |
| Figure 6 | Validation-selected seed-level final-boundary comparison | `scripts/reproduce_paper.sh main-figure-6` |
| Figure 7 | Validation-target iso-accuracy representation comparison | `scripts/reproduce_paper.sh main-figure-7` |
| Figure 8 | Depth-wise ResNet representation profile | `scripts/reproduce_paper.sh main-figure-8` |
| Figure 9 | Relative update pressure | `scripts/reproduce_paper.sh main-figure-9` |
| Figure 10 | Decoupled upstream-LR-only sweep | `scripts/reproduce_paper.sh main-figure-10` |
| Figure 11 | LR-by-activation factorial | `scripts/reproduce_paper.sh main-figure-11` |
| Figure 12 | Validation-controlled scope profile | `scripts/reproduce_paper.sh main-figure-12` |

### Supplementary material: one command per table

The numbering follows the compiled supplementary PDF, which ends at Table XXV.

| Result | Evidence | Command |
|---|---|---|
| Table I | Seed-level exact same-cut comparison | `scripts/reproduce_paper.sh supp-table-I` |
| Table II | Selection--confirmation decomposition | `scripts/reproduce_paper.sh supp-table-II` |
| Table III | Late-stage representation diagnostics | `scripts/reproduce_paper.sh supp-table-III` |
| Table IV | Consolidated confound diagnostics | `scripts/reproduce_paper.sh supp-table-IV` |
| Table V | LR-by-activation factorial | `scripts/reproduce_paper.sh supp-table-V` |
| Table VI | Decoupled upstream-LR-only sweep | `scripts/reproduce_paper.sh supp-table-VI` |
| Table VII | CIFAR-100-C corruption and SVHN OOD evaluation | `scripts/reproduce_paper.sh supp-table-VII` |
| Table VIII | Validation-controlled VGG19-BN depth profile | `scripts/reproduce_paper.sh supp-table-VIII` |
| Table IX | Frozen seven-seed VGG tuned-comparator confirmation | `scripts/reproduce_paper.sh supp-table-IX` |
| Table X | Tiny-ImageNet depth profile | `scripts/reproduce_paper.sh supp-table-X` |
| Table XI | Tiny-ImageNet after-late dose profile | `scripts/reproduce_paper.sh supp-table-XI` |
| Table XII | Depth-wise ResNet representation profile | `scripts/reproduce_paper.sh supp-table-XII` |
| Table XIII | Ten-seed controlled exact ResNet/VGG comparisons | `scripts/reproduce_paper.sh supp-table-XIII` |
| Table XIV | Fresh-seed practical ResNet accuracy | `scripts/reproduce_paper.sh supp-table-XIV` |
| Table XV | Fresh-seed practical VGG19-BN accuracy | `scripts/reproduce_paper.sh supp-table-XV` |
| Table XVI | Practical interface diagnostics | `scripts/reproduce_paper.sh supp-table-XVI` |
| Table XVII | Classifier-head and pre-head normalization ablations | `scripts/reproduce_paper.sh supp-table-XVII` |
| Table XVIII | Exact same-cut VGG19-BN depth family | `scripts/reproduce_paper.sh supp-table-XVIII` |
| Table XIX | Validation-controlled scope family | `scripts/reproduce_paper.sh supp-table-XIX` |
| Table XX | Intervention-protocol ledger | `scripts/reproduce_paper.sh supp-table-XX` |
| Table XXI | Statistical-analysis ledger | `scripts/reproduce_paper.sh supp-table-XXI` |
| Table XXII | Budget and schedule sensitivity | `scripts/reproduce_paper.sh supp-table-XXII` |
| Table XXIII | Focused boundary-location grid | `scripts/reproduce_paper.sh supp-table-XXIII` |
| Table XXIV | VGG identity-activation upstream-LR comparator grid | `scripts/reproduce_paper.sh supp-table-XXIV` |
| Table XXV | Iso-accuracy ResNet/VGG late-block representation differences | `scripts/reproduce_paper.sh supp-table-XXV` |

To execute every unique experiment family and generate all experimental figures:

```bash
scripts/reproduce_paper.sh all
```

### Validation-controlled re-aggregation

Operating-point selection (global reference cell, after-late multiplier) and
per-run checkpoint selection use a held-out validation split: 10% of the
CIFAR-100 training set held out by a fixed `data.val_seed`, with the reported
test accuracy taken at the epoch that maximizes validation accuracy. The grid job
files give each cell a unique `experiment.name` so that `gpu_pool_run.py`'s
resume/skip check (keyed on `experiment.name`+seed) does not collapse distinct
cells onto one another.

| Artifact | Run | Aggregate / plot with |
|----------|-----|------------------------|
| Global-FLS heatmap + boundary calibration (val) | `python scripts/gen_regrid_jobs.py 0 configs/jobs_val_regrid_seed0.txt` and `... 1,2 configs/jobs_val_regrid_seed12.txt` (built from `configs/jobs_val_global_grid.txt`, `configs/jobs_val_boundary_calib.txt`), then `scripts/gpu_pool_run.py --jobs <file>` | `scripts/build_val_grid_summary.py` -> `paper/tables/global_fls_val_summary.csv`, `paper/tables/location_calib_val_summary.csv`; figures `scripts/plot_global_heatmap_val.py`, `scripts/plot_location_sensitivity_val.py` |
| Confound controls + LR x activation factorial (val) | `python scripts/gen_confound_val_jobs.py` -> `configs/jobs_val_confound.txt` (uses `configs/val_cf_{nocomp,fullcomp,highlr}.yaml`); `configs/jobs_val_cf_ext.txt` extends the LR-only sweep; run both via `scripts/gpu_pool_run.py` | `scripts/build_val_confound_summary.py` -> `paper/tables/confound_val_summary.csv`, `paper/tables/factorial_val_summary.csv`; figures `scripts/plot_c1_val.py`, `scripts/plot_factorial_val.py` |

The validation split is enabled by `data.val_fraction` (see `src/fls/data/datasets.py`
and `scripts/train.py`); `fls.lr_compensation_multiplier` /
`training.location_lr_compensation_multipliers` decouple the upstream learning-rate
pattern from the activation multiplier so that each learning-rate-only control
reproduces a boundary condition's LR schedule with identity activations.

## Output paths

Training runs write checkpoints and metrics below `outputs/` or
`paper_reproduction/raw_metrics/`. Aggregation and plotting commands write
tables and figures below `results/`, `paper/`, and the generated subdirectories
of `paper_reproduction/`.

## License

MIT (see `LICENSE`).
