#!/usr/bin/env python
"""Build the fixed validation-controlled ViT-CIFAR job manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


SEEDS = tuple(range(5))
EXPECTED_JOBS = 25


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def common(config: str, seed: int) -> str:
    """Return overrides shared by every ViT-CIFAR run."""
    return (
        f"{config} training.seed={seed} training.max_epochs=80 "
        "model.name=vit_cifar model.width=192 "
        "data.val_fraction=0.1 data.val_seed=0 "
        "training.save_checkpoints=false training.checkpoint_at_accs=[] "
        "diagnostics.enabled=false"
    )


def build_jobs() -> list[str]:
    """Build five fixed conditions over seeds 0--4."""
    jobs: list[str] = []
    for seed in SEEDS:
        jobs.extend(
            [
                (
                    f"{common('configs/experiment_49_global_matched_seed_expansion.yaml', seed)} "
                    "fls.global.output_multiplier=1.0 experiment.name=val_vit25_baseline"
                ),
                (
                    f"{common('configs/experiment_49_global_matched_seed_expansion.yaml', seed)} "
                    "fls.global.output_multiplier=0.25 experiment.name=val_vit25_global_c0p25"
                ),
                (
                    f"{common('configs/experiment_49_global_matched_seed_expansion.yaml', seed)} "
                    "fls.global.output_multiplier=0.0625 experiment.name=val_vit25_global_c0p0625"
                ),
                (
                    f"{common('configs/experiment_40_boundary_late_seed_expansion.yaml', seed)} "
                    "fls.boundaries.after_late.output_multiplier=0.0625 "
                    "experiment.name=val_vit25_boundary_late_c0p0625"
                ),
                (
                    f"{common('configs/experiment_40_boundary_late_seed_expansion.yaml', seed)} "
                    "fls.boundaries.after_late.output_multiplier=1.0 "
                    "training.location_lr_compensation_multipliers.after_early=1.0 "
                    "training.location_lr_compensation_multipliers.after_middle=1.0 "
                    "training.location_lr_compensation_multipliers.after_late=0.0625 "
                    "experiment.name=val_vit25_lr_only_late_16x"
                ),
            ]
        )
    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(f"Expected {EXPECTED_JOBS} jobs, built {len(jobs)}")
    return jobs


def main() -> None:
    """Write the fixed manifest."""
    args = parse_args()
    jobs = build_jobs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Validation-controlled ViT-CIFAR architecture-transfer package.\n"
        "# Five conditions x seeds 0--4; all conditions fixed before evaluation.\n"
        "# Checkpoints are selected by a fixed 10% validation split only.\n"
        + "\n".join(jobs)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
