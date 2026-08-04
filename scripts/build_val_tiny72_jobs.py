#!/usr/bin/env python
"""Build the 57-job development manifest for the Tiny-ImageNet validation study."""

from __future__ import annotations

import argparse
from pathlib import Path


CUTS = tuple(range(2, 9))
DEVELOPMENT_SEEDS = (0, 1, 2)
DOSE_MULTIPLIERS = (0.03125, 0.125, 0.25, 0.5)
EXPECTED_JOBS = 57


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def suffix(value: float) -> str:
    """Return a filesystem-safe decimal suffix."""
    return f"{value:g}".replace(".", "p")


def common(seed: int) -> str:
    """Return overrides shared by every development run."""
    return (
        "configs/imp_pos_base.yaml "
        f"training.seed={seed} training.max_epochs=80 "
        "data.name=tiny_imagenet data.num_classes=200 model.num_classes=200 "
        "data.val_fraction=0.1 data.val_seed=0 "
        "training.save_checkpoints=false training.checkpoint_at_accs=[] "
        "diagnostics.enabled=false"
    )


def build_jobs() -> list[str]:
    """Build the fixed 57-run development grid."""
    jobs: list[str] = []
    for seed in DEVELOPMENT_SEEDS:
        jobs.append(
            f"{common(seed)} fls.position=8 fls.output_multiplier=1.0 "
            "experiment.name=val_tiny72_baseline"
        )
        for cut in CUTS:
            jobs.append(
                f"{common(seed)} fls.position={cut} fls.output_multiplier=0.0625 "
                f"experiment.name=val_tiny72_bnd_cut{cut}"
            )
            jobs.append(
                f"{common(seed)} fls.position={cut} fls.output_multiplier=1.0 "
                "fls.lr_compensation_multiplier=0.0625 "
                f"experiment.name=val_tiny72_lr_cut{cut}_16x"
            )
        for multiplier in DOSE_MULTIPLIERS:
            jobs.append(
                f"{common(seed)} fls.position=8 fls.output_multiplier={multiplier:g} "
                f"experiment.name=val_tiny72_dose_cut8_c{suffix(multiplier)}"
            )
    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(f"Expected {EXPECTED_JOBS} jobs, built {len(jobs)}")
    return jobs


def main() -> None:
    """Write the development manifest."""
    args = parse_args()
    jobs = build_jobs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Tiny-ImageNet validation-controlled 72-run package: phase 1/2.\n"
        "# 3 baseline + 21 boundary + 21 exact LR-only + 12 extra dose = 57.\n"
        + "\n".join(jobs)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
