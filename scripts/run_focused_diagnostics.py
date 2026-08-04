#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagnosticCell:
    name: str
    config: str
    overrides: tuple[str, ...]


CELLS = {
    "global": DiagnosticCell(
        name="global",
        config="configs/paper_reproduction_cifar100_fullgrid.yaml",
        overrides=(
            "experiment.name=diagnostic_global_best_seed0",
            "training.seed=0",
            "training.lr=0.04096",
            "training.save_checkpoints=true",
            "fls.global.output_multiplier=0.25",
        ),
    ),
    "early": DiagnosticCell(
        name="early",
        config="configs/experiment_18_independent_block_sensitivity_lrcomp.yaml",
        overrides=(
            "experiment.name=diagnostic_early_lrcomp_best_seed0",
            "training.seed=0",
            "training.lr=0.16384",
            "training.save_checkpoints=true",
            "fls.profile_name=null",
            "fls.groups.early.output_multiplier=2.0",
            "fls.groups.middle.output_multiplier=1.0",
            "fls.groups.late.output_multiplier=1.0",
            "fls.groups.head.output_multiplier=1.0",
        ),
    ),
    "middle": DiagnosticCell(
        name="middle",
        config="configs/experiment_18_independent_block_sensitivity_lrcomp.yaml",
        overrides=(
            "experiment.name=diagnostic_middle_lrcomp_best_seed0",
            "training.seed=0",
            "training.lr=0.08192",
            "training.save_checkpoints=true",
            "fls.profile_name=null",
            "fls.groups.early.output_multiplier=1.0",
            "fls.groups.middle.output_multiplier=0.25",
            "fls.groups.late.output_multiplier=1.0",
            "fls.groups.head.output_multiplier=1.0",
        ),
    ),
    "head": DiagnosticCell(
        name="head",
        config="configs/experiment_18_independent_block_sensitivity_lrcomp.yaml",
        overrides=(
            "experiment.name=diagnostic_head_lrcomp_best_seed0",
            "training.seed=0",
            "training.lr=0.08192",
            "training.save_checkpoints=true",
            "fls.profile_name=null",
            "fls.groups.early.output_multiplier=1.0",
            "fls.groups.middle.output_multiplier=1.0",
            "fls.groups.late.output_multiplier=1.0",
            "fls.groups.head.output_multiplier=0.25",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused diagnostic reruns for selected paper cells.")
    parser.add_argument("--cells", nargs="+", choices=sorted(CELLS), default=sorted(CELLS))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def run_cell(cell: DiagnosticCell, args: argparse.Namespace) -> Path:
    train_command = [sys.executable, "scripts/train.py", "--config", cell.config, *cell.overrides]
    result = run_command(train_command)
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    diagnostic_command = [
        sys.executable,
        "scripts/compute_diagnostics.py",
        "--run-dir",
        str(run_dir),
        "--split",
        args.split,
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    run_command(diagnostic_command)
    return run_dir


def main() -> None:
    args = parse_args()
    for cell_name in args.cells:
        run_dir = run_cell(CELLS[cell_name], args)
        print(f"{cell_name}: {run_dir}")


if __name__ == "__main__":
    main()
