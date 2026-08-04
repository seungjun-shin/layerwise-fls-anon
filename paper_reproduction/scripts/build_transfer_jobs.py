#!/usr/bin/env python
"""Build frozen VGG19-BN practical-transfer manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIGS = WORKSPACE / "configs" / "practical_transfer"
CONDITIONS = {
    "T-REF": CONFIGS / "t_ref.yaml",
    "T-ACT-FINAL": CONFIGS / "t_act_final.yaml",
    "T-LR-FINAL": CONFIGS / "t_lr_final.yaml",
}


def parse_args() -> argparse.Namespace:
    """Parse output paths for the frozen manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(WORKSPACE / "manifests" / "practical_transfer_jobs.json"),
    )
    parser.add_argument(
        "--pilot-output",
        default=str(WORKSPACE / "manifests" / "practical_transfer_pilot_jobs.json"),
    )
    return parser.parse_args()


def job(condition: str, config: Path, seed: int, block: str, *, pilot: bool) -> dict:
    """Return one immutable queue record."""
    overrides = [f"training.seed={seed}"]
    if pilot:
        overrides.append("training.max_epochs=1")
    return {
        "id": f"{block}__{condition.lower().replace('-', '_')}__seed{seed}",
        "work_package": "WP3",
        "block": block,
        "family": "vgg_practical",
        "pair_role": {
            "T-REF": "ref",
            "T-ACT-FINAL": "act",
            "T-LR-FINAL": "lr",
        }[condition],
        "condition": condition,
        "architecture": "vgg19_bn_cifar",
        "dataset": "cifar100",
        "recipe": "practical_cosine",
        "expected_boundary_ratio": {
            "T-REF": 1.0,
            "T-ACT-FINAL": 0.0625,
            "T-LR-FINAL": 1.0,
        }[condition],
        "seed": seed,
        "config": str(config.relative_to(ROOT)),
        "overrides": overrides,
        "status": "pending",
        "attempts": 0,
    }


def payload(jobs: list[dict], project: str) -> dict:
    """Wrap jobs in the crash-safe queue schema."""
    return {
        "project": project,
        "created_before_test_evaluation": True,
        "expected_conditions": list(CONDITIONS),
        "expected_ratios": {"T-REF": 1.0, "T-ACT-FINAL": 0.0625, "T-LR-FINAL": 1.0},
        "max_parallel": 4,
        "gpus": [0, 1, 2, 3],
        "oom_retry": {"max_attempts": 2, "delay_seconds": 30},
        "jobs": jobs,
    }


def write(path: Path, contents: dict) -> None:
    """Write a deterministic JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    """Generate ten-seed full and one-seed pilot manifests."""
    args = parse_args()
    full_jobs = [
        job(condition, config, seed, "transfer", pilot=False)
        for seed in range(10, 20)
        for condition, config in CONDITIONS.items()
    ]
    pilot_jobs = [
        job(condition, config, 202, "pilot", pilot=True)
        for condition, config in CONDITIONS.items()
    ]
    assert len(full_jobs) == 30
    assert len(pilot_jobs) == 3
    write(Path(args.output), payload(full_jobs, "paper_vgg_practical_transfer"))
    write(Path(args.pilot_output), payload(pilot_jobs, "paper_vgg_practical_pilot"))


if __name__ == "__main__":
    main()
