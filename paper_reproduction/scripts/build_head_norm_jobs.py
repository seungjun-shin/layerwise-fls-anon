#!/usr/bin/env python
"""Build frozen classifier-head and pre-head-normalization ablation jobs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIG_DIR = WORKSPACE / "configs" / "head_norm"
FULL_MANIFEST = WORKSPACE / "manifests" / "head_norm_jobs.json"
PILOT_MANIFEST = WORKSPACE / "manifests" / "head_norm_pilot_jobs.json"
MULTIPLIER = 0.0625
PILOT_SEED = 220
VARIANTS = {
    "cosine": {"head_type": "cosine", "head_logit_scale": 1.0},
    "preln": {"pre_head_norm": "layer_norm_affine_free"},
}


def config(variant: str, role: str) -> dict[str, Any]:
    """Return one explicit validation-only training configuration."""
    boundaries = {
        "after_early": {"output_multiplier": 1.0, "init_scale": 1.0},
        "after_middle": {"output_multiplier": 1.0, "init_scale": 1.0},
        "after_late": {
            "output_multiplier": MULTIPLIER if role == "act" else 1.0,
            "init_scale": 1.0,
        },
    }
    training: dict[str, Any] = {
        "seed": 20,
        "device": "auto",
        "max_epochs": 80,
        "max_steps": None,
        "optimizer": "sgd",
        "lr": 0.04096,
        "location_lr_compensation": "upstream",
        "weight_decay": 0.0,
        "momentum": 0.0,
        "save_checkpoints": True,
        "evaluate_test_each_epoch": False,
        "protocol_sanity": {"enabled": True},
    }
    if role == "lr":
        training["location_lr_compensation_multipliers"] = {
            "after_early": 1.0,
            "after_middle": 1.0,
            "after_late": MULTIPLIER,
        }
    model = {
        "name": "resnet18_cifar",
        "in_channels": 3,
        "num_classes": 100,
        "width": 64,
        "use_bn": True,
        **VARIANTS[variant],
    }
    return {
        "experiment": {"name": f"paper_head_norm_{variant}_{role}"},
        "output": {
            "root": "paper_reproduction/raw_metrics"
        },
        "data": {
            "name": "cifar100",
            "root": "data",
            "num_classes": 100,
            "image_size": 32,
            "batch_size": 128,
            "num_workers": 2,
            "augment": False,
            "normalize": True,
            "return_clean_labels": True,
            "val_fraction": 0.1,
            "val_seed": 0,
            "deterministic_validation": True,
        },
        "label_noise": {"rate": 0.0, "seed": 0},
        "model": model,
        "fls": {"mode": "location", "boundaries": boundaries},
        "training": training,
        "diagnostics": {"enabled": False},
    }


def job(variant: str, role: str, seed: int, *, pilot: bool) -> dict[str, Any]:
    """Return one immutable queue record."""
    config_path = CONFIG_DIR / f"{variant}_{role}.yaml"
    block = "head_norm_pilot" if pilot else "head_norm"
    overrides = [f"training.seed={seed}"]
    if pilot:
        overrides.append("training.max_epochs=1")
    return {
        "id": f"{block}__{variant}__{role}__seed{seed}",
        "work_package": "WP4",
        "block": block,
        "family": f"head_norm_{variant}",
        "pair_role": role,
        "condition": f"HN-{variant.upper()}-{role.upper()}",
        "architecture": "resnet18_cifar",
        "dataset": "cifar100",
        "recipe": "controlled",
        "expected_boundary_ratio": MULTIPLIER if role == "act" else 1.0,
        "seed": seed,
        "config": str(config_path.relative_to(ROOT)),
        "overrides": overrides,
        "status": "pending",
        "attempts": 0,
    }


def payload(jobs: list[dict[str, Any]], project: str) -> dict[str, Any]:
    """Wrap jobs in the crash-safe queue schema."""
    return {
        "project": project,
        "created_before_test_evaluation": True,
        "max_parallel": 4,
        "gpus": [0, 1, 2, 3],
        "oom_retry": {"max_attempts": 2, "delay_seconds": 30},
        "jobs": jobs,
    }


def write_json(path: Path, contents: dict[str, Any]) -> None:
    """Write a deterministic JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Write four configs, four pilot jobs, and twenty full jobs."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for role in ("act", "lr"):
            path = CONFIG_DIR / f"{variant}_{role}.yaml"
            path.write_text(yaml.safe_dump(config(variant, role), sort_keys=False), encoding="utf-8")
    full_jobs = [
        job(variant, role, seed, pilot=False)
        for seed in range(20, 25)
        for variant in VARIANTS
        for role in ("act", "lr")
    ]
    pilot_jobs = [
        job(variant, role, PILOT_SEED, pilot=True)
        for variant in VARIANTS
        for role in ("act", "lr")
    ]
    assert len(full_jobs) == 20
    assert len(pilot_jobs) == 4
    write_json(FULL_MANIFEST, payload(full_jobs, "paper_head_norm"))
    write_json(PILOT_MANIFEST, payload(pilot_jobs, "paper_head_norm_pilot"))
    print(f"Wrote {len(full_jobs)} full jobs and {len(pilot_jobs)} pilot jobs.")


if __name__ == "__main__":
    main()
