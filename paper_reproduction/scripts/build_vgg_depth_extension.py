#!/usr/bin/env python
"""Generate the complete frozen VGG19-BN exact depth family."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIG_DIR = WORKSPACE / "configs" / "vgg_depth_extension"
MANIFEST = WORKSPACE / "manifests" / "vgg_depth_extension_jobs.json"
CUTS = [1, 3, 5, 7, 9, 11, 13, 15]
MULTIPLIER = 0.0625


def config(cut: int, role: str) -> dict[str, Any]:
    """Return one explicit no-test VGG position config."""
    output_multiplier = MULTIPLIER if role == "act" else 1.0
    fls: dict[str, Any] = {
        "mode": "position",
        "position": cut,
        "output_multiplier": output_multiplier,
    }
    if role == "lr":
        fls["lr_compensation_multiplier"] = MULTIPLIER
    return {
        "experiment": {"name": f"paper_vgg_depth_cut{cut}_{role}"},
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
        "model": {
            "name": "vgg19_bn_cifar",
            "in_channels": 3,
            "num_classes": 100,
            "width": 64,
            "use_bn": True,
        },
        "fls": fls,
        "training": {
            "seed": 3,
            "device": "auto",
            "max_epochs": 80,
            "max_steps": None,
            "optimizer": "sgd",
            "lr": 0.04096,
            "position_lr_compensation": True,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "save_checkpoints": True,
            "evaluate_test_each_epoch": False,
            "protocol_sanity": {"enabled": True},
        },
        "diagnostics": {"enabled": False},
    }


def main() -> None:
    """Write sixteen configs and a self-contained 80-job manifest."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for cut in CUTS:
        for role in ("act", "lr"):
            path = CONFIG_DIR / f"cut{cut}_{role}.yaml"
            path.write_text(yaml.safe_dump(config(cut, role), sort_keys=False), encoding="utf-8")
            for seed in range(5):
                jobs.append(
                    {
                        "id": f"vgg_depth__cut{cut}__{role}__seed{seed}",
                        "work_package": "WP3",
                        "block": "vgg_depth_extension",
                        "family": f"vgg_depth_cut{cut}",
                        "pair_role": role,
                        "condition": f"VD-CUT{cut}-{role.upper()}",
                        "architecture": "vgg19_bn_cifar",
                        "dataset": "cifar100",
                        "recipe": "controlled",
                        "expected_boundary_kind": "position",
                        "expected_boundary_index": cut,
                        "expected_boundary_ratio": MULTIPLIER if role == "act" else 1.0,
                        "seed": seed,
                        "config": str(path.relative_to(ROOT)),
                        "overrides": [f"training.seed={seed}"],
                        "status": "pending",
                        "attempts": 0,
                    }
                )
    assert len(jobs) == 80
    payload = {
        "project": "paper_vgg_depth_extension",
        "created_before_test_evaluation": True,
        "max_parallel": 4,
        "gpus": [0, 1, 2, 3],
        "oom_retry": {"max_attempts": 2, "delay_seconds": 30},
        "jobs": jobs,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {MANIFEST}")


if __name__ == "__main__":
    main()
