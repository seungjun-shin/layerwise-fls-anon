#!/usr/bin/env python
"""Generate the complete frozen validation-only scope experiment family."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIG_DIR = WORKSPACE / "configs" / "scope_preservation"
MANIFEST = WORKSPACE / "manifests" / "scope_preservation_jobs.json"

FAMILIES: dict[str, dict[str, Any]] = {
    "stl10": {
        "dataset": "stl10",
        "num_classes": 10,
        "use_bn": True,
        "augment": False,
        "scheduler": None,
        "noise": 0.0,
        "global_c": 0.25,
        "global_lr": 0.04096,
        "act_c": 0.0625,
        "act_lr": 0.04096,
    },
    "svhn": {
        "dataset": "svhn",
        "num_classes": 10,
        "use_bn": True,
        "augment": False,
        "scheduler": None,
        "noise": 0.0,
        "global_c": 0.25,
        "global_lr": 0.04096,
        "act_c": 0.0625,
        "act_lr": 0.04096,
    },
    "noise20": {
        "dataset": "cifar100",
        "num_classes": 100,
        "use_bn": True,
        "augment": False,
        "scheduler": None,
        "noise": 0.2,
        "global_c": 0.25,
        "global_lr": 0.04096,
        "act_c": 0.0625,
        "act_lr": 0.04096,
    },
    "noise50": {
        "dataset": "cifar100",
        "num_classes": 100,
        "use_bn": True,
        "augment": False,
        "scheduler": None,
        "noise": 0.5,
        "global_c": 0.25,
        "global_lr": 0.04096,
        "act_c": 0.0625,
        "act_lr": 0.04096,
    },
    "cifar10": {
        "dataset": "cifar10",
        "num_classes": 10,
        "use_bn": True,
        "augment": False,
        "scheduler": None,
        "noise": 0.0,
        "global_c": 0.00390625,
        "global_lr": 0.01024,
        "act_c": 0.0078125,
        "act_lr": 0.04096,
    },
    "nobn": {
        "dataset": "cifar100",
        "num_classes": 100,
        "use_bn": False,
        "augment": False,
        "scheduler": None,
        "noise": 0.0,
        "global_c": 4.0,
        "global_lr": 0.32768,
        "act_c": 4.0,
        "act_lr": 0.32768,
    },
    "persistence": {
        "dataset": "cifar100",
        "num_classes": 100,
        "use_bn": True,
        "augment": True,
        "scheduler": "cosine",
        "noise": 0.0,
        "global_c": 0.25,
        "global_lr": 0.04096,
        "act_c": 0.0625,
        "act_lr": 0.04096,
    },
}


def base_config(family: str, spec: dict[str, Any], role: str) -> dict[str, Any]:
    """Construct one explicit config from recovered reviewed settings."""
    config: dict[str, Any] = {
        "experiment": {"name": f"paper_scope_{family}_{role}"},
        "output": {
            "root": "paper_reproduction/raw_metrics"
        },
        "data": {
            "name": spec["dataset"],
            "root": "data",
            "num_classes": spec["num_classes"],
            "image_size": 32,
            "batch_size": 128,
            "num_workers": 2,
            "augment": spec["augment"],
            "normalize": True,
            "return_clean_labels": True,
            "val_fraction": 0.1,
            "val_seed": 0,
            "deterministic_validation": True,
        },
        "label_noise": {"rate": spec["noise"], "seed": 0},
        "model": {
            "name": "resnet18_cifar",
            "in_channels": 3,
            "num_classes": spec["num_classes"],
            "width": 64,
            "use_bn": spec["use_bn"],
        },
        "training": {
            "seed": 0,
            "device": "auto",
            "max_epochs": 80,
            "max_steps": None,
            "optimizer": "sgd",
            "lr": spec["global_lr"] if role == "ref" else spec["act_lr"],
            "weight_decay": 0.0,
            "momentum": 0.0,
            "save_checkpoints": True,
            "evaluate_test_each_epoch": False,
            "protocol_sanity": {"enabled": True},
        },
        "diagnostics": {"enabled": False},
    }
    if spec["scheduler"] is not None:
        config["training"]["lr_scheduler"] = spec["scheduler"]
    if spec["noise"]:
        config["data"]["val_use_clean_labels"] = True
    if role == "ref":
        config["fls"] = {
            "mode": "global",
            "global": {"output_multiplier": spec["global_c"], "lr_compensation": True},
        }
    else:
        multiplier = spec["act_c"] if role == "act" else 1.0
        config["fls"] = {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_middle": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_late": {"output_multiplier": multiplier, "init_scale": 1.0},
            },
        }
        config["training"]["location_lr_compensation"] = "upstream"
        if role == "lr":
            config["training"]["location_lr_compensation_multipliers"] = {
                "after_early": 1.0,
                "after_middle": 1.0,
                "after_late": spec["act_c"],
            }
    return config


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    """Write a generated explicit YAML config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    """Generate all fixed configs and the self-contained 105-job manifest."""
    jobs: list[dict[str, Any]] = []
    for family, original_spec in FAMILIES.items():
        spec = deepcopy(original_spec)
        config_paths: dict[str, Path] = {}
        for role in ("ref", "act", "lr"):
            path = CONFIG_DIR / f"{family}_{role}.yaml"
            write_yaml(path, base_config(family, spec, role))
            config_paths[role] = path

        for role in ("ref", "act", "lr"):
            for seed in range(5):
                jobs.append(
                    {
                        "id": f"scope__{family}__{role}__seed{seed}",
                        "work_package": "WP1-WP5",
                        "block": "scope_preservation",
                        "family": family,
                        "pair_role": role,
                        "condition": f"S-{family.upper()}-{role.upper()}",
                        "architecture": "resnet18_cifar",
                        "dataset": spec["dataset"],
                        "recipe": "augmentation_cosine" if spec["augment"] else "controlled",
                        "expected_boundary_ratio": spec["act_c"] if role == "act" else 1.0,
                        "seed": seed,
                        "config": str(config_paths[role].relative_to(ROOT)),
                        "overrides": [f"training.seed={seed}"],
                        "status": "pending",
                        "attempts": 0,
                    }
                )

    assert len(jobs) == 105, len(jobs)
    assert len({job["id"] for job in jobs}) == len(jobs)
    payload = {
        "project": "paper_scope_preservation",
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
