import csv
from pathlib import Path

import yaml

from fls.scaling.profiles import build_profile
from scripts.sweep import (
    anchored_profile_groups,
    boundary_overrides,
    bn_mode_overrides,
    comparable_config,
    completed_run_exists,
    group_overrides,
    interaction_values_for_boundary_pair,
    interaction_values_for_pair,
)


def test_comparable_config_normalizes_profile_groups() -> None:
    groups = ["early", "middle", "late", "head"]
    profile_config = {
        "experiment": {"name": "exp"},
        "data": {"name": "fake", "num_classes": 10, "augment": False, "normalize": False},
        "model": {"name": "small_cnn", "width": 8, "use_bn": True},
        "fls": {"mode": "blockwise", "profile_name": "progressive_increasing", "groups": None},
        "label_noise": {"rate": 0.0, "seed": 0},
        "training": {"seed": 1, "max_epochs": 2, "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.0},
    }
    resolved_config = {
        **profile_config,
        "fls": {
            "mode": "blockwise",
            "profile_name": "progressive_increasing",
            "groups": build_profile("progressive_increasing", groups, seed=1),
        },
    }
    assert comparable_config(profile_config) == comparable_config(resolved_config)


def test_completed_run_exists_ignores_checkpoint_policy(tmp_path: Path) -> None:
    config = {
        "experiment": {"name": "exp"},
        "output": {"root": str(tmp_path / "outputs")},
        "data": {
            "name": "fake",
            "root": "data",
            "num_classes": 10,
            "image_size": 32,
            "train_size": 8,
            "test_size": 4,
            "batch_size": 4,
            "augment": False,
            "normalize": False,
        },
        "model": {"name": "small_cnn", "in_channels": 3, "num_classes": 10, "width": 8, "use_bn": True},
        "fls": {"mode": "blockwise", "profile_name": "progressive_increasing", "groups": None},
        "label_noise": {"rate": 0.0, "seed": 0},
        "training": {
            "seed": 0,
            "device": "cpu",
            "max_epochs": 2,
            "optimizer": "sgd",
            "lr": 0.01,
            "weight_decay": 0.0,
            "momentum": 0.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_dir = tmp_path / "outputs" / "exp" / "20260101_000000" / "seed_0"
    run_dir.mkdir(parents=True)
    resolved = {
        **config,
        "fls": {
            "mode": "blockwise",
            "profile_name": "progressive_increasing",
            "groups": build_profile("progressive_increasing", ["early", "middle", "late", "head"], seed=0),
        },
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "test_accuracy"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "test_accuracy": 0.1})
        writer.writerow({"epoch": 1, "test_accuracy": 0.2})

    assert completed_run_exists(str(config_path), ["training.save_checkpoints=false"])


def test_comparable_config_distinguishes_location_lr_compensation() -> None:
    base = {
        "experiment": {"name": "exp"},
        "data": {"name": "fake", "num_classes": 10, "augment": False, "normalize": False},
        "model": {"name": "small_cnn", "width": 8, "use_bn": True},
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_middle": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_late": {"output_multiplier": 0.5, "init_scale": 1.0},
            },
        },
        "label_noise": {"rate": 0.0, "seed": 0},
        "training": {"seed": 0, "max_epochs": 2, "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.0},
    }
    compensated = {**base, "training": {**base["training"], "location_lr_compensation": "upstream"}}
    uncompensated = {**base, "training": {**base["training"], "location_lr_compensation": None}}
    assert comparable_config(compensated) != comparable_config(uncompensated)


def test_comparable_config_distinguishes_location_lr_only_reference() -> None:
    base = {
        "experiment": {"name": "exp"},
        "data": {"name": "fake", "num_classes": 10, "augment": False, "normalize": False},
        "model": {"name": "small_cnn", "width": 8, "use_bn": True},
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_middle": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_late": {"output_multiplier": 1.0, "init_scale": 1.0},
            },
        },
        "label_noise": {"rate": 0.0, "seed": 0},
        "training": {
            "seed": 0,
            "max_epochs": 2,
            "optimizer": "sgd",
            "lr": 0.01,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "location_lr_compensation": "upstream",
        },
    }
    lr_only = {
        **base,
        "training": {
            **base["training"],
            "location_lr_compensation_multipliers": {
                "after_early": 1.0,
                "after_middle": 1.0,
                "after_late": 0.0625,
            },
        },
    }
    assert comparable_config(base) != comparable_config(lr_only)


def test_anchor_profile_groups_scale_all_groups() -> None:
    assert anchored_profile_groups(0.25, [2.0, 1.0, 0.5, 1.0]) == {
        "early": 0.5,
        "middle": 0.25,
        "late": 0.125,
        "head": 0.25,
    }


def test_group_overrides_clear_profile_name() -> None:
    overrides = group_overrides({"early": 0.5, "middle": 0.25, "late": 0.125, "head": 0.25})
    assert "fls.profile_name=null" in overrides
    assert "fls.groups.late.output_multiplier=0.125" in overrides


def test_boundary_overrides_set_all_valid_boundaries_without_head() -> None:
    overrides = boundary_overrides({"after_early": 0.5, "after_middle": 1.0, "after_late": 0.125})
    assert "fls.boundaries.after_early.output_multiplier=0.5" in overrides
    assert "fls.boundaries.after_middle.output_multiplier=1.0" in overrides
    assert "fls.boundaries.after_late.output_multiplier=0.125" in overrides
    assert all("after_head" not in override for override in overrides)


def test_bn_mode_overrides_freeze_all() -> None:
    overrides, suffix = bn_mode_overrides("freeze_all")
    assert overrides == ["model.freeze_bn_affine=true", "model.freeze_bn_stats=true"]
    assert suffix == ["freeze_all"]


def test_interaction_values_for_pair_supports_group_specific_values() -> None:
    sweep = {
        "output_multipliers": [1.0],
        "interaction_values": {"early": [1.0, 2.0, 4.0], "late": [0.03125, 0.0625]},
    }
    first_values, second_values = interaction_values_for_pair(sweep, "early", "late")
    assert first_values == ["1.0", "2.0", "4.0"]
    assert second_values == ["0.03125", "0.0625"]


def test_interaction_values_for_boundary_pair_supports_boundary_specific_values() -> None:
    sweep = {
        "output_multipliers": [1.0],
        "interaction_values": {"after_early": [1.0, 2.0, 4.0], "after_late": [0.03125, 0.0625]},
    }
    first_values, second_values = interaction_values_for_boundary_pair(sweep, "after_early", "after_late")
    assert first_values == ["1.0", "2.0", "4.0"]
    assert second_values == ["0.03125", "0.0625"]
