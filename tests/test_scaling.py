import torch
import pytest

from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import OutputScaledRegion, apply_location_scaling
from fls.scaling.position_fls import apply_position_scaling
from fls.scaling.profiles import build_profile
from fls.training.optim import build_optimizer


def test_global_output_multiplier() -> None:
    config = {"data": {"num_classes": 10}, "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True}}
    base = build_model(config).eval()
    wrapped = GlobalOutputMultiplier(base, 0.5).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = 0.5 * base(x)
        actual = wrapped(x)
    assert torch.allclose(actual, expected)


def test_global_lr_only_compensation_uses_reference_multiplier() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True},
        "fls": {
            "mode": "global",
            "global": {
                "output_multiplier": 1.0,
                "lr_compensation": True,
                "lr_compensation_multiplier": 0.0625,
            },
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.04096,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
    }
    model = GlobalOutputMultiplier(build_model(config), 1.0)
    optimizer = build_optimizer(model, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.65536)


def test_blockwise_scaling_forward() -> None:
    config = {"data": {"num_classes": 10}, "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True}}
    model = build_model(config)
    profile = {"early": {"output_multiplier": 1.0, "init_scale": 1.0}}
    model = apply_blockwise_scaling(model, profile)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_location_scaling_wraps_only_valid_representation_boundary() -> None:
    config = {"data": {"num_classes": 10}, "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True}}
    model = build_model(config)
    model = apply_location_scaling(model, {"after_late": {"output_multiplier": 0.5, "init_scale": 1.0}})
    assert isinstance(model.late, OutputScaledRegion)
    assert not isinstance(model.head, OutputScaledRegion)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_location_scaling_rejects_head_boundary() -> None:
    config = {"data": {"num_classes": 10}, "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True}}
    model = build_model(config)
    with pytest.raises(ValueError, match="Unknown scaling boundary"):
        apply_location_scaling(model, {"after_head": {"output_multiplier": 0.5, "init_scale": 1.0}})


def test_progressive_profile_order() -> None:
    profile = build_profile("progressive_increasing", ["early", "middle", "late"])
    assert profile["early"]["output_multiplier"] > profile["middle"]["output_multiplier"]
    assert profile["middle"]["output_multiplier"] > profile["late"]["output_multiplier"]


def test_uniform_same_average_profile_matches_progressive_budget() -> None:
    groups = ["early", "middle", "late", "head"]
    progressive = build_profile("progressive_increasing", groups)
    same_average = build_profile("uniform_same_average", groups)
    progressive_mean = sum(v["output_multiplier"] for v in progressive.values()) / len(groups)
    assert all(v["output_multiplier"] == progressive_mean for v in same_average.values())


def test_blockwise_lr_compensation_uses_upstream_cumulative_groups() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True},
        "fls": {
            "mode": "blockwise",
            "profile_name": None,
            "groups": {
                "early": {"output_multiplier": 0.5, "init_scale": 1.0},
                "middle": {"output_multiplier": 1.0, "init_scale": 1.0},
                "late": {"output_multiplier": 2.0, "init_scale": 1.0},
                "head": {"output_multiplier": 1.0, "init_scale": 1.0},
            },
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "blockwise_lr_compensation": "upstream_cumulative",
        },
    }
    model = apply_blockwise_scaling(build_model(config), config["fls"]["groups"])
    optimizer = build_optimizer(model, config)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs["early"] == 0.1
    assert lrs["middle"] == 0.05
    assert lrs["late"] == 0.05
    assert lrs["head"] == 0.1


def test_location_lr_compensation_uses_upstream_boundaries() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True},
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 0.5, "init_scale": 1.0},
                "after_middle": {"output_multiplier": 2.0, "init_scale": 1.0},
                "after_late": {"output_multiplier": 0.25, "init_scale": 1.0},
            },
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "location_lr_compensation": "upstream",
        },
    }
    model = apply_location_scaling(build_model(config), config["fls"]["boundaries"])
    optimizer = build_optimizer(model, config)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs["early"] == pytest.approx(0.4)
    assert lrs["middle"] == pytest.approx(0.2)
    assert lrs["late"] == pytest.approx(0.4)
    assert lrs["head"] == pytest.approx(0.1)


def test_location_lr_only_compensation_uses_reference_multipliers() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {"name": "small_cnn", "width": 8, "num_classes": 10, "use_bn": True},
        "fls": {
            "mode": "location",
            "boundaries": {
                "after_early": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_middle": {"output_multiplier": 1.0, "init_scale": 1.0},
                "after_late": {"output_multiplier": 1.0, "init_scale": 1.0},
            },
        },
        "training": {
            "optimizer": "sgd",
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "location_lr_compensation": "upstream",
            "location_lr_compensation_multipliers": {
                "after_early": 1.0,
                "after_middle": 1.0,
                "after_late": 0.25,
            },
        },
    }
    model = apply_location_scaling(build_model(config), config["fls"]["boundaries"])
    optimizer = build_optimizer(model, config)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs["early"] == pytest.approx(0.4)
    assert lrs["middle"] == pytest.approx(0.4)
    assert lrs["late"] == pytest.approx(0.4)
    assert lrs["head"] == pytest.approx(0.1)


def test_position_scaling_exposes_feature_and_group_interfaces() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {"name": "resnet18_cifar", "width": 8, "num_classes": 10, "use_bn": True},
    }
    scaled = apply_position_scaling(build_model(config), position=8, multiplier=0.25).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        logits = scaled(x)
        feature_logits, features = scaled.forward_with_features(x)

    assert torch.allclose(logits, feature_logits)
    assert set(features) == {"early", "middle", "late", "head"}
    assert set(scaled.get_block_groups()) == {"early", "middle", "late", "head"}
    assert len(scaled.ordered_blocks()) == 9
