from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fls.models.registry import build_model
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.position_fls import PositionScaledResNet
from fls.training.model_factory import build_scaled_model
from fls.training.optim import build_optimizer
from fls.training.protocol_audit import _boundary_observation
from fls.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "paper_reproduction" / "configs" / "practical_main"
TRANSFER_ROOT = ROOT / "paper_reproduction" / "configs" / "practical_transfer"
SCOPE_ROOT = ROOT / "paper_reproduction" / "configs" / "scope_preservation"
VGG_DEPTH_ROOT = ROOT / "paper_reproduction" / "configs" / "vgg_depth_extension"
HEAD_NORM_ROOT = ROOT / "paper_reproduction" / "configs" / "head_norm"


def _build(config_path: Path) -> tuple[dict, torch.nn.Module, torch.optim.Optimizer]:
    config = load_config(config_path)
    torch.manual_seed(int(config["training"]["seed"]))
    model = build_model(config)
    model = apply_location_scaling(model, config["fls"]["boundaries"])
    optimizer = build_optimizer(model, config)
    return config, model, optimizer


def _group_layout(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, tuple[float, tuple[str, ...]]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        str(group["name"]): (
            float(group["lr"]),
            tuple(sorted(names[id(parameter)] for parameter in group["params"])),
        )
        for group in optimizer.param_groups
    }


def test_final_activation_and_lr_only_have_identical_lr_allocation() -> None:
    _, act_model, act_optimizer = _build(CONFIG_ROOT / "p_act_final.yaml")
    _, lr_model, lr_optimizer = _build(CONFIG_ROOT / "p_lr_final.yaml")

    act_layout = _group_layout(act_model, act_optimizer)
    lr_layout = _group_layout(lr_model, lr_optimizer)

    assert act_layout == lr_layout
    assert act_layout["early"][0] == pytest.approx(0.00512 / 1.5)
    assert act_layout["middle"][0] == pytest.approx(0.00512 / 1.5)
    assert act_layout["late"][0] == pytest.approx(0.00512 / 1.5)
    assert act_layout["head"][0] == pytest.approx(0.00512)


def test_final_activation_changes_only_the_declared_boundary_scale() -> None:
    _, act_model, _ = _build(CONFIG_ROOT / "p_act_final.yaml")
    _, lr_model, _ = _build(CONFIG_ROOT / "p_lr_final.yaml")
    x = torch.randn(2, 3, 32, 32)

    act_observation = _boundary_observation(act_model, x)
    lr_observation = _boundary_observation(lr_model, x)

    assert act_observation["boundary"] == "after_late"
    assert act_observation["observed_scale_ratio"] == pytest.approx(1.5, rel=1e-6)
    assert lr_observation["observed_scale_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_paired_models_start_from_identical_trainable_tensors() -> None:
    _, act_model, _ = _build(CONFIG_ROOT / "p_act_final.yaml")
    _, lr_model, _ = _build(CONFIG_ROOT / "p_lr_final.yaml")

    act_parameters = dict(act_model.named_parameters())
    lr_parameters = dict(lr_model.named_parameters())
    assert act_parameters.keys() == lr_parameters.keys()
    for name in act_parameters:
        assert torch.equal(act_parameters[name], lr_parameters[name]), name


def test_vgg_transfer_uses_the_exact_final_cut_lr_only_control() -> None:
    _, act_model, act_optimizer = _build(TRANSFER_ROOT / "t_act_final.yaml")
    _, lr_model, lr_optimizer = _build(TRANSFER_ROOT / "t_lr_final.yaml")

    act_layout = _group_layout(act_model, act_optimizer)
    lr_layout = _group_layout(lr_model, lr_optimizer)
    assert act_layout == lr_layout
    assert act_layout["early"][0] == pytest.approx(0.65536)
    assert act_layout["middle"][0] == pytest.approx(0.65536)
    assert act_layout["late"][0] == pytest.approx(0.65536)
    assert act_layout["head"][0] == pytest.approx(0.04096)

    x = torch.randn(1, 3, 32, 32)
    assert _boundary_observation(act_model, x)["observed_scale_ratio"] == pytest.approx(
        0.0625, rel=1e-6
    )
    assert _boundary_observation(lr_model, x)["observed_scale_ratio"] == pytest.approx(
        1.0, rel=1e-6
    )


@pytest.mark.parametrize(
    ("family", "expected_ratio"),
    [
        ("stl10", 0.0625),
        ("svhn", 0.0625),
        ("noise20", 0.0625),
        ("noise50", 0.0625),
        ("cifar10", 0.0078125),
        ("nobn", 4.0),
        ("persistence", 0.0625),
    ],
)
def test_scope_activation_and_lr_only_use_identical_parameter_group_lrs(
    family: str,
    expected_ratio: float,
) -> None:
    _, act_model, act_optimizer = _build(SCOPE_ROOT / f"{family}_act.yaml")
    _, lr_model, lr_optimizer = _build(SCOPE_ROOT / f"{family}_lr.yaml")

    assert _group_layout(act_model, act_optimizer) == _group_layout(lr_model, lr_optimizer)
    x = torch.randn(1, 3, 32, 32)
    assert _boundary_observation(act_model, x)["observed_scale_ratio"] == pytest.approx(
        expected_ratio, rel=1e-6
    )
    assert _boundary_observation(lr_model, x)["observed_scale_ratio"] == pytest.approx(
        1.0, rel=1e-6
    )


@pytest.mark.parametrize("cut", [1, 3, 5, 7, 9, 11, 13])
def test_vgg_depth_extension_uses_exact_per_cut_lr_allocation(cut: int) -> None:
    act_config = load_config(VGG_DEPTH_ROOT / f"cut{cut}_act.yaml")
    lr_config = load_config(VGG_DEPTH_ROOT / f"cut{cut}_lr.yaml")
    torch.manual_seed(int(act_config["training"]["seed"]))
    act_model = build_scaled_model(act_config)
    act_optimizer = build_optimizer(act_model, act_config)
    torch.manual_seed(int(lr_config["training"]["seed"]))
    lr_model = build_scaled_model(lr_config)
    lr_optimizer = build_optimizer(lr_model, lr_config)

    assert isinstance(act_model, PositionScaledResNet)
    assert isinstance(lr_model, PositionScaledResNet)
    assert act_model.cut == lr_model.cut == cut
    assert act_model.mult == pytest.approx(0.0625)
    assert lr_model.mult == pytest.approx(1.0)
    assert _group_layout(act_model, act_optimizer) == _group_layout(lr_model, lr_optimizer)
    x = torch.randn(1, 3, 32, 32)
    act_observation = _boundary_observation(act_model, x)
    lr_observation = _boundary_observation(lr_model, x)
    assert act_observation["boundary"] == f"position_{cut}"
    assert lr_observation["boundary"] == f"position_{cut}"
    assert act_observation["observed_scale_ratio"] == pytest.approx(0.0625, rel=1e-6)
    assert lr_observation["observed_scale_ratio"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("variant", ["cosine", "preln"])
def test_head_norm_ablation_pairs_share_initialization_and_lr_layout(variant: str) -> None:
    _, act_model, act_optimizer = _build(HEAD_NORM_ROOT / f"{variant}_act.yaml")
    _, lr_model, lr_optimizer = _build(HEAD_NORM_ROOT / f"{variant}_lr.yaml")

    assert _group_layout(act_model, act_optimizer) == _group_layout(lr_model, lr_optimizer)
    act_parameters = dict(act_model.named_parameters())
    lr_parameters = dict(lr_model.named_parameters())
    assert act_parameters.keys() == lr_parameters.keys()
    assert all(torch.equal(act_parameters[name], lr_parameters[name]) for name in act_parameters)
    x = torch.randn(1, 3, 32, 32)
    assert _boundary_observation(act_model, x)["observed_scale_ratio"] == pytest.approx(
        0.0625, rel=1e-6
    )
    assert _boundary_observation(lr_model, x)["observed_scale_ratio"] == pytest.approx(
        1.0, rel=1e-6
    )
