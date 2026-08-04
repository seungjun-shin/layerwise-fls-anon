import torch

from fls.models.registry import build_model
from fls.models.vit import ViTCIFAR, vit_cifar
from fls.scaling.location_fls import apply_location_scaling


def _tiny_model(num_classes: int = 10) -> ViTCIFAR:
    # Tiny dims for speed: width=48, two layers per stage.
    return vit_cifar(
        num_classes=num_classes,
        width=48,
        use_bn=True,
        in_channels=3,
        patch_size=4,
        stage_depths=(1, 1, 1),
        num_heads=4,
    )


def test_forward_shape() -> None:
    model = _tiny_model(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_forward_with_features_keys() -> None:
    model = _tiny_model(num_classes=7)
    x = torch.randn(3, 3, 32, 32)
    logits, features = model.forward_with_features(x)
    assert logits.shape == (3, 7)
    assert {"early", "middle", "late", "head"} == set(features)
    for key in ("early", "middle", "late"):
        assert features[key].shape[0] == 3  # batch dim preserved
        assert features[key].dim() == 3  # [B, N, D] token tensor
    assert features["head"].shape == (3, 7)


def test_block_groups() -> None:
    model = _tiny_model()
    groups = model.get_block_groups()
    assert set(groups) == {"early", "middle", "late", "head"}
    for modules in groups.values():
        assert all(isinstance(m, torch.nn.Module) for m in modules)


def test_apply_location_scaling_runs() -> None:
    model = _tiny_model(num_classes=10)
    boundaries = {
        "after_early": {"init_scale": 1.0, "output_multiplier": 2.0},
        "after_middle": {"init_scale": 1.0, "output_multiplier": 0.5},
        "after_late": {"init_scale": 1.0, "output_multiplier": 1.5},
    }
    model = apply_location_scaling(model, boundaries)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_build_model() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {
            "name": "vit_cifar",
            "in_channels": 3,
            "num_classes": 10,
            "width": 48,
            "use_bn": True,
        },
    }
    model = build_model(config)
    assert isinstance(model, ViTCIFAR)
    x = torch.randn(2, 3, 32, 32)
    logits, features = model.forward_with_features(x)
    assert logits.shape == (2, 10)
    assert {"early", "middle", "late", "head"} == set(features)
