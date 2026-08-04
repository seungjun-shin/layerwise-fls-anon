import torch

from fls.models.registry import build_model
from fls.models.resnet import CosineClassifier


def _config(name: str) -> dict:
    width = 8 if name not in {"resnet18_cifar", "vgg19_bn_cifar"} else 4
    return {
        "data": {"num_classes": 10},
        "model": {"name": name, "in_channels": 3, "num_classes": 10, "width": width, "use_bn": True},
    }


def test_models_forward_and_groups() -> None:
    for name in ["small_cnn", "vgg_like", "vgg19_bn_cifar", "resnet18_cifar"]:
        model = build_model(_config(name))
        x = torch.randn(2, 3, 32, 32)
        logits, features = model.forward_with_features(x)
        assert logits.shape == (2, 10)
        assert {"early", "middle", "late", "head"}.issubset(set(model.get_block_groups()))
        assert features


def test_resnet_cosine_head_uses_fixed_unit_scale() -> None:
    config = _config("resnet18_cifar")
    config["model"].update({"head_type": "cosine", "head_logit_scale": 1.0})
    model = build_model(config).eval()

    with torch.no_grad():
        logits, _ = model.forward_with_features(torch.randn(2, 3, 32, 32))

    assert isinstance(model.head, CosineClassifier)
    assert model.head.logit_scale == 1.0
    assert logits.abs().max().item() <= 1.0 + 1e-6


def test_resnet_affine_free_pre_head_layer_norm() -> None:
    config = _config("resnet18_cifar")
    config["model"]["pre_head_norm"] = "layer_norm_affine_free"
    model = build_model(config).eval()

    with torch.no_grad():
        captured: dict[str, torch.Tensor] = {}
        handle = model.pre_head_norm.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("head_input", output)
        )
        model.forward_with_features(torch.randn(2, 3, 32, 32))
        handle.remove()

    assert isinstance(model.pre_head_norm, torch.nn.LayerNorm)
    assert not model.pre_head_norm.elementwise_affine
    assert torch.allclose(
        captured["head_input"].mean(dim=1),
        torch.zeros(2),
        atol=1e-5,
    )
