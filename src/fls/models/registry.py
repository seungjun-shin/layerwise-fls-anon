from __future__ import annotations

from torch import nn

from fls.models.cnn import SmallCNN
from fls.models.imagenet import convnext_tiny_imagenet, resnet50_imagenet
from fls.models.resnet import resnet18_cifar
from fls.models.vgg import VGG19BNCIFAR, VGGLike
from fls.models.vit import vit_cifar


def build_model(config: dict) -> nn.Module:
    """Build a model from config."""
    cfg = config["model"]
    name = cfg["name"]
    kwargs = {
        "num_classes": int(cfg.get("num_classes", config["data"].get("num_classes", 10))),
        "width": int(cfg.get("width", 32)),
        "use_bn": bool(cfg.get("use_bn", True)),
    }
    if name == "small_cnn":
        return SmallCNN(in_channels=int(cfg.get("in_channels", 3)), **kwargs)
    if name == "vgg_like":
        return VGGLike(in_channels=int(cfg.get("in_channels", 3)), **kwargs)
    if name == "vgg19_bn_cifar":
        return VGG19BNCIFAR(in_channels=int(cfg.get("in_channels", 3)), **kwargs)
    if name == "resnet18_cifar":
        return resnet18_cifar(
            **kwargs,
            head_type=str(cfg.get("head_type", "linear")),
            head_logit_scale=float(cfg.get("head_logit_scale", 1.0)),
            pre_head_norm=str(cfg.get("pre_head_norm", "none")),
        )
    if name == "resnet50_imagenet":
        return resnet50_imagenet(
            num_classes=kwargs["num_classes"],
            pretrained=bool(cfg.get("pretrained", True)),
        )
    if name == "convnext_tiny_imagenet":
        return convnext_tiny_imagenet(
            num_classes=kwargs["num_classes"],
            pretrained=bool(cfg.get("pretrained", True)),
        )
    if name == "vit_cifar":
        return vit_cifar(in_channels=int(cfg.get("in_channels", 3)), **kwargs)
    raise ValueError(f"Unknown model: {name}")
