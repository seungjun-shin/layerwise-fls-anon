from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny, resnet50


class ImageNetResNet50(nn.Module):
    """Torchvision ResNet-50 exposed through the repository block-group API.

    The ImageNet classifier is replaced after loading pretrained weights, so the
    backbone starts from the requested ImageNet initialization while the target
    dataset head is initialized from the run seed.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = int(backbone.fc.in_features)

        self.early = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.middle = backbone.layer2
        self.late = nn.Sequential(backbone.layer3, backbone.layer4)
        self.pool = backbone.avgpool
        self.pre_head_norm = nn.Identity()
        self.head = nn.Linear(in_features, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and coarse features, including classifier input."""
        features: dict[str, torch.Tensor] = {}
        x = self.early(x)
        features["early"] = x
        x = self.middle(x)
        features["middle"] = x
        x = self.late(x)
        features["late"] = x
        x = self.pool(x).flatten(1)
        x = self.pre_head_norm(x)
        features["pre_head"] = x
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        """Return the standard early/middle/late/head parameter groups."""
        return {
            "early": [self.early],
            "middle": [self.middle],
            "late": [self.late],
            "head": [self.head],
        }


def resnet50_imagenet(num_classes: int, pretrained: bool = True) -> ImageNetResNet50:
    """Build an ImageNet-initialized ResNet-50 for visual transfer learning."""
    return ImageNetResNet50(num_classes=num_classes, pretrained=pretrained)


class ConvNeXtLateTail(nn.Module):
    """Final ConvNeXt stages followed by pooling and the pretrained final norm."""

    def __init__(
        self,
        body: nn.Module,
        avgpool: nn.Module,
        final_norm: nn.Module,
        flatten: nn.Module,
    ) -> None:
        super().__init__()
        self.body = body
        self.avgpool = avgpool
        self.final_norm = final_norm
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.avgpool(x)
        x = self.final_norm(x)
        return self.flatten(x)


class ImageNetConvNeXtTiny(nn.Module):
    """ImageNet ConvNeXt-Tiny with a post-final-norm classifier boundary.

    ``late`` owns the final LayerNorm parameters.  The exposed ``pool`` module
    is an identity placed immediately after that normalization, so an
    ``after_pool`` operational multiplier acts on the exact classifier input
    and upstream LR compensation includes the final normalization parameters.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.early = nn.Sequential(*list(backbone.features[:2]))
        self.middle = nn.Sequential(*list(backbone.features[2:4]))
        late_body = nn.Sequential(*list(backbone.features[4:]))
        self.late = ConvNeXtLateTail(
            late_body,
            backbone.avgpool,
            backbone.classifier[0],
            backbone.classifier[1],
        )
        self.pool = nn.Identity()
        self.pre_head_norm = nn.Identity()
        in_features = int(backbone.classifier[2].in_features)
        self.head = nn.Linear(in_features, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and coarse features, including post-norm classifier input."""
        features: dict[str, torch.Tensor] = {}
        x = self.early(x)
        features["early"] = x
        x = self.middle(x)
        features["middle"] = x
        x = self.late(x)
        features["late"] = x
        x = self.pool(x)
        x = self.pre_head_norm(x)
        features["pre_head"] = x
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        """Return standard groups with final normalization owned by ``late``."""
        return {
            "early": [self.early],
            "middle": [self.middle],
            "late": [self.late],
            "head": [self.head],
        }


def convnext_tiny_imagenet(
    num_classes: int, pretrained: bool = True
) -> ImageNetConvNeXtTiny:
    """Build ImageNet-initialized ConvNeXt-Tiny for visual transfer learning."""
    return ImageNetConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)
