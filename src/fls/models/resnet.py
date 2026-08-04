from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fls.models.blocks import BasicBlock


class CosineClassifier(nn.Module):
    """Cosine-normalized classifier with a fixed, non-trainable logit scale."""

    def __init__(self, in_features: int, num_classes: int, logit_scale: float = 1.0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(num_classes)
        self.logit_scale = float(logit_scale)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.normalize(x, dim=1)
        weights = F.normalize(self.weight, dim=1)
        return self.logit_scale * F.linear(features, weights)


class ResNetCIFAR(nn.Module):
    """CIFAR-style ResNet with a 3x3 stride-1 stem."""

    def __init__(
        self,
        layers: list[int],
        num_classes: int = 10,
        width: int = 64,
        use_bn: bool = True,
        head_type: str = "linear",
        head_logit_scale: float = 1.0,
        pre_head_norm: str = "none",
    ) -> None:
        super().__init__()
        self.in_planes = width
        stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=not use_bn),
            nn.BatchNorm2d(width) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        layer1 = self._make_layer(width, layers[0], stride=1, use_bn=use_bn)
        layer2 = self._make_layer(width * 2, layers[1], stride=2, use_bn=use_bn)
        layer3 = self._make_layer(width * 4, layers[2], stride=2, use_bn=use_bn)
        layer4 = self._make_layer(width * 8, layers[3], stride=2, use_bn=use_bn)
        self.early = nn.Sequential(stem, layer1)
        self.middle = layer2
        self.late = nn.Sequential(layer3, layer4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = width * 8
        if pre_head_norm == "none":
            self.pre_head_norm = nn.Identity()
        elif pre_head_norm == "layer_norm_affine_free":
            self.pre_head_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown pre_head_norm: {pre_head_norm}")
        if head_type == "linear":
            self.head = nn.Linear(feature_dim, num_classes)
        elif head_type == "cosine":
            self.head = CosineClassifier(feature_dim, num_classes, head_logit_scale)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def _make_layer(self, planes: int, blocks: int, stride: int, use_bn: bool) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride, use_bn))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features: dict[str, torch.Tensor] = {}
        x = self.early(x)
        features["early"] = x
        x = self.middle(x)
        features["middle"] = x
        x = self.late(x)
        features["late"] = x
        x = self.pool(x).flatten(1)
        x = self.pre_head_norm(x)
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        return {
            "early": [self.early],
            "middle": [self.middle],
            "late": [self.late],
            "head": [self.head],
        }

    def ordered_blocks(self) -> list[nn.Module]:
        """Flat depth-ordered list of body blocks (stem + each BasicBlock).

        Treats the network as a depth axis rather than early/middle/late chunks,
        so a scaling boundary can be placed at an arbitrary depth position.
        For ResNet18 this yields 9 blocks: stem, then the 8 BasicBlocks of
        layer1..layer4. Indices map to the coarse boundaries as:
        after_early=2, after_middle=4, after_late=8.
        """
        stem, layer1 = self.early[0], self.early[1]
        layer2 = self.middle
        layer3, layer4 = self.late[0], self.late[1]
        blocks: list[nn.Module] = [stem]
        for layer in (layer1, layer2, layer3, layer4):
            blocks.extend(list(layer))
        return blocks


def resnet18_cifar(
    num_classes: int = 10,
    width: int = 64,
    use_bn: bool = True,
    head_type: str = "linear",
    head_logit_scale: float = 1.0,
    pre_head_norm: str = "none",
) -> ResNetCIFAR:
    """Build ResNet-18 for CIFAR-sized images."""
    return ResNetCIFAR(
        [2, 2, 2, 2],
        num_classes=num_classes,
        width=width,
        use_bn=use_bn,
        head_type=head_type,
        head_logit_scale=head_logit_scale,
        pre_head_norm=pre_head_norm,
    )
