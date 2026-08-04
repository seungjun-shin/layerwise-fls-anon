from __future__ import annotations

import torch
from torch import nn

from fls.models.blocks import conv_block


class VGGLike(nn.Module):
    """Compact VGG-like CNN for CIFAR-sized images."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: int = 32, use_bn: bool = True) -> None:
        super().__init__()
        self.early = nn.Sequential(conv_block(in_channels, width, use_bn), conv_block(width, width, use_bn))
        self.middle = nn.Sequential(conv_block(width, width * 2, use_bn), conv_block(width * 2, width * 2, use_bn))
        self.late = nn.Sequential(conv_block(width * 2, width * 4, use_bn))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width * 4, num_classes)

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
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        return {"early": [self.early], "middle": [self.middle], "late": [self.late], "head": [self.head]}


def _vgg_stage(in_channels: int, out_channels: int, convs: int, use_bn: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    for _ in range(convs):
        layers.append(nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        current = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG19BNCIFAR(nn.Module):
    """VGG19-BN adapted to CIFAR-sized images.

    The convolutional layout follows VGG19: 2-2-4-4-4 convolutional stages.
    The classifier is a single linear head after global average pooling, which
    is standard for CIFAR-scale VGG variants and keeps the FLS intervention
    points aligned with the repository's block-group convention.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, width: int = 64, use_bn: bool = True) -> None:
        super().__init__()
        self.stage1 = _vgg_stage(in_channels, width, convs=2, use_bn=use_bn)
        self.stage2 = _vgg_stage(width, width * 2, convs=2, use_bn=use_bn)
        self.stage3 = _vgg_stage(width * 2, width * 4, convs=4, use_bn=use_bn)
        self.stage4 = _vgg_stage(width * 4, width * 8, convs=4, use_bn=use_bn)
        self.stage5 = _vgg_stage(width * 8, width * 8, convs=4, use_bn=use_bn)
        self.early = nn.Sequential(self.stage1, self.stage2)
        self.middle = self.stage3
        self.late = nn.Sequential(self.stage4, self.stage5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width * 8, num_classes)

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
        logits = self.head(x)
        features["head"] = logits
        return logits, features

    def get_block_groups(self) -> dict[str, list[nn.Module]]:
        return {"early": [self.early], "middle": [self.middle], "late": [self.late], "head": [self.head]}

    def ordered_blocks(self) -> list[nn.Module]:
        """Flat depth-ordered list of conv blocks for position-based scaling.

        Each VGG19 conv unit (conv, optional BN, ReLU) is one depth block; the
        stage-ending MaxPool is folded into the last block of its stage. This
        yields a 2+2+4+4+4 = 16 block depth axis, mirroring ResNet's
        ``ordered_blocks`` so a scaling boundary can be placed at an arbitrary
        depth. The returned blocks wrap the same submodules (shared parameters,
        no duplication), and applying them in order reproduces the
        stage1..stage5 forward pass. The coarse early/middle/late boundaries map
        to indices after_early=3, after_middle=7, after_late=15.
        """
        blocks: list[nn.Module] = []
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4, self.stage5):
            stage_blocks: list[nn.Module] = []
            unit: list[nn.Module] = []
            for m in stage.children():
                unit.append(m)
                if isinstance(m, nn.ReLU):
                    stage_blocks.append(nn.Sequential(*unit))
                    unit = []
            if unit:  # trailing MaxPool (no ReLU after it): fold into last block
                last = list(stage_blocks[-1].children())
                stage_blocks[-1] = nn.Sequential(*last, *unit)
            blocks.extend(stage_blocks)
        return blocks
