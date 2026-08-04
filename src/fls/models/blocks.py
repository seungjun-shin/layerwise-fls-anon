from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, use_bn: bool = True) -> nn.Sequential:
    """Create a simple Conv-BN-ReLU-MaxPool block."""
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_bn: bool = True) -> None:
        super().__init__()
        bias = not use_bn
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        if stride != 1 or in_planes != planes:
            shortcut: list[nn.Module] = [nn.Conv2d(in_planes, planes, 1, stride=stride, bias=bias)]
            if use_bn:
                shortcut.append(nn.BatchNorm2d(planes))
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)

