from __future__ import annotations

import torch
from torch import nn


class PatchEmbedStem(nn.Module):
    """Patchify a CIFAR image, linearly embed, prepend a class token, add positions.

    Input:  ``[B, in_channels, H, W]``
    Output: ``[B, N + 1, dim]`` token sequence (class token at index 0).

    The learnable class token and positional embedding live here so that they
    are owned by the ``early`` region (this module is the first sub-module of
    ``self.early``). That keeps every trainable parameter assigned to a block
    group in :meth:`ViTCIFAR.get_block_groups`.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        dim: int = 192,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)  # [B, num_patches + 1, dim]
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block (MHSA + MLP, GELU).

    Operates on and returns a ``[B, N, dim]`` token sequence, so the whole
    sequence tolerates the scalar multiplication applied by
    ``apply_location_scaling`` at region boundaries.
    """

    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTPool(nn.Module):
    """Final LayerNorm followed by class-token selection -> ``[B, dim]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x[:, 0]  # class token


class ViTCIFAR(nn.Module):
    """Small Vision Transformer for 32x32 images, exposing the FLS block-group interface.

    The transformer encoder of ``depth`` layers is split into three stages mapped
    to the ``early`` / ``middle`` / ``late`` regions. The patch embed plus the
    class and positional embeddings live inside ``self.early`` (mirroring the
    ResNet where ``early = stem + layer1``), so that scaling the ``early`` region
    multiplies the embedded token sequence and so that every trainable parameter
    is owned by a block group.

    ``use_bn`` is accepted for interface compatibility but ignored: a ViT uses
    LayerNorm throughout regardless of this flag.

    Region tensor shapes (all ``[B, N + 1, dim]``) make the scalar multiplication
    performed by ``apply_location_scaling`` well defined.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 192,
        use_bn: bool = True,  # noqa: ARG002 - accepted for interface parity, ignored (ViT uses LayerNorm)
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        depth: int = 6,
        stage_depths: tuple[int, int, int] | None = None,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = width
        if stage_depths is None:
            base = depth // 3
            rem = depth - base * 3
            # Distribute any remainder to the later stages.
            stage_depths = (base, base + (rem >= 2), base + (rem >= 1))
        if sum(stage_depths) <= 0:
            raise ValueError(f"stage_depths must sum to a positive depth, got {stage_depths}")

        self.dim = dim
        self.stage_depths = tuple(stage_depths)

        def make_stage(n: int) -> nn.Sequential:
            return nn.Sequential(
                *[TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(n)]
            )

        stem = PatchEmbedStem(image_size, patch_size, in_channels, dim)
        # early = patch embed (+ cls/pos) + first transformer stage.
        self.early = nn.Sequential(stem, make_stage(stage_depths[0]))
        self.middle = make_stage(stage_depths[1])
        self.late = make_stage(stage_depths[2])
        self.pool = ViTPool(dim)
        self.head = nn.Linear(dim, num_classes)

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
        x = self.pool(x)
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


def vit_cifar(
    num_classes: int = 10,
    width: int = 192,
    use_bn: bool = True,
    in_channels: int = 3,
    **kwargs,
) -> ViTCIFAR:
    """Build a small CIFAR Vision Transformer with the FLS block-group interface."""
    return ViTCIFAR(
        num_classes=num_classes,
        width=width,
        use_bn=use_bn,
        in_channels=in_channels,
        **kwargs,
    )
