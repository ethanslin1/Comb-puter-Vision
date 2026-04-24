"""UNet-style segmentation head on a DINOv2 ViT-S encoder.

Architecture
------------
- **Encoder**: ``dinov2_vits14`` loaded from ``facebookresearch/dinov2`` via
  ``torch.hub``. Patch size 14, embed dim 384. Pretrained weights by default;
  frozen by default (we have 42 training frames, fine-tuning the backbone
  overfits immediately).
- **Decoder**: progressive bilinear upsample + 3×3 conv blocks, no skip
  connections (ViT has no multi-scale feature pyramid). Four blocks take the
  H/14 × W/14 feature map up to roughly input resolution; a final bilinear
  interpolate snaps to exact ``(H, W)``, then a 1×1 conv projects to
  ``num_classes``.

The encoder expects input spatial dims divisible by 14. ``build_model`` does
not enforce this — the caller should pick an ``image_size`` like 518 (= 37·14)
or 224 (= 16·14). The augment pipeline already produces square inputs.

Output is logits of shape ``(B, num_classes, H, W)``. Apply ``softmax``
downstream for probabilities; ``argmax`` for class indices.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("models.segmentation")

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}
DINOV2_PATCH = 14


# ---------- Encoder --------------------------------------------------------


class DINOv2Encoder(nn.Module):
    """Wraps a DINOv2 ViT and returns patch features as a dense map.

    ``forward(x)`` returns a tensor of shape ``(B, embed_dim, H/14, W/14)``.
    The CLS token is dropped; patch tokens are reshaped back to a 2D grid.
    """

    def __init__(
        self,
        name: str = "dinov2_vits14",
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if name not in DINOV2_EMBED_DIMS:
            raise ValueError(
                f"unknown DINOv2 variant {name!r}; known: {sorted(DINOV2_EMBED_DIMS)}"
            )
        self.name = name
        self.embed_dim = DINOV2_EMBED_DIMS[name]
        self.patch = DINOV2_PATCH

        backbone = torch.hub.load(DINOV2_REPO, name, pretrained=pretrained, trust_repo=True)
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.frozen = freeze

    def train(self, mode: bool = True) -> "DINOv2Encoder":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if H % self.patch or W % self.patch:
            raise ValueError(
                f"input H={H}, W={W} not divisible by patch={self.patch}; "
                "resize to a multiple of 14"
            )
        feats = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=False, norm=True
        )
        return feats[0]  # (B, C, H/14, W/14)


# ---------- Decoder --------------------------------------------------------


class _UpBlock(nn.Module):
    """Upsample 2× then two 3×3 convs with BatchNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class UNetDecoder(nn.Module):
    """Dense prediction head: 4× progressive upsample + final 1×1 to classes."""

    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, decoder_channels[0], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        prev = decoder_channels[0]
        for ch in decoder_channels[1:]:
            blocks.append(_UpBlock(prev, ch))
            prev = ch
        # One extra up-block at the final channel width so we get 2^(len-1)
        # upsamples total from the projected feature map.
        blocks.append(_UpBlock(prev, prev))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(prev, num_classes, 1)

    def forward(self, feat: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.proj(feat)
        for block in self.blocks:
            x = block(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ---------- Full model -----------------------------------------------------


@dataclass
class SegModelConfig:
    num_classes: int = 2
    encoder: str = "dinov2_vits14"
    pretrained: bool = True
    freeze_encoder: bool = True
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32)


class DINOv2UNet(nn.Module):
    """DINOv2 encoder + UNet-style upsampling decoder."""

    def __init__(self, cfg: SegModelConfig | None = None, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SegModelConfig()
        if encoder is None:
            self.encoder: nn.Module = DINOv2Encoder(
                name=self.cfg.encoder,
                pretrained=self.cfg.pretrained,
                freeze=self.cfg.freeze_encoder,
            )
            in_ch = DINOV2_EMBED_DIMS[self.cfg.encoder]
        else:
            self.encoder = encoder
            in_ch = getattr(encoder, "embed_dim", None)
            if in_ch is None:
                raise ValueError("custom encoder must expose .embed_dim")
        self.decoder = UNetDecoder(
            in_ch=in_ch,
            num_classes=self.cfg.num_classes,
            decoder_channels=self.cfg.decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.decoder(feat, target_size=x.shape[-2:])

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# ---------- Factory --------------------------------------------------------


def build_model(cfg: dict[str, Any] | SegModelConfig | None = None) -> DINOv2UNet:
    """Build a ``DINOv2UNet`` from a dict config or ``SegModelConfig``."""
    if cfg is None:
        return DINOv2UNet()
    if isinstance(cfg, SegModelConfig):
        return DINOv2UNet(cfg)
    model_cfg = SegModelConfig(
        num_classes=int(cfg.get("num_classes", 2)),
        encoder=str(cfg.get("encoder", "dinov2_vits14")),
        pretrained=bool(cfg.get("pretrained", True)),
        freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        decoder_channels=tuple(cfg.get("decoder_channels", (256, 128, 64, 32))),
    )
    return DINOv2UNet(model_cfg)
