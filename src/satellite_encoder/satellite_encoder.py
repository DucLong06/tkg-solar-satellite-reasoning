"""Satellite frame-sequence encoder -> F_sat [B, 128].

Per-frame timm backbone (channel-adapted) -> temporal Transformer over frames ->
mean-pool -> projection to the locked 128-dim embedding.

Backbone is configurable: ``resnet18`` (lightweight default, works at 64x64) or
``vit_base_patch16_224`` (paper-grade; requires img_size=224). ViT for the full
run is a Phase-6 accuracy lever, not a skeleton need.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.common.shapes import EMBED_DIM, SAT_CHANNELS, assert_embedding
from src.satellite_encoder.channel_adapter import ChannelAdapter


class SatelliteEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = SAT_CHANNELS,
        out_dim: int = EMBED_DIM,
        backbone: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.channel_adapter = ChannelAdapter(in_channels, 3)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=3, num_classes=0
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feat_dim = self.backbone.num_features

        self.frame_proj = nn.Linear(feat_dim, out_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=n_heads, dim_feedforward=out_dim * 2,
            batch_first=True, dropout=dropout,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, sat_seq: torch.Tensor) -> torch.Tensor:
        # sat_seq: [B, T, C, H, W]
        b, t, c, h, w = sat_seq.shape
        x = sat_seq.reshape(b * t, c, h, w)
        x = self.channel_adapter(x)              # [B*T, 3, H, W]
        x = self.backbone(x)                     # [B*T, feat_dim]
        x = self.frame_proj(x).reshape(b, t, -1)  # [B, T, out_dim]
        x = self.temporal(x)                     # [B, T, out_dim]
        pooled = x.mean(dim=1)                    # [B, out_dim]
        out = self.out_proj(pooled)
        assert_embedding(out, "SatelliteEncoder")
        return out
