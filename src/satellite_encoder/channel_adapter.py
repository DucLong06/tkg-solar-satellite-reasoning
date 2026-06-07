"""Channel adapter: map C-channel satellite input -> 3 channels for an ImageNet backbone.

A cheap 1x1 conv (per SatMAE / adapter pattern) lets us reuse pretrained RGB
weights on non-RGB multispectral input instead of retraining the stem.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] -> [N, out_channels, H, W]
        return self.proj(x)
