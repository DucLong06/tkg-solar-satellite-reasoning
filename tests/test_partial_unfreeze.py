"""Partial ViT unfreeze: requires_grad pattern + optimizer param groups.

Uses vit_tiny_patch16_224 (random init, no download) so the ViT block structure
is real but the test stays CPU-fast.
"""

from __future__ import annotations

import pytest
import torch

from src.satellite_encoder.satellite_encoder import SatelliteEncoder
from src.training.config import Config
from src.training.train_loop import _build_optimizer

VIT = "vit_tiny_patch16_224"


def _make(**kw) -> SatelliteEncoder:
    return SatelliteEncoder(in_channels=1, backbone=VIT, pretrained=False, **kw)


def test_full_freeze_still_works():
    enc = _make(freeze_backbone=True)
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    # heads stay trainable
    assert all(p.requires_grad for p in enc.frame_proj.parameters())


def test_no_freeze_still_works():
    enc = _make()
    assert all(p.requires_grad for p in enc.backbone.parameters())


def test_unfreeze_last_blocks_pattern():
    enc = _make(freeze_backbone=True, unfreeze_last_blocks=2)
    blocks = enc.backbone.blocks
    for blk in blocks[:-2]:
        assert not any(p.requires_grad for p in blk.parameters())
    for blk in blocks[-2:]:
        assert all(p.requires_grad for p in blk.parameters())
    assert all(p.requires_grad for p in enc.backbone.norm.parameters())
    # everything before the blocks (patch embed, pos embed, cls token) stays frozen
    assert not enc.backbone.patch_embed.proj.weight.requires_grad
    assert not enc.backbone.pos_embed.requires_grad
    assert not enc.backbone.cls_token.requires_grad


def test_unfreeze_requires_vit_backbone():
    with pytest.raises(ValueError, match="ViT-style"):
        SatelliteEncoder(in_channels=1, backbone="resnet18", unfreeze_last_blocks=2)


class _SatOnly(torch.nn.Module):
    """Minimal model exposing .sat_encoder the way TKGSolarModel does."""

    def __init__(self, enc: SatelliteEncoder) -> None:
        super().__init__()
        self.sat_encoder = enc


def test_optimizer_param_groups_split_and_lrs():
    cfg = Config(lr=1e-4, unfreeze_last_blocks=2, backbone_lr_scale=0.1)
    model = _SatOnly(_make(freeze_backbone=True, unfreeze_last_blocks=2))
    opt = _build_optimizer(model, cfg)
    assert len(opt.param_groups) == 2
    head_g, bb_g = opt.param_groups
    assert head_g["lr"] == pytest.approx(1e-4)
    assert bb_g["lr"] == pytest.approx(1e-5)
    # group contents: backbone group == exactly the trainable backbone params
    bb_trainable = [p for p in model.sat_encoder.backbone.parameters() if p.requires_grad]
    assert len(bb_g["params"]) == len(bb_trainable)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(head_g["params"]) + len(bb_g["params"]) == n_trainable


def test_optimizer_single_group_when_frozen():
    cfg = Config(lr=1e-4)  # unfreeze_last_blocks defaults to 0
    model = _SatOnly(_make(freeze_backbone=True))
    opt = _build_optimizer(model, cfg)
    assert len(opt.param_groups) == 1
