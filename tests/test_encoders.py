"""M5/M6 encoder tests: output [B,128], no-NaN, grads flow, handles C!=3."""

from __future__ import annotations

import torch

from src.common.shapes import EMBED_DIM, N_METEO_FEATURES
from src.meteo_encoder.meteo_encoder import MeteoEncoder
from src.satellite_encoder.satellite_encoder import SatelliteEncoder


def test_meteo_encoder_shape_and_grad():
    enc = MeteoEncoder()
    x = torch.randn(4, 12, N_METEO_FEATURES, requires_grad=True)
    out = enc(x)
    assert out.shape == (4, EMBED_DIM)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert x.grad is not None


def test_satellite_encoder_shape_and_grad():
    enc = SatelliteEncoder(in_channels=1, backbone="resnet18", pretrained=False)
    x = torch.randn(2, 6, 1, 64, 64, requires_grad=True)  # C=1 (non-RGB) via adapter
    out = enc(x)
    assert out.shape == (2, EMBED_DIM)
    assert not torch.isnan(out).any()
    out.sum().backward()
    assert x.grad is not None
