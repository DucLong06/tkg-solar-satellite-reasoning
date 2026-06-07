"""Phase 5 tests: TKGSolarModel forward [B,3], grad flow, composite loss runs."""

from __future__ import annotations

import torch

from src.common.shapes import N_HORIZONS
from src.advanced_loss.composite_loss import CompositeLoss
from src.advanced_loss.probabilistic_loss import gaussian_nll
from src.training.config import Config
from src.fusion_predictor.tkg_solar_model import TKGSolarModel


def test_full_model_forward_and_grad(pipeline_splits):
    cfg = Config(model="full", n_graph_nodes=8, sat_backbone="resnet18", pretrained_backbone=False)
    model = TKGSolarModel.from_config(cfg)
    batch = next(iter(pipeline_splits.train_loader))
    pred = model(batch)
    assert pred.shape == (batch["target"].shape[0], N_HORIZONS)
    assert not torch.isnan(pred).any()
    loss = torch.nn.functional.mse_loss(pred, batch["target"])
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_composite_loss_runs_without_nan():
    pred = torch.randn(8, N_HORIZONS, requires_grad=True)
    target = torch.rand(8, N_HORIZONS)
    loss = CompositeLoss(lambda_physics=0.1)(pred, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_gaussian_nll_finite():
    mean = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    target = torch.ones(4, 3)
    assert torch.isfinite(gaussian_nll(mean, logvar, target))
