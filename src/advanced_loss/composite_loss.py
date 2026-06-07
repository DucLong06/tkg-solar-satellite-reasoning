"""Composite loss = MSE + lambda_physics * physics_penalty.

MSE stays the default/baseline objective; the physics term is a small, toggleable
add-on (start at lambda=0 and ramp). The probabilistic term (lambda_prob) needs a
variance head the point-estimate model does not have yet, so it is deferred: a
warning is emitted if lambda_prob > 0 and the term is skipped (see
docs/assumptions.md).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.advanced_loss.physics_informed_loss import physics_penalty


class CompositeLoss(nn.Module):
    def __init__(self, lambda_prob: float = 0.0, lambda_physics: float = 0.0, ramp_max: float = 0.5) -> None:
        super().__init__()
        self.lambda_prob = lambda_prob
        self.lambda_physics = lambda_physics
        self.ramp_max = ramp_max
        self.mse = nn.MSELoss()
        if lambda_prob > 0:
            warnings.warn(
                "lambda_prob > 0 but the model is point-estimate (no variance head); "
                "probabilistic term skipped (deferred).",
                RuntimeWarning,
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.lambda_physics > 0:
            loss = loss + self.lambda_physics * physics_penalty(pred, self.ramp_max)
        return loss
