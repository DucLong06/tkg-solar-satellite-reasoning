"""Gaussian negative-log-likelihood for uncertainty-aware regression.

Standalone util for a future heteroscedastic head (mean + logvar). The current
point-estimate TKGSolarModel has no variance head, so this is NOT wired into the
default composite loss (probabilistic head deferred). Kept + tested
so enabling it later is a drop-in.
"""

from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * ( exp(-logvar) * (target-mean)^2 + logvar ), mean-reduced.

    logvar is clamped for numerical stability.
    """
    logvar = logvar.clamp(min=-10.0, max=10.0)
    return 0.5 * (torch.exp(-logvar) * (target - mean).pow(2) + logvar).mean()
