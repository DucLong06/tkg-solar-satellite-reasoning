"""Physics-informed penalties enforcing basic PV-power consistency.

Paper is vague on the physical rules; these are conservative, documented choices
(see docs/assumptions.md):
  - non-negativity: PV power cannot be negative
  - ramp limit: power should not jump implausibly between adjacent horizons
Penalties operate on predictions only (in scaled units) and are 0 when satisfied.
"""

from __future__ import annotations

import torch


def non_negativity_penalty(pred: torch.Tensor) -> torch.Tensor:
    return torch.relu(-pred).pow(2).mean()


def ramp_penalty(pred: torch.Tensor, ramp_max: float = 0.5) -> torch.Tensor:
    """Penalize horizon-to-horizon changes exceeding ``ramp_max`` (scaled units)."""
    if pred.shape[-1] < 2:
        return pred.new_zeros(())
    diff = (pred[:, 1:] - pred[:, :-1]).abs()
    return torch.relu(diff - ramp_max).pow(2).mean()


def physics_penalty(pred: torch.Tensor, ramp_max: float = 0.5) -> torch.Tensor:
    return non_negativity_penalty(pred) + ramp_penalty(pred, ramp_max)
