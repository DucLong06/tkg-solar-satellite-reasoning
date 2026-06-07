"""Chronological 70/15/15 train/val/test split (no shuffle across time).

Test is the latest period. Splitting happens BEFORE windowing so each split is
windowed independently — windows never straddle a split boundary (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    total: int

    @property
    def train(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test(self) -> slice:
        return slice(self.val_end, self.total)


def chronological_bounds(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> SplitBounds:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 and each in (0,1)")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return SplitBounds(train_end=train_end, val_end=val_end, total=n)
