"""Chronological train/val/test split (no shuffle across time).

Two split modes, both BEFORE windowing so windows never straddle a split boundary
(no leakage):
  - ``bounds_from_dates``: fixed calendar boundaries (the DKASC reproduction split,
    e.g. train 2020-01..2021-09, val 2021-10..12, test 2022).
  - ``chronological_bounds``: fractional 70/15/15 (used for smoke runs / tests on
    short synthetic windows where fixed dates do not apply).

Test is always the latest period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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


def bounds_from_dates(timestamps, train_end: str, val_end: str) -> SplitBounds:
    """Index bounds from calendar boundaries on a sorted UTC timestamp index.

    ``train_end`` / ``val_end`` are exclusive upper boundaries (ISO date strings):
    train = ``[.. , train_end)``, val = ``[train_end, val_end)``, test = ``[val_end, ..]``.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    train_e = pd.Timestamp(train_end, tz="UTC")
    val_e = pd.Timestamp(val_end, tz="UTC")
    if not (ts[0] < train_e <= val_e <= ts[-1] + pd.Timedelta(days=1)):
        raise ValueError(
            f"split dates out of range: data {ts[0]}..{ts[-1]}, "
            f"train_end={train_end}, val_end={val_end}"
        )
    train_end_idx = int((ts < train_e).sum())
    val_end_idx = int((ts < val_e).sum())
    b = SplitBounds(train_end=train_end_idx, val_end=val_end_idx, total=len(ts))
    if b.train_end == 0 or b.val_end == b.train_end or b.val_end == b.total:
        raise ValueError(
            f"empty split from dates: train={b.train_end}, "
            f"val={b.val_end - b.train_end}, test={b.total - b.val_end}"
        )
    return b
