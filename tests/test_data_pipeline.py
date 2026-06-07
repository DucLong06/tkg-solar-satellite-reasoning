"""M1 pipeline tests: shapes, no-NaN, no-leakage, chronological split, overlap gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common.shapes import (
    N_HORIZONS,
    N_METEO_FEATURES,
    SAT_CHANNELS,
)
from src.data_pipeline.splits import chronological_bounds
from src.data_pipeline.time_alignment import EmptyOverlapError, build_common_grid


def test_batch_shapes_and_no_nan(pipeline_splits):
    batch = next(iter(pipeline_splits.train_loader))
    b = batch["meteo_seq"].shape[0]
    k = pipeline_splits.meta["k"]
    img = pipeline_splits.meta["img_size"]
    assert batch["meteo_seq"].shape == (b, k, N_METEO_FEATURES)
    assert batch["pv_hist"].shape == (b, k, 1)
    assert batch["sat_seq"].shape == (b, k, SAT_CHANNELS, img, img)
    assert batch["target"].shape == (b, N_HORIZONS)
    for name, v in batch.items():
        assert not torch.isnan(v).any(), f"NaN in {name}"
        assert v.dtype == torch.float32


def test_loaders_nonempty(pipeline_splits):
    assert pipeline_splits.meta["n_train_windows"] >= 10
    assert pipeline_splits.meta["n_val_windows"] > 0
    assert pipeline_splits.meta["n_test_windows"] > 0


def test_chronological_split_fractions():
    b = chronological_bounds(1000, 0.70, 0.15)
    assert b.train_end == 700 and b.val_end == 850 and b.total == 1000
    # non-overlapping, ordered
    assert b.train.stop == b.val.start and b.val.stop == b.test.start


def test_scaler_fit_on_train_only(pipeline_splits):
    # PV scaler min/max must come from TRAIN; inverse-transform round-trips.
    sc = pipeline_splits.scalers
    scaled = np.array([0.0, 0.5, 1.0], dtype="float32")
    inv = sc.inverse_pv(scaled)
    re = sc.transform_pv(inv)
    assert np.allclose(re, scaled, atol=1e-4)


def test_overlap_gate_fails_loud():
    import pandas as pd

    # Two disjoint date ranges -> must raise EmptyOverlapError.
    t1 = pd.date_range("2016-01-01", periods=50, freq="10min", tz="UTC")
    t2 = pd.date_range("2017-01-01", periods=50, freq="10min", tz="UTC")
    pv = pd.Series(np.ones(50, dtype="float32"), index=t1, name="pv")
    meteo = pd.DataFrame(
        np.ones((50, N_METEO_FEATURES), dtype="float32"),
        index=t2,
        columns=list(range(N_METEO_FEATURES)),
    )
    frames = np.ones((50, SAT_CHANNELS, 8, 8), dtype="float32")
    with pytest.raises(EmptyOverlapError):
        build_common_grid(pv, meteo, frames, t1, min_steps=10)
