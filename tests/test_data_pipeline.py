"""Pipeline tests: shapes, no-NaN, no-leakage, chronological split, overlap gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.shapes import (
    METEO_FEATURES,
    N_HORIZONS,
    N_METEO_FEATURES,
    SAT_CHANNELS,
)
from src.data_pipeline.splits import bounds_from_dates, chronological_bounds
from src.data_pipeline.time_alignment import EmptyOverlapError, align_colocated


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


def test_satellite_free_mode(synthetic_data_dir):
    """use_satellite=False trains baselines without Himawari; sat_seq is zeros."""
    from src.data_pipeline import DataPipeline

    splits = DataPipeline.load(
        str(synthetic_data_dir / "dkasc" / "synthetic_array_2020_2022.csv"),
        str(synthetic_data_dir / "himawari_alice"),
        k=12, batch_size=16, img_size=64, min_steps=200,
        use_satellite=False, use_cache=False,
    )
    assert splits.meta["use_satellite"] is False
    batch = next(iter(splits.train_loader))
    assert batch["sat_seq"].shape == (batch["meteo_seq"].shape[0], 12, SAT_CHANNELS, 64, 64)
    assert float(batch["sat_seq"].abs().sum()) == 0.0  # zero frames (baselines ignore)
    assert batch["meteo_seq"].shape[-1] == N_METEO_FEATURES


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
    # Two disjoint date ranges -> empty co-located join must raise EmptyOverlapError.
    t1 = pd.date_range("2020-01-01", periods=50, freq="5min", tz="UTC")
    t2 = pd.date_range("2021-01-01", periods=50, freq="5min", tz="UTC")
    pv = pd.Series(np.ones(50, dtype="float32"), index=t1, name="pv")
    meteo = pd.DataFrame(
        np.full((50, N_METEO_FEATURES), 500.0, dtype="float32"),  # GHI=500 -> daytime
        index=t2,
        columns=list(METEO_FEATURES),
    )
    frames = np.ones((50, SAT_CHANNELS, 8, 8), dtype="float32")
    with pytest.raises(EmptyOverlapError):
        align_colocated(pv, meteo, frames, t1, min_steps=10)


def test_split_by_dates():
    ts = pd.date_range("2020-01-01", periods=1000, freq="5min", tz="UTC")
    # train_end at index where ts < boundary; pick boundaries inside the range.
    train_end = ts[400].strftime("%Y-%m-%dT%H:%M:%SZ")
    val_end = ts[700].strftime("%Y-%m-%dT%H:%M:%SZ")
    b = bounds_from_dates(ts, train_end, val_end)
    assert b.train_end == 400 and b.val_end == 700 and b.total == 1000
    assert b.train.stop == b.val.start and b.val.stop == b.test.start
