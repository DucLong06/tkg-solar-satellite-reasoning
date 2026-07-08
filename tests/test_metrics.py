"""M2 metric tests against hand-computed values + night-MAPE masking."""

from __future__ import annotations

import math

import numpy as np

from src.metrics.regression_metrics import compute_all, mae, mape, rmse


def test_perfect_prediction():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_hand_values():
    yt = [2.0, 4.0]
    yp = [0.0, 0.0]
    assert mae(yt, yp) == 3.0
    assert math.isclose(rmse(yt, yp), math.sqrt(10.0), rel_tol=1e-6)
    assert math.isclose(mape(yt, yp), 100.0, rel_tol=1e-6)


def test_mape_masks_night_zeros():
    # half the samples are 0 (night) -> excluded from MAPE, counted in fraction.
    yt = np.array([0.0, 0.0, 5.0, 10.0])
    yp = np.array([1.0, 2.0, 5.0, 10.0])
    out = compute_all(yt, yp, mape_min_value=1e-3)
    assert out["mape"] == 0.0                 # daytime preds perfect
    assert math.isclose(out["mape_excluded_frac"], 0.5)
