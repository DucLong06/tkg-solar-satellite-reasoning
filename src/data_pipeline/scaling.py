"""Min-Max scaling fit on TRAIN ONLY (no leakage), with PV inverse-transform.

Two separate scalers so PV inverse-transform at eval is unambiguous:
  - ``meteo_scaler``: per-column MinMax over the meteo features (see METEO_FEATURES)
  - ``pv_scaler``   : single-column MinMax over PV power (used to inverse metrics)
Satellite frames are assumed near-[0,1] (visible reflectance); a train-fit
scalar min/max keeps them bounded without per-pixel scalers.

Note: a chronologically-later test peak can exceed the train max -> scaled value
> 1. We do NOT clip (clipping would corrupt the target); metrics are computed in
original units via inverse-transform, so this is correct, just documented.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Scalers:
    meteo_scaler: MinMaxScaler
    pv_scaler: MinMaxScaler
    sat_min: float
    sat_max: float

    def transform_meteo(self, x: np.ndarray) -> np.ndarray:
        return self.meteo_scaler.transform(x).astype("float32")

    def transform_pv(self, x: np.ndarray) -> np.ndarray:
        return self.pv_scaler.transform(x.reshape(-1, 1)).astype("float32").ravel()

    def transform_sat(self, x: np.ndarray) -> np.ndarray:
        rng = max(self.sat_max - self.sat_min, 1e-6)
        return ((x - self.sat_min) / rng).astype("float32")

    def inverse_pv(self, x: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled PV back to original units (for metrics)."""
        flat = np.asarray(x, dtype="float32").reshape(-1, 1)
        return self.pv_scaler.inverse_transform(flat).reshape(np.asarray(x).shape)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "Scalers":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def fit_scalers(meteo_train: np.ndarray, pv_train: np.ndarray, sat_train: np.ndarray) -> Scalers:
    """Fit all scalers on the TRAIN split only."""
    meteo_scaler = MinMaxScaler().fit(meteo_train)
    pv_scaler = MinMaxScaler().fit(pv_train.reshape(-1, 1))
    return Scalers(
        meteo_scaler=meteo_scaler,
        pv_scaler=pv_scaler,
        sat_min=float(sat_train.min()),
        sat_max=float(sat_train.max()),
    )
