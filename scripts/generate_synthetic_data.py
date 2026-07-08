"""Generate small synthetic DKASC + Himawari files in the REAL on-disk formats.

Purpose: exercise the actual data loaders end-to-end without the multi-GB real
downloads (DKASC array CSV + 3 years of Himawari frames). The synthetic series
carry a diurnal solar shape so night filtering / MAPE masking behave realistically.

This is a TEST FIXTURE generator, not part of the reproduction's data path.
Run:  python scripts/generate_synthetic_data.py [--days 30 --out data]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Allow running as a plain script (python scripts/x.py): put project root on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.shapes import BASE_CADENCE_MIN, SAT_CHANNELS, SAT_IMG_SIZE

# DKASC-era start so synthetic data is plausible for fixed-date splits too.
START = datetime(2020, 6, 1, tzinfo=timezone.utc)
CADENCE_MIN = BASE_CADENCE_MIN  # 5-min DKASC cadence


def _diurnal(minutes_since_start: np.ndarray) -> np.ndarray:
    """Daily solar bell curve in [0, 1], zero at night (sun up ~06:00-18:00)."""
    minutes_of_day = minutes_since_start % (24 * 60)
    frac = (minutes_of_day - 6 * 60) / (12 * 60)
    bell = np.sin(np.clip(frac, 0, 1) * np.pi)
    return np.clip(bell, 0.0, 1.0)


def _timestamps(n: int, cadence_min: int) -> pd.DatetimeIndex:
    return pd.date_range(START, periods=n, freq=f"{cadence_min}min", tz="UTC")


def gen_dkasc(out: Path, days: int, rng: np.random.Generator, array_id: str = "synthetic_array") -> None:
    """DKASC: clean CSV ``timestamp, Pac, GHI, Tamb, RH, WS`` at 5-min cadence."""
    n = days * 24 * (60 // CADENCE_MIN)
    ts = _timestamps(n, CADENCE_MIN)
    minutes = np.arange(n) * CADENCE_MIN
    bell = _diurnal(minutes)
    cloud = 0.7 + 0.6 * rng.random(n)  # multiplicative cloud factor
    df = pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Pac": np.round(5.0 * bell * cloud, 4),                       # kW (~5 kW array)
            "GHI": np.round(1000.0 * bell * cloud, 2),                    # W/m^2 (0 at night)
            "Tamb": np.round(20 + 12 * bell + rng.normal(0, 1, n), 2),    # deg C
            "RH": np.clip(np.round(60 - 25 * bell + rng.normal(0, 3, n), 2), 0, 100),
            "WS": np.round(np.abs(3 + rng.normal(0, 1.5, n)), 2),         # m/s
        }
    )
    (out / "dkasc").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dkasc" / f"{array_id}_2020_2022.csv", index=False)


def gen_himawari_alice(out: Path, days: int, rng: np.random.Generator) -> None:
    """Himawari: h5 of frames [T, C, H, W] + timestamps, 5-min visible band."""
    n = days * 24 * (60 // CADENCE_MIN)
    ts = _timestamps(n, CADENCE_MIN)
    minutes = np.arange(n) * CADENCE_MIN
    bell = _diurnal(minutes)
    h = w = SAT_IMG_SIZE
    base = bell[:, None, None, None] * np.ones((n, SAT_CHANNELS, h, w), dtype="float32")
    clouds = rng.random((n, SAT_CHANNELS, h, w)).astype("float32") * 0.3
    frames = np.clip(base * 0.7 + clouds, 0.0, 1.0).astype("float32")
    (out / "himawari_alice").mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "himawari_alice" / "frames.h5", "w") as f:
        f.create_dataset("frames", data=frames, compression="gzip")
        f.create_dataset(
            "timestamps",
            data=np.array(ts.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(), dtype="S20"),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30, help="days of co-located data")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    gen_dkasc(args.out, args.days, rng)
    gen_himawari_alice(args.out, args.days, rng)
    print(f"Synthetic DKASC data written to {args.out}/ ({args.days} days, start {START.date()}).")


if __name__ == "__main__":
    main()
