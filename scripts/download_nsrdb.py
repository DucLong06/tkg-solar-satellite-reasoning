"""Download NSRDB meteo + irradiance for a Vietnam ROI via the NREL PSM v3 API.

Uses the Himawari-based PSM v3 endpoint (the legacy `vietnam` endpoint is
deprecated). Requires a free API key:  https://developer.nrel.gov/signup/
Set it in the environment:  export NREL_API_KEY=...

The point-download API returns CSV; this script converts it to the h5 layout the
M1 loader expects (one dataset per meteo var + a time_index).

Run:
    NREL_API_KEY=xxx python scripts/download_nsrdb.py \
        --lat 10.82 --lon 106.63 --year 2016 [--out data/nsrdb/vietnam_2016.h5]

NOTE: bulk/multi-point NSRDB requests are delivered asynchronously by email
(24-48 h). This single-point CSV path is the quick-start; for the full ROI grid
submit a bulk request via the NSRDB Viewer and drop the files in data/nsrdb/.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Allow running as a plain script (python scripts/x.py): put project root on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.shapes import METEO_FEATURES

# Correct host is developer.nrel.gov (NOT nlr.gov — common typo).
PSM_V3_ENDPOINT = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.csv"

# Map NSRDB CSV column names -> our fixed meteo feature names.
NSRDB_COLUMN_MAP = {
    "GHI": "ghi",
    "DNI": "dni",
    "DHI": "dhi",
    "Temperature": "air_temperature",
    "Relative Humidity": "relative_humidity",
    "Wind Speed": "wind_speed",
    "Pressure": "surface_pressure",
}


def fetch_csv(lat: float, lon: float, year: int, api_key: str) -> str:
    import requests  # optional dep (extra: data)

    attrs = "ghi,dni,dhi,air_temperature,relative_humidity,wind_speed,surface_pressure"
    params = {
        "api_key": api_key,
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "interval": "10",
        "attributes": attrs,
        "utc": "true",
        "email": os.environ.get("NREL_EMAIL", "anonymous@example.com"),
        "full_name": "tkg-repro",
        "reason": "research",
        "affiliation": "thesis",
    }
    resp = requests.get(PSM_V3_ENDPOINT, params=params, timeout=120)
    resp.raise_for_status()
    return resp.text


def csv_to_h5(csv_text: str, dest: Path) -> None:
    # NSRDB CSV has 2 metadata header rows before the data table.
    df = pd.read_csv(io.StringIO(csv_text), skiprows=2)
    ts = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as f:
        f.create_dataset(
            "time_index",
            data=np.array(ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(), dtype="S20"),
        )
        for src_col, name in NSRDB_COLUMN_MAP.items():
            if src_col not in df.columns:
                raise KeyError(f"NSRDB CSV missing column {src_col!r}")
            f.create_dataset(name, data=df[src_col].to_numpy(dtype="float32"))
    assert set(NSRDB_COLUMN_MAP.values()) == set(METEO_FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat", type=float, default=10.82, help="Vietnam ROI latitude")
    ap.add_argument("--lon", type=float, default=106.63, help="Vietnam ROI longitude")
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--out", type=Path, default=Path("data/nsrdb/vietnam_2016.h5"))
    args = ap.parse_args()

    api_key = os.environ.get("NREL_API_KEY")
    if not api_key:
        print(
            "NREL_API_KEY not set. Get a free key at https://developer.nrel.gov/signup/\n"
            "then: export NREL_API_KEY=... and re-run.\n"
            "(For a runnable pipeline without a key, use scripts/generate_synthetic_data.py)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Requesting NSRDB PSM v3 @ ({args.lat},{args.lon}) {args.year} ...")
    try:
        csv_text = fetch_csv(args.lat, args.lon, args.year, api_key)
        csv_to_h5(csv_text, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
