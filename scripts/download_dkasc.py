"""Download DKASC Alice Springs PV + meteo data and write a canonical clean CSV.

SOURCE ASSUMPTION
-----------------
Data is fetched from the Desert Knowledge Australia Solar Centre (DKASC) public
data portal at https://dkasc.com.au.  The portal exposes a CSV download endpoint:

    POST https://dkasc.com.au/getData
    Content-Type: application/x-www-form-urlencoded
    Body: SystemID=<id>&StartDate=YYYY-MM-DD&EndDate=YYYY-MM-DD

Each array (SystemID) exports one CSV per request with a header row containing
column names.  The typical columns produced by a DKASC array export are:

    Timestamp                   -> timestamp (ISO8601 UTC after tz strip)
    Active Power - [kW]         -> Pac
    Irradiance - [W/m²]         -> GHI
    Ambient Temperature - [°C]  -> Tamb
    Relative Humidity - [%]     -> RH
    Wind Speed - [m/s]          -> WS

Column names vary slightly by array firmware version; the mapping below handles the
common variants.  If a future firmware revision changes them, update COLUMN_MAP.

ARRAY CHOICE
------------
Default array: 26 (a well-maintained, gap-light Alice Springs array installed 2010
with a large PV string and all five co-located meteo sensors).  Adjust --array-id
if preferred.

RESUMABLE STRATEGY
------------------
Downloads are split into monthly chunks.  A chunk is skipped if its rows are
already present in the output CSV (last written timestamp compared against the
chunk's date range).  The manifest records progress so a re-run after a Colab
quota cutoff continues where it left off.

Run (full 2020-2022, default array):
    python scripts/download_dkasc.py

Run (custom array, date range, output dir):
    python scripts/download_dkasc.py --array-id 26 \
        --start 2020-01-01 --end 2022-12-31 --out data/dkasc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# DKASC data portal base URL.
DKASC_ENDPOINT = "https://dkasc.com.au/getData"

# Alice Springs site coordinates.
ALICE_LAT = -23.762
ALICE_LON = 133.874

# Mapping from raw DKASC column name fragments to canonical names.
# Keys are lowercase substrings; matching is done via `str.lower().find(key)`.
COLUMN_MAP: list[tuple[str, str]] = [
    ("active power",    "Pac"),   # PV active power kW
    ("irradiance",      "GHI"),   # Global horizontal irradiance W/m²
    ("ambient temp",    "Tamb"),  # Ambient temperature °C
    ("relative humid",  "RH"),    # Relative humidity %
    ("wind speed",      "WS"),    # Wind speed m/s
]

# Canonical output columns (must match loaders.py:load_dkasc).
CANONICAL_COLS = ["timestamp", "Pac", "GHI", "Tamb", "RH", "WS"]

# Retry parameters for transient HTTP errors.
_MAX_RETRIES = 5
_RETRY_BACKOFF = 2.0  # seconds, doubles on each retry


# --------------------------------------------------------------------------- #
# Column normalisation
# --------------------------------------------------------------------------- #

def _find_col(df_cols: list[str], fragment: str) -> str | None:
    """Return the first column whose lowercased name contains `fragment`."""
    for col in df_cols:
        if fragment in col.lower():
            return col
    return None


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw DKASC columns to canonical names using COLUMN_MAP.

    Raises KeyError if any canonical column cannot be matched.
    """
    rename = {}
    missing = []
    for fragment, canonical in COLUMN_MAP:
        src = _find_col(list(df.columns), fragment)
        if src is None:
            missing.append(canonical)
        else:
            rename[src] = canonical
    if missing:
        raise KeyError(
            f"Could not find DKASC source column(s) for {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.rename(columns=rename)


# --------------------------------------------------------------------------- #
# Core transformation (testable, no I/O)
# --------------------------------------------------------------------------- #

def to_clean_csv(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw DKASC export DataFrame into the canonical clean schema.

    Parameters
    ----------
    raw_df:
        DataFrame as returned by parse_chunk_csv().  Must contain a timestamp
        column and the five measurement columns (or their DKASC aliases).

    Returns
    -------
    DataFrame with columns [timestamp, Pac, GHI, Tamb, RH, WS]:
        - timestamp: ISO8601 UTC strings, 5-min cadence, no timezone offset.
        - Pac: PV active power kW (float).
        - GHI, Tamb, RH, WS: meteo values (float).

    Notes
    -----
    - Duplicate timestamps are dropped (keep first).
    - Negative Pac values are clipped to 0 (nighttime sensor noise).
    - Rows where all five measurement columns are NaN are dropped.
    """
    df = raw_df.copy()

    # Identify timestamp column (first column or named "Timestamp").
    ts_col = None
    for col in df.columns:
        if "timestamp" in col.lower() or col.lower() == "date":
            ts_col = col
            break
    if ts_col is None:
        ts_col = df.columns[0]

    # Parse timestamp; DKASC exports in local Alice Springs time (UTC+9:30).
    # Convert to UTC by subtracting 9h30m.
    ts = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    ts_utc = ts - pd.Timedelta(hours=9, minutes=30)
    df["timestamp"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Rename measurement columns to canonical names.
    df = _map_columns(df)

    # Coerce numeric columns.
    for col in ["Pac", "GHI", "Tamb", "RH", "WS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip negative PV power to zero.
    df["Pac"] = df["Pac"].clip(lower=0.0)

    # Keep only canonical columns.
    df = df[CANONICAL_COLS].copy()

    # Drop rows where all measurement values are NaN (gap rows).
    measure_cols = ["Pac", "GHI", "Tamb", "RH", "WS"]
    df = df.dropna(subset=measure_cols, how="all")

    # Drop duplicates on timestamp.
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# HTTP fetch helpers
# --------------------------------------------------------------------------- #

def _post_with_retry(session: requests.Session, payload: dict) -> str:
    """POST to DKASC endpoint with exponential-backoff retries."""
    backoff = _RETRY_BACKOFF
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.post(DKASC_ENDPOINT, data=payload, timeout=120)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            print(f"  [retry {attempt + 1}/{_MAX_RETRIES}] {exc} — waiting {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Download failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc


def parse_chunk_csv(text: str) -> pd.DataFrame | None:
    """Parse a single DKASC CSV response text into a DataFrame.

    Returns None if the response is empty or contains no data rows.
    """
    import io
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Could not parse CSV chunk: {exc}")
        return None
    if df.empty:
        return None
    return df


# --------------------------------------------------------------------------- #
# Monthly chunk iteration (for resumable downloads)
# --------------------------------------------------------------------------- #

def _month_ranges(start: str, end: str):
    """Yield (month_start, month_end) date strings covering [start, end]."""
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    year, month = d0.year, d0.month
    while True:
        ms = datetime(year, month, 1)
        # Last day of month.
        if month == 12:
            me = datetime(year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            me = datetime(year, month + 1, 1) - pd.Timedelta(days=1)
        me = min(me, d1)
        yield ms.strftime("%Y-%m-%d"), me.strftime("%Y-%m-%d")
        if me >= d1:
            break
        month += 1
        if month > 12:
            month = 1
            year += 1


def _already_fetched(out_csv: Path, month_start: str, month_end: str) -> bool:
    """Return True if the output CSV already covers the full month."""
    if not out_csv.exists():
        return False
    try:
        df = pd.read_csv(out_csv, usecols=["timestamp"], nrows=1)
        # Quick check: read last timestamp from file tail.
        import subprocess
        result = subprocess.run(
            ["tail", "-1", str(out_csv)], capture_output=True, text=True
        )
        last_line = result.stdout.strip()
        if not last_line or last_line.startswith("timestamp"):
            return False
        last_ts = last_line.split(",")[0].strip('"')
        last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%SZ")
        month_end_dt = datetime.strptime(month_end, "%Y-%m-%d")
        return last_dt.date() >= month_end_dt.date()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _write_manifest(
    out_dir: Path,
    array_id: int,
    start: str,
    end: str,
    out_csv: Path,
    row_count: int,
) -> None:
    manifest = {
        "array_id": array_id,
        "lat": ALICE_LAT,
        "lon": ALICE_LON,
        "date_range": {"start": start, "end": end},
        "source_url": DKASC_ENDPOINT,
        "output_csv": str(out_csv),
        "row_count": row_count,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {path}")


# --------------------------------------------------------------------------- #
# Main download driver
# --------------------------------------------------------------------------- #

def download(array_id: int, start: str, end: str, out_dir: Path) -> None:
    """Download a DKASC array for the given date range and write clean CSV.

    The download is split into monthly chunks.  Already-completed months (where
    the output CSV already contains rows up to the month end) are skipped so a
    re-run after a Colab quota cutoff resumes cleanly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{array_id}_2020_2022.csv"

    session = requests.Session()
    session.headers.update({"User-Agent": "tkg-solar-thesis/1.0 (research)"})

    all_frames: list[pd.DataFrame] = []

    # Load existing data so we can append only new rows.
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        all_frames.append(existing)
        last_existing = pd.to_datetime(existing["timestamp"]).max()
        print(f"  found existing {out_csv} ({len(existing)} rows, last={last_existing})")
    else:
        last_existing = None

    month_ranges = list(_month_ranges(start, end))
    print(f"  {len(month_ranges)} monthly chunks to check")

    for chunk_start, chunk_end in month_ranges:
        # Skip if this month is already fully covered.
        if last_existing is not None:
            chunk_end_dt = datetime.strptime(chunk_end, "%Y-%m-%d")
            if last_existing.date() >= chunk_end_dt.date():
                print(f"  skip {chunk_start}..{chunk_end} (already in CSV)")
                continue

        print(f"  fetching {chunk_start} .. {chunk_end} (array {array_id}) ...", flush=True)
        payload = {
            "SystemID": str(array_id),
            "StartDate": chunk_start,
            "EndDate": chunk_end,
        }
        try:
            raw_text = _post_with_retry(session, payload)
        except RuntimeError as exc:
            print(f"  [error] {exc} — skipping chunk", file=sys.stderr)
            continue

        raw_df = parse_chunk_csv(raw_text)
        if raw_df is None:
            print(f"  [warn] empty response for {chunk_start}..{chunk_end}")
            continue

        try:
            clean = to_clean_csv(raw_df)
        except KeyError as exc:
            print(f"  [error] column mapping failed: {exc}", file=sys.stderr)
            continue

        all_frames.append(clean)
        print(f"  +{len(clean)} rows")

        # Write incrementally so progress survives a crash.
        combined = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="first")
            .sort_values("timestamp")
        )
        combined.to_csv(out_csv, index=False)
        last_existing = pd.to_datetime(combined["timestamp"]).max()

        # Brief pause to be polite to the public portal.
        time.sleep(1.0)

    # Final consolidated write.
    if all_frames:
        combined = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="first")
            .sort_values("timestamp")
        )
        combined.to_csv(out_csv, index=False)
        row_count = len(combined)
        print(f"  wrote {out_csv} ({row_count} rows)")
        _write_manifest(out_dir, array_id, start, end, out_csv, row_count)
    else:
        print("[warn] No data downloaded; manifest not written.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download DKASC Alice Springs PV+meteo and write a canonical clean CSV."
    )
    ap.add_argument(
        "--array-id", type=int, default=26,
        help="DKASC SystemID (default: 26, a well-maintained Alice Springs array).",
    )
    ap.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD.")
    ap.add_argument("--end", default="2022-12-31", help="End date YYYY-MM-DD.")
    ap.add_argument(
        "--out", type=Path, default=Path("data/dkasc"),
        help="Output directory (default: data/dkasc).",
    )
    args = ap.parse_args()

    print(f"DKASC download: array={args.array_id} {args.start}..{args.end} -> {args.out}")
    try:
        download(args.array_id, args.start, args.end, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
