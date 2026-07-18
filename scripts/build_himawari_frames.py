"""Build real Himawari-8/9 frames.h5 from the NOAA AWS open buckets.

Completes the acquisition the repo's download_himawari.py left as a TODO: for each
UTC timestep it downloads the B03 visible segment (era-dependent: R05 0.5 km ~17 MB
for 2020+, R20 2 km legacy), decompresses,
reads with satpy (reader='ahi_hsd'), crops to the chosen ROI, resamples to a fixed
HxW lat/lon grid, and stacks into an output frames.h5 with the layout the satellite
loader expects (frames [T,1,H,W] float32 + byte-string timestamps).

PARALLEL: a process pool runs the heavy per-frame work (download + bz2 decompress +
satpy resample) concurrently across CPU cores; the parent process is the sole h5
writer, so the store stays single-writer and consistent. Each worker downloads to a
per-timestep temp file and deletes it immediately, keeping peak disk low.

RESUMABLE: frames stream into the output h5 (resizable datasets, flushed every
checkpoint) and any timestamp already present is skipped on restart, so a multi-hour
full-year pull can be killed and re-launched without losing progress.

REGIONS
-------
Two built-in regions (select with --region):
  vietnam  BBox (102,8,110,24) daytime UTC 0-9   output data/himawari/frames.h5
  alice    BBox (131.9,-25.8,135.9,-21.8) daytime UTC 22-23,0-8 (UTC+9:30)
           output data/himawari_alice/frames.h5

Custom region:
  --bbox lon_min lat_min lon_max lat_max --hours 0,1,2 --out data/custom/frames.h5

Run (Vietnam smoke, same as before):
    python scripts/build_himawari_frames.py --start 2016-06-01 --end 2016-06-07

Run (Alice Springs smoke, 2020):
    python scripts/build_himawari_frames.py --region alice \
        --start 2020-06-01 --end 2020-06-07

Run (Alice Springs full 2020-2022, 8 workers):
    python scripts/build_himawari_frames.py --region alice \
        --start 2020-01-01 --end 2022-12-31 --workers 8
"""

from __future__ import annotations

import os

# Limit native BLAS/OpenMP threads BEFORE numpy import so N worker processes do not
# each spawn a full thread pool and oversubscribe the cores (set once, inherited on fork).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import bz2
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PREFIX = "AHI-L1b-FLDK"

# Himawari-8 ops ended 2022-12-13 ~05:00 UTC; Himawari-9 (same AHI sensor, same
# FLDK tile layout) took over the same day. Frames on/after the handover day come
# from the H9 bucket; already-stored H8 frames that morning are resume-skipped.
H9_CUTOVER = datetime(2022, 12, 13)


def bucket_for(day: datetime) -> tuple[str, str]:
    """(s3 bucket, satellite filename tag) for a given UTC day."""
    if day < H9_CUTOVER:
        return "noaa-himawari8", "H08"
    return "noaa-himawari9", "H09"

# Built-in region bounding boxes: (lon_min, lat_min, lon_max, lat_max).
VN_BBOX = (102.0, 8.0, 110.0, 24.0)
ALICE_BBOX = (131.9, -25.8, 135.9, -21.8)  # ±2° crop around Alice Springs 133.87/−23.76

# Alice Springs is UTC+9:30 → civil daytime ~06:30–19:30 local → UTC 21:00–10:00.
# Use UTC 22–23 (wrap-around midnight) + 0–8 to cover solar-active hours.
ALICE_HOURS = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8]

# Per-region defaults (bbox, daytime UTC hours, default output path).
# B03 segment/resolution differ by AHI archive era:
#   - 2016 Vietnam build used the 2 km full-disk product (R20, single segment S0101).
#   - 2020-2022 B03 on AWS is the native 0.5 km product (R05) split into 10 FLDK
#     latitude segments S0110..S1010. Alice Springs (-23.76 lat) falls entirely in
#     segment 8 (S0810) — verified by cropping the bbox to a 100%-finite frame.
REGION_DEFAULTS: dict[str, dict] = {
    "vietnam": {
        "bbox": VN_BBOX,
        "hours": list(range(0, 10)),  # UTC 0-9
        "out": "data/himawari/frames.h5",
        "area_id": "vn",
        "resolution": "R20",
        "segment": "S0101",
        "description": "Vietnam ROI",
    },
    "alice": {
        "bbox": ALICE_BBOX,
        "hours": ALICE_HOURS,
        "out": "data/himawari_alice/frames.h5",
        "area_id": "alice",
        "resolution": "R05",
        "segment": "S0810",
        "description": "Alice Springs ROI",
    },
}

CHECKPOINT_EVERY = 25  # flush h5 to disk every N new frames

# Per-worker globals (initialised once per process to avoid re-creating clients).
_S3 = None
_AREA = None
_RAW: Path | None = None


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def segment_for_lat(lat: float, n_segments: int = 10) -> str:
    """FLDK B03 R05 segment id (S0NN10) whose latitude band contains ``lat``.

    AHI full-disk is split top->bottom (north->south) into ``n_segments`` equal
    line strips. Project lat (at the sub-satellite longitude) to the geostationary
    y axis, map to a line, then to a 1-based segment. Used for custom bboxes.
    """
    from pyproj import Transformer

    geos = ("+proj=geos +h=35785831 +lon_0=140.7 +a=6378137 +b=6356752.3 +units=m +no_defs")
    y_top, y_bot = 5500000.0, -5500000.0  # full-disk y extent (m)
    _, y = Transformer.from_crs("EPSG:4326", geos, always_xy=True).transform(140.7, lat)
    frac = (y_top - y) / (y_top - y_bot)
    seg = min(n_segments, max(1, int(frac * n_segments) + 1))
    return f"S{seg:02d}{n_segments:02d}"


def seg_key(day: datetime, hour: int, minute: int, resolution: str, segment: str, sat: str) -> str:
    hm = f"{hour:02d}{minute:02d}"
    s = f"{day:%Y/%m/%d}/{hm}"
    name = f"HS_{sat}_{day:%Y%m%d}_{hm}_B03_FLDK_{resolution}_{segment}.DAT.bz2"
    return f"{PREFIX}/{s}/{name}"


def make_area(h: int, w: int, bbox: tuple[float, float, float, float], area_id: str = "roi"):
    """Build a pyresample AreaDefinition for the given lon/lat bounding box."""
    from pyresample.geometry import AreaDefinition

    return AreaDefinition(
        area_id, area_id, area_id,
        {"proj": "longlat", "datum": "WGS84"},
        w, h, bbox,
    )


def frame_from_dat(dat: Path, area) -> np.ndarray:
    from satpy import Scene

    scn = Scene(reader="ahi_hsd", filenames=[str(dat)])
    scn.load(["B03"])
    out = scn.resample(area, resampler="nearest", radius_of_influence=20000)
    return np.asarray(out["B03"].values, dtype="float32")


# --------------------------------------------------------------------------- #
# Worker side: download + decompress + resample one timestep.
# --------------------------------------------------------------------------- #
def _worker_init(
    size: int,
    raw_dir: str,
    bbox: tuple[float, float, float, float],
    area_id: str,
) -> None:
    global _S3, _AREA, _RAW
    import boto3
    import dask
    from botocore import UNSIGNED
    from botocore.config import Config

    dask.config.set(scheduler="synchronous")  # one dask thread per worker process
    _S3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "standard"}),
    )
    _AREA = make_area(size, size, bbox, area_id)
    _RAW = Path(raw_dir)
    _RAW.mkdir(parents=True, exist_ok=True)


def _process_timestep(task: tuple[str, str, str]):
    """task = (stamp, bucket, s3_key). Returns (stamp, frame[H,W] | None, reason)."""
    stamp, bucket, key = task
    bz = _RAW / Path(key).name
    try:
        _S3.download_file(bucket, key, str(bz))
    except Exception as exc:  # noqa: BLE001 (missing timestep on AWS -> skip)
        bz.unlink(missing_ok=True)
        return stamp, None, f"download:{type(exc).__name__}"
    dat = _RAW / bz.stem
    try:
        with bz2.open(bz, "rb") as fi, open(dat, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        arr = frame_from_dat(dat, _AREA)
    except Exception as exc:  # noqa: BLE001 (corrupt segment -> skip)
        return stamp, None, f"decode:{type(exc).__name__}"
    finally:
        dat.unlink(missing_ok=True)
        bz.unlink(missing_ok=True)
    return stamp, np.nan_to_num(arr, nan=0.0).astype("float32"), "ok"


# --------------------------------------------------------------------------- #
# Parent side: resumable single-writer h5 store.
# --------------------------------------------------------------------------- #
def open_store(path: Path, size: int):
    """Open (or create) the resizable h5 store; return (file, done_timestamps)."""
    if path.exists():
        f = h5py.File(path, "a")
        if "frames" in f and "timestamps" in f:
            h, w = f["frames"].shape[2], f["frames"].shape[3]
            if (h, w) != (size, size):
                f.close()
                raise SystemExit(
                    f"{path} has grid {h}x{w} != requested {size}x{size}. "
                    "Move it aside or pass --size to match before resuming."
                )
            done = {t.decode() if isinstance(t, bytes) else t for t in f["timestamps"][:]}
            print(f"  resuming {path}: {len(done)} frames already stored")
            return f, done
        f.close()
        path.unlink()

    f = h5py.File(path, "w")
    f.create_dataset(
        "frames", shape=(0, 1, size, size), maxshape=(None, 1, size, size),
        dtype="float32", chunks=(1, 1, size, size),
    )
    f.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype="S20")
    return f, set()


def append_frame(f, arr: np.ndarray, stamp: str) -> None:
    n = f["frames"].shape[0]
    f["frames"].resize(n + 1, axis=0)
    f["frames"][n] = arr[None, :, :]
    f["timestamps"].resize(n + 1, axis=0)
    f["timestamps"][n] = stamp.encode()


def build_tasks(start, end, hours, step_min, done, resolution, segment):
    """All (stamp, bucket, key) timesteps in range not already stored."""
    minutes = list(range(0, 60, step_min))
    tasks = []
    for day in daterange(start, end):
        bucket, sat = bucket_for(day)
        for hour in hours:
            for minute in minutes:
                stamp = f"{day:%Y-%m-%d}T{hour:02d}:{minute:02d}:00Z"
                if stamp in done:
                    continue
                tasks.append((stamp, bucket, seg_key(day, hour, minute, resolution, segment, sat)))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Region selection: named shortcut OR fully custom bbox + hours.
    region_group = ap.add_mutually_exclusive_group()
    region_group.add_argument(
        "--region", choices=list(REGION_DEFAULTS), default=None,
        help="Named ROI preset (vietnam|alice). Implies default bbox, hours, and --out.",
    )
    region_group.add_argument(
        "--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Custom bounding box. Requires --hours and --out.",
    )

    ap.add_argument(
        "--hours",
        help=(
            "Comma-separated UTC hours to download (e.g. 22,23,0,1,2,3,4,5,6,7,8). "
            "Required when --bbox is used. Overrides region default when --region is used."
        ),
    )
    ap.add_argument("--start", default="2016-06-01", help="Start date YYYY-MM-DD.")
    ap.add_argument("--end", default="2016-06-07", help="End date YYYY-MM-DD.")
    ap.add_argument("--step-min", type=int, default=10, help="Intra-hour cadence in minutes.")
    ap.add_argument("--size", type=int, default=64, help="Output grid size (H=W, default 64).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel download/resample workers.")
    ap.add_argument("--out", type=Path, default=None, help="Output h5 path (overrides region default).")
    ap.add_argument("--resolution", default=None,
                    help="B03 resolution tag (R05=0.5km native 2020+, R20=2km legacy). "
                         "Defaults: region preset, or R05 for custom --bbox.")
    ap.add_argument("--segment", default=None,
                    help="FLDK segment tag (e.g. S0810). Defaults: region preset, or "
                         "auto-computed from the custom bbox centre latitude.")
    args = ap.parse_args()

    # Resolve region / bbox / hours / out / resolution / segment.
    if args.bbox is not None:
        # Custom bbox path.
        if args.hours is None:
            ap.error("--bbox requires --hours (comma-separated UTC hours, e.g. 0,1,2,3)")
        if args.out is None:
            ap.error("--bbox requires --out (output h5 path)")
        bbox = tuple(args.bbox)
        hours = [int(h) for h in args.hours.split(",")]
        out_path = args.out
        area_id = "custom"
        resolution = args.resolution or "R05"
        segment = args.segment or segment_for_lat((bbox[1] + bbox[3]) / 2.0)
    else:
        # Named region path (default: vietnam for backward compat).
        region_name = args.region if args.region is not None else "vietnam"
        region = REGION_DEFAULTS[region_name]
        bbox = region["bbox"]
        hours = [int(h) for h in args.hours.split(",")] if args.hours else region["hours"]
        out_path = args.out if args.out is not None else Path(region["out"])
        area_id = region["area_id"]
        resolution = args.resolution or region["resolution"]
        segment = args.segment or region["segment"]

    raw = out_path.parent / "raw"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"  region bbox={bbox}  hours={hours}  out={out_path}  "
        f"size={args.size}x{args.size}  workers={args.workers}  "
        f"B03 {resolution}/{segment}",
        flush=True,
    )

    store, done = open_store(out_path, args.size)
    tasks = build_tasks(args.start, args.end, hours, args.step_min, done, resolution, segment)
    print(f"  {len(tasks)} timesteps to fetch with {args.workers} workers", flush=True)

    n_new = n_miss = 0
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.size, str(raw), bbox, area_id),
        ) as ex:
            futures = [ex.submit(_process_timestep, t) for t in tasks]
            for fut in as_completed(futures):
                stamp, arr, reason = fut.result()
                if arr is None:
                    n_miss += 1
                    continue
                append_frame(store, arr, stamp)
                n_new += 1
                if n_new % CHECKPOINT_EVERY == 0:
                    store.flush()
                    total = store["frames"].shape[0]
                    print(
                        f"  +{n_new} new (total {total}, missing {n_miss}) "
                        f"last={stamp} mean={float(np.nanmean(arr)):.1f}",
                        flush=True,
                    )
    finally:
        store.flush()
        total = store["frames"].shape[0]
        store.close()

    print(
        f"DONE {out_path}  total_frames={total}  new={n_new}  missing={n_miss}  "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
