"""Build real Himawari-8 frames.h5 from the NOAA AWS open bucket.

Completes the acquisition the repo's download_himawari.py left as a TODO: for each
UTC timestep it downloads the B03 visible segment (R20 = 2 km, ~22 MB), decompresses,
reads with satpy (reader='ahi_hsd'), crops to the Vietnam ROI, resamples to a fixed
HxW lat/lon grid, and stacks into data/himawari/frames.h5 with the layout the M1
satellite loader expects (frames [T,1,H,W] float32 + byte-string timestamps).

PARALLEL: a process pool runs the heavy per-frame work (download + bz2 decompress +
satpy resample) concurrently across CPU cores; the parent process is the sole h5
writer, so the store stays single-writer and consistent. Each worker downloads to a
per-timestep temp file and deletes it immediately, keeping peak disk low.

RESUMABLE: frames stream into the output h5 (resizable datasets, flushed every
checkpoint) and any timestamp already present is skipped on restart, so a multi-hour
full-year pull can be killed and re-launched without losing progress.

Run (full daytime 2016, ~21.9k frames, ~480 GB transient download, 8 workers):
    python scripts/build_himawari_frames.py --start 2016-01-01 --end 2016-12-31 --workers 8

Run (default smoke = 2016-06-01..06-07, daytime UTC 00..09):
    python scripts/build_himawari_frames.py --start 2016-06-01 --end 2016-06-07
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

BUCKET = "noaa-himawari8"
PREFIX = "AHI-L1b-FLDK"
VN_BBOX = (102.0, 8.0, 110.0, 24.0)  # lon_min, lat_min, lon_max, lat_max
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


def seg_key(day: datetime, hour: int, minute: int = 0) -> str:
    hm = f"{hour:02d}{minute:02d}"
    s = f"{day:%Y/%m/%d}/{hm}"
    name = f"HS_H08_{day:%Y%m%d}_{hm}_B03_FLDK_R20_S0101.DAT.bz2"
    return f"{PREFIX}/{s}/{name}"


def make_area(h: int, w: int):
    from pyresample.geometry import AreaDefinition

    return AreaDefinition(
        "vn", "Vietnam ROI", "vn",
        {"proj": "longlat", "datum": "WGS84"},
        w, h, VN_BBOX,
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
def _worker_init(size: int, raw_dir: str) -> None:
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
    _AREA = make_area(size, size)
    _RAW = Path(raw_dir)
    _RAW.mkdir(parents=True, exist_ok=True)


def _process_timestep(task: tuple[str, str]):
    """task = (stamp, s3_key). Returns (stamp, frame[H,W] | None, reason)."""
    stamp, key = task
    bz = _RAW / Path(key).name
    try:
        _S3.download_file(BUCKET, key, str(bz))
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


def build_tasks(start, end, hours, step_min, done):
    """All (stamp, key) timesteps in range not already stored."""
    minutes = list(range(0, 60, step_min))
    tasks = []
    for day in daterange(start, end):
        for hour in hours:
            for minute in minutes:
                stamp = f"{day:%Y-%m-%d}T{hour:02d}:{minute:02d}:00Z"
                if stamp in done:
                    continue
                tasks.append((stamp, seg_key(day, hour, minute)))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-06-01")
    ap.add_argument("--end", default="2016-06-07")
    ap.add_argument("--hours", default="0,1,2,3,4,5,6,7,8,9", help="UTC hours (VN daytime)")
    ap.add_argument("--step-min", type=int, default=10, help="intra-hour cadence (min)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="parallel download/resample workers")
    ap.add_argument("--out", type=Path, default=Path("data/himawari/frames.h5"))
    args = ap.parse_args()

    hours = [int(h) for h in args.hours.split(",")]
    raw = args.out.parent / "raw"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    store, done = open_store(args.out, args.size)
    tasks = build_tasks(args.start, args.end, hours, args.step_min, done)
    print(f"  {len(tasks)} timesteps to fetch with {args.workers} workers", flush=True)

    n_new = n_miss = 0
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
            initargs=(args.size, str(raw)),
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
        f"DONE {args.out}  total_frames={total}  new={n_new}  missing={n_miss}  "
        f"({args.out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
