"""Shared pytest fixtures: a tiny synthetic dataset + a loaded M1 pipeline.

Synthetic data is generated into a temp dir so tests never depend on real
downloads and never pollute the repo's data/ folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.generate_synthetic_data import gen_himawari, gen_nsrdb, gen_opsd
from src.common.seeding import seed_everything


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    seed_everything(42)
    out = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    days = 14  # enough for 70/15/15 split + windows at 10-min cadence
    gen_opsd(out, days, rng)
    gen_nsrdb(out, days, rng)
    gen_himawari(out, days, rng)
    return out


@pytest.fixture(scope="session")
def pipeline_splits(synthetic_data_dir):
    from src.data_pipeline import DataPipeline

    seed_everything(42)
    return DataPipeline.load(
        str(synthetic_data_dir / "opsd" / "time_series_15min_singleindex.csv"),
        str(synthetic_data_dir / "nsrdb" / "vietnam_2016.h5"),
        str(synthetic_data_dir / "himawari"),
        k=12,
        batch_size=16,
        img_size=64,
        min_steps=200,
        use_cache=False,
    )
