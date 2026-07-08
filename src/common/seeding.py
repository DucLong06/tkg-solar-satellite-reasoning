"""Global determinism utility — for a reproduction thesis, determinism IS the product.

Every entrypoint (main.py, tests, baseline scripts) calls ``seed_everything`` first
so runs are bit-reproducible on the same hardware.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python / numpy / torch (CPU + CUDA) and set deterministic flags.

    Returns the seed so callers can log it into the resolved run config.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Import torch lazily so non-torch utilities (e.g. data download) don't need it.
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS reproducibility for matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed
