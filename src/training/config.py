"""Hyperparameter config (paper Table 5 defaults) with YAML load + resolved save.

Defaults reproduce the paper config; smoke runs override via YAML (tiny model,
few epochs) so `python main.py` finishes in seconds on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from src.common.shapes import EMBED_DIM, SAT_IMG_SIZE


@dataclass
class Config:
    # --- run / reproduction ---
    seed: int = 42
    device: str = "cpu"               # "cuda" on a GPU box
    model: str = "full"               # "lstm" (baseline) | "full" (TKGSolarModel)

    # --- optimisation (paper Table 5) ---
    lr: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    early_stop_patience: int = 20
    grad_clip: float = 1.0

    # --- architecture (paper Table 5) ---
    embed_dim: int = EMBED_DIM        # 128, locked
    n_heads: int = 4
    dropout: float = 0.2

    # --- data / windowing ---
    k: int = 12                       # input window length (steps)
    img_size: int = SAT_IMG_SIZE
    min_steps: int = 200
    num_workers: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15

    # --- satellite encoder ---
    sat_backbone: str = "resnet18"    # lightweight default; "vit_base_patch16_224" for paper
    pretrained_backbone: bool = False  # offline-friendly default; True downloads weights
    freeze_backbone: bool = False

    # --- graph ---
    n_graph_nodes: int = 8            # one node per entity type (MVP)

    # --- advanced loss ---
    use_advanced_loss: bool = False
    lambda_prob: float = 0.0
    lambda_physics: float = 0.0

    # --- metrics ---
    mape_min_value: float = 1e-3      # daytime MAPE floor (original units)

    # --- paths ---
    opsd_path: str = "data/opsd/time_series_15min_singleindex.csv"
    nsrdb_path: str = "data/nsrdb/vietnam_2016.h5"
    himawari_dir: str = "data/himawari"
    cache_dir: str = "data/cache"
    checkpoint_dir: str = "checkpoints"
    scaler_out: str = "data/cache/scaler.pkl"

    extras: dict = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in Config().__dict__}
        cfg_kwargs = {k: v for k, v in raw.items() if k in known}
        extras = {k: v for k, v in raw.items() if k not in known}
        cfg = Config(**cfg_kwargs)
        cfg.extras = extras
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
