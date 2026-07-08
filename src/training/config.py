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
    weight_decay: float = 0.0          # AdamW L2 decay; 0.0 == plain Adam behaviour
    lr_scheduler: str = "none"         # "none" | "plateau" | "cosine"
    lr_factor: float = 0.5             # plateau: LR *= factor on a val stall
    lr_patience: int = 5               # plateau: epochs to wait (< early_stop_patience)

    # --- architecture (paper Table 5) ---
    embed_dim: int = EMBED_DIM        # 128, locked
    n_heads: int = 4
    dropout: float = 0.2

    # --- data / windowing ---
    k: int = 12                       # input window length (steps)
    img_size: int = SAT_IMG_SIZE
    min_steps: int = 200
    num_workers: int = 0
    cadence_min: int = 5              # DKASC 5-min grid
    night_ghi_thresh: float = 5.0     # drop samples with GHI < this (W/m^2)
    # Fixed-date split (DKASC reproduction); leave null to use fractional split.
    train_end: str | None = None      # e.g. "2021-10-01"
    val_end: str | None = None        # e.g. "2022-01-01"
    train_frac: float = 0.70
    val_frac: float = 0.15

    # --- satellite encoder ---
    sat_backbone: str = "resnet18"    # lightweight default; "vit_base_patch16_224" for paper
    pretrained_backbone: bool = False  # offline-friendly default; True downloads weights
    freeze_backbone: bool = False

    # --- GPU utilization (CUDA only; CPU keeps fp32, accum=1) ---
    precision: str = "fp32"           # "fp32" | "bf16" | "fp16" | "auto" (pick on cuda)
    grad_accum_steps: int = 1         # effective batch = batch_size * grad_accum_steps
    auto_batch: bool = False          # on cuda, size batch/accum from detected VRAM
    channels_last: bool = False       # channels_last memory format for conv backbones
    compile: bool = False             # torch.compile(model) (guarded, eager fallback)

    # --- graph ---
    n_graph_nodes: int = 8            # one node per entity type (MVP)

    # --- ablation arms (proposed model) ---
    use_sat: bool = True              # satellite branch
    use_meteo: bool = True            # meteo branch
    use_graph: bool = True            # graph branch

    # --- advanced loss ---
    use_advanced_loss: bool = False
    lambda_prob: float = 0.0
    lambda_physics: float = 0.0

    # --- metrics ---
    mape_min_value: float = 1e-3      # daytime MAPE floor (original units)

    # --- paths ---
    dkasc_csv: str = "data/dkasc/synthetic_array_2020_2022.csv"
    himawari_dir: str = "data/himawari_alice"
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
