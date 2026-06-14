"""Single source of truth for the tensor-shape / dimension contract.

Every module imports these constants instead of hard-coding dims, so the
data pipeline -> encoders -> fusion cannot silently drift (the integration
landmine an all-128-dim contract prevents). All module embeddings are
128-dim per the paper (Table 5); fusion concatenates 3 x 128 = 384.

Naming note: the windowing axis is called ``k`` in the data pipeline and ``T``
inside the encoders. They are THE SAME axis (sequence length). We standardise on
``SEQ_LEN`` here to reconcile the two.
"""

from __future__ import annotations

# --- Locked embedding contract (paper Table 5) ------------------------------
EMBED_DIM: int = 128          # output dim of EVERY modality encoder
FUSION_DIM: int = EMBED_DIM * 3  # concat(F_sat, H_met, H_graph) = 384
N_HORIZONS: int = 3           # forecast at 10 / 30 / 60 minutes

# Horizon labels in minutes. DKASC cadence = 5 min -> steps ahead = 2, 6, 12.
HORIZON_MINUTES: tuple[int, int, int] = (10, 30, 60)
BASE_CADENCE_MIN: int = 5
HORIZON_STEPS: tuple[int, ...] = tuple(h // BASE_CADENCE_MIN for h in HORIZON_MINUTES)

# --- Meteo / PV feature layout ----------------------------------------------
# Order is fixed and shared by the loader, scaler, and encoders.
# DKASC Alice Springs co-located meteo: GHI, ambient temp, humidity, wind speed.
# (The paper drops DNI/DHI/surface-pressure; DKASC standard download has no DNI/DHI.)
METEO_FEATURES: tuple[str, ...] = (
    "ghi",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
)
N_METEO_FEATURES: int = len(METEO_FEATURES)  # 4
N_PV_FEATURES: int = 1                        # historical PV power (target var, kW)

# --- Satellite tensor layout -------------------------------------------------
# Visible band (B03) -> single channel by default; resize handled in the data pipeline.
SAT_CHANNELS: int = 1
SAT_IMG_SIZE: int = 64        # H == W; set to 224 in config when using a ViT backbone


def assert_embedding(tensor, name: str) -> None:
    """Fail loud if an encoder output is not [B, EMBED_DIM]."""
    if tensor.dim() != 2 or tensor.shape[-1] != EMBED_DIM:
        raise ValueError(
            f"{name} must be [B, {EMBED_DIM}], got tuple(shape)={tuple(tensor.shape)}"
        )
