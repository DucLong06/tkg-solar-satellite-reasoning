# TKG Solar Power Forecasting — Paper Reproduction

Reproduction of *"Temporal Knowledge Graph Reasoning for Real-Time Solar Power
Forecasting Using Satellite Data"* (Hoàng Đức Long, 2026). A multi-modal pipeline
that fuses **satellite imagery + meteorology + a temporal knowledge graph**
to forecast PV power at 10/30/60-min horizons, run end-to-end via `python main.py`.

> **Reproduction scope (read first).** The three source datasets are not
> co-located (OPSD PV = Europe; NSRDB/Himawari = Asia). Per a fully-informed user
> decision this stays **faithful to the paper's data choice**, which makes this a
> **mechanical / code reproduction**, not a physically-valid model. The temporal
> knowledge graph's geographic edges are not semantically meaningful on mismatched
> data. Success = a correct, deterministic, runnable pipeline with honest
> documented caveats, **not** the paper's absolute MAE≈0.082.

## Components (`src/`)

Each directory is named for what it does (no cryptic codes):

| Component | Path | Role |
|-----------|------|------|
| Data pipeline | `src/data_pipeline/` | load → align → clean → scale → window → split |
| Metrics | `src/metrics/` | MAE / RMSE / MAPE (inverse-scaled) |
| Training | `src/training/` | train loop, config, evaluate |
| LSTM baseline | `src/lstm_baseline/` | LSTM baseline forecaster |
| Meteo encoder | `src/meteo_encoder/` | meteo seq → `H_met [B,128]` (GRU+Transformer) |
| Satellite encoder | `src/satellite_encoder/` | sat frames → `F_sat [B,128]` (timm + channel adapter) |
| TKG builder | `src/tkg_builder/` | entities/relations → dynamic adjacency `A_t` |
| Graph learner | `src/graph_learner/` | GAT + Fourier + temporal attn → `H_graph [B,128]` |
| Fusion predictor | `src/fusion_predictor/` | concat 3×128 → MLP → `[B,3]`; full `TKGSolarModel` |
| Advanced loss | `src/advanced_loss/` | probabilistic + physics-informed composite loss |

The dimension contract (all encoders = 128-dim) is pinned in
[`src/common/shapes.py`](src/common/shapes.py).

## Quick start

```bash
# 1. Environment (uv; CPU torch index is configured in pyproject.toml)
uv sync --extra dev

# 2. Data — pick ONE:
#    (a) Synthetic smoke data (no keys, runs anywhere):
uv run python scripts/generate_synthetic_data.py --days 30
#    (b) Real data (faithful to paper):
uv run python scripts/download_opsd.py
NREL_API_KEY=... uv run python scripts/download_nsrdb.py --lat 10.82 --lon 106.63 --year 2016
uv run python scripts/download_himawari.py --date 2016-06-01   # ROI subset, long-lead

# 3. Run
uv run python main.py --config configs/smoke_config.yaml     # fast end-to-end check
uv run python main.py --config configs/paper_config.yaml     # full paper config (GPU recommended)

# 4. Tests (determinism + shapes + no-NaN/grad)
uv run pytest
```

## Notebooks

- [`notebooks/data_pipeline_walkthrough.ipynb`](notebooks/data_pipeline_walkthrough.ipynb) — đi qua từng bước data pipeline (load→align→clean→split→clip→scale→window) kèm công thức.
- [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) — clone-and-run trên Colab GPU: mount dữ liệu Drive → train → đánh giá → lưu checkpoint.
- [`notebooks/results_analysis.ipynb`](notebooks/results_analysis.ipynb) — phân tích kết quả pred-vs-actual + lỗi theo horizon.

## Compute note

Local dev is CPU-only. The full 200-epoch run (per-frame ViT + per-timestep GAT)
needs a GPU — train on Colab (see `notebooks/colab_train.ipynb`) or an SSH GPU box.
The smoke config keeps the model tiny so `python main.py` finishes in seconds for
wiring verification.
