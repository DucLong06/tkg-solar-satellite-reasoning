"""Entrypoint: data -> model -> train -> evaluate (MAE/RMSE/MAPE @ 10/30/60 min).

    python main.py --config configs/smoke_config.yaml      # fast wiring check
    python main.py --config configs/paper_config.yaml      # full paper config
    python main.py --config configs/smoke_config.yaml --generate-synthetic

Model is selected by config.model: "lstm" (baseline) or "full" (TKGSolarModel).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.seeding import seed_everything
from src.common.shapes import HORIZON_MINUTES
from src.data_pipeline import DataPipeline
from src.training.config import Config
from src.training.train_loop import fit, predict_loader


def build_model(cfg: Config):
    if cfg.model == "lstm":
        from src.lstm_baseline.lstm_forecaster import LSTMForecaster

        return LSTMForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if cfg.model == "full":
        from src.fusion_predictor.tkg_solar_model import TKGSolarModel

        return TKGSolarModel.from_config(cfg)
    raise ValueError(f"unknown model {cfg.model!r} (use 'lstm' or 'full')")


def print_metrics(metrics: dict) -> None:
    print("\n=== Test metrics (original units) ===")
    print(f"{'horizon':>10} | {'MAE':>9} | {'RMSE':>9} | {'MAPE %':>9}")
    print("-" * 46)
    for label in ["overall", *[f"{m}min" for m in HORIZON_MINUTES]]:
        m = metrics[label]
        print(f"{label:>10} | {m['mae']:9.4f} | {m['rmse']:9.4f} | {m['mape']:9.2f}")
    print(f"(MAPE excludes {metrics['overall']['mape_excluded_frac']*100:.1f}% near-zero/night samples)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/smoke_config.yaml")
    ap.add_argument("--generate-synthetic", action="store_true",
                    help="generate synthetic data first if data/ is empty")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    seed_everything(cfg.seed)

    if args.generate_synthetic:
        import numpy as np
        from scripts.generate_synthetic_data import gen_himawari, gen_nsrdb, gen_opsd

        rng = np.random.default_rng(cfg.seed)
        out = Path("data")
        gen_opsd(out, 30, rng); gen_nsrdb(out, 30, rng); gen_himawari(out, 30, rng)
        print("Synthetic data generated in data/.")

    splits = DataPipeline.load(
        cfg.opsd_path, cfg.nsrdb_path, cfg.himawari_dir,
        k=cfg.k, batch_size=cfg.batch_size, img_size=cfg.img_size,
        min_steps=cfg.min_steps, train_frac=cfg.train_frac, val_frac=cfg.val_frac,
        cache_dir=cfg.cache_dir, num_workers=cfg.num_workers, scaler_out=cfg.scaler_out,
    )
    print("M1 pipeline:", splits.meta)

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model '{cfg.model}' ({n_params/1e6:.2f}M params) on {cfg.device}")

    loss_fn = None
    if cfg.use_advanced_loss:
        from src.advanced_loss.composite_loss import CompositeLoss

        loss_fn = CompositeLoss(lambda_prob=cfg.lambda_prob, lambda_physics=cfg.lambda_physics)

    history = fit(model, splits, cfg, loss_fn=loss_fn)
    print(f"\nBest val MAE: {history['best_val_mae']:.5f}  ({history['best_checkpoint']})")

    # Save the resolved config alongside the checkpoint for reproducibility.
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(str(Path(cfg.checkpoint_dir) / f"resolved_config_{cfg.model}.yaml"))

    from src.metrics.regression_metrics import compute_per_horizon

    yt, yp = predict_loader(model, splits.test_loader, cfg.device)
    metrics = compute_per_horizon(
        splits.scalers.inverse_pv(yt.numpy()),
        splits.scalers.inverse_pv(yp.numpy()),
        HORIZON_MINUTES,
        cfg.mape_min_value,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
