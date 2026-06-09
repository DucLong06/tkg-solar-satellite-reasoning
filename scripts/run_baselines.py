"""Train all Table-6 models (LSTM, GRU, Transformer, Temporal-GNN, Proposed),
evaluate at 10/30/60 min, and write the benchmark comparison report.

    python scripts/run_baselines.py --config configs/paper_config.yaml

Heavy (5 models x epochs) — this is the Phase-6 GPU job. Use a small config for a
quick local sanity pass. Results + relative-ordering check are written to
plans/reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.seeding import seed_everything
from src.evaluation.baseline_models import build_baseline
from src.evaluation.benchmark_table import build_benchmark, relative_ordering
from src.data_pipeline import DataPipeline
from src.training.config import Config
from src.training.evaluate import evaluate_model
from src.training.train_loop import fit
from src.lstm_baseline.lstm_forecaster import LSTMForecaster
from src.fusion_predictor.tkg_solar_model import TKGSolarModel

MODELS = ["LSTM", "GRU", "Transformer", "Temporal-GNN", "Proposed"]


def build(name: str, cfg: Config):
    if name == "LSTM":
        return LSTMForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name == "Proposed":
        return TKGSolarModel.from_config(cfg)
    return build_baseline(name, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paper_config.yaml")
    ap.add_argument("--out", default="plans/reports")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    seed_everything(cfg.seed)
    splits = DataPipeline.load(
        cfg.opsd_path, cfg.nsrdb_path, cfg.himawari_dir,
        k=cfg.k, batch_size=cfg.batch_size, img_size=cfg.img_size,
        min_steps=cfg.min_steps, train_frac=cfg.train_frac, val_frac=cfg.val_frac,
        cache_dir=cfg.cache_dir, num_workers=cfg.num_workers,
    )
    print("Pipeline:", splits.meta)

    results = {}
    for name in MODELS:
        print(f"\n=== Training {name} ===")
        seed_everything(cfg.seed)  # same init seed per model for fair comparison
        model = build(name, cfg)
        fit(model, splits, cfg, verbose=False)
        metrics = evaluate_model(model, splits.test_loader, splits.scalers, cfg.device, cfg.mape_min_value)
        results[name] = metrics["overall"]
        print(f"{name}: MAE={metrics['overall']['mae']:.4f} "
              f"RMSE={metrics['overall']['rmse']:.4f} MAPE={metrics['overall']['mape']:.2f}%")

    table = build_benchmark(results, scalers=splits.scalers)  # also emit normalized [0,1] cols
    ordering = relative_ordering(results)
    report = (
        "# Benchmark vs Paper Table 6\n\n"
        f"Config: `{args.config}`\n\n"
        f"{table}\n\n"
        f"**Relative ordering (best->worst MAE):** {' < '.join(ordering)}\n\n"
        "> Absolute reproduction not expected on faithful-to-paper mismatched data "
        "(see docs/assumptions.md A0). Check relative ordering, not the paper gap.\n"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark-table6-comparison-report.md"
    out_path.write_text(report)
    print(f"\n{table}\n\nReport -> {out_path}")


if __name__ == "__main__":
    main()
