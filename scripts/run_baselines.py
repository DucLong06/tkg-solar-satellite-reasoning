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
from src.evaluation.benchmark_table import build_ablation_table, build_benchmark, relative_ordering
from src.data_pipeline import DataPipeline
from src.evaluation.baseline_models import is_eval_only
from src.training.config import Config
from src.training.evaluate import evaluate_model
from src.training.train_loop import fit
from src.lstm_baseline.lstm_forecaster import LSTMForecaster
from src.fusion_predictor.tkg_solar_model import TKGSolarModel

# Full comparison set. Persistence + ARIMA run eval-only (no gradient training);
# the rest train with Adam. (TFT via pytorch-forecasting is a documented follow-up
# — heavy version-sensitive integration, see plans/.../phase-05.)
MODELS = ["Persistence", "ARIMA", "LSTM", "GRU", "Transformer", "Temporal-GNN", "Proposed"]


def build(name: str, cfg: Config):
    if name == "LSTM":
        return LSTMForecaster(hidden_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name == "Proposed":
        return TKGSolarModel.from_config(cfg)
    return build_baseline(name, cfg)


# Proposed-model ablation arms: (label, {branch overrides}). Full = all branches.
ABLATION_ARMS = [
    ("Full", {}),
    ("-sat", {"use_sat": False}),
    ("-graph", {"use_graph": False}),
    ("-meteo", {"use_meteo": False}),
]


def run_ablation(cfg: Config, splits) -> dict[str, dict]:
    """Train the proposed model with each branch arm on identical data/seed."""
    out: dict[str, dict] = {}
    for label, overrides in ABLATION_ARMS:
        for k in ("use_sat", "use_meteo", "use_graph"):
            setattr(cfg, k, True)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        seed_everything(cfg.seed)
        print(f"\n=== Ablation arm: {label} ===")
        model = TKGSolarModel.from_config(cfg)
        fit(model, splits, cfg, verbose=False, desc=f"abl-{label}")
        metrics = evaluate_model(model, splits.test_loader, splits.scalers, cfg.device, cfg.mape_min_value)
        out[label] = metrics["overall"]
        print(f"{label}: MAE={metrics['overall']['mae']:.4f}")
    for k in ("use_sat", "use_meteo", "use_graph"):  # restore full-model state
        setattr(cfg, k, True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/paper_config.yaml")
    ap.add_argument("--out", default="plans/reports")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the proposed-model branch ablation (Full/-sat/-graph/-meteo)")
    ap.add_argument("--skip-proposed", action="store_true",
                    help="baselines-only on the satellite-ALIGNED splits — same windows + scaler "
                         "as the Proposed model (frames load but baselines ignore them). Use this "
                         "for the fair same-condition benchmark table.")
    ap.add_argument("--no-sat", action="store_true",
                    help="baselines-only on FULL DKASC without Himawari (skips Proposed + ablation). "
                         "WARNING: alignment switches to the full DKASC span -> windows/scaler differ "
                         "from satellite-aligned runs; NOT comparable with Proposed. For a fair "
                         "benchmark use --skip-proposed instead.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    seed_everything(cfg.seed)
    from src.training.gpu_autoscale import resolve_runtime
    resolve_runtime(cfg)
    splits = DataPipeline.load(
        cfg.dkasc_csv, cfg.himawari_dir,
        k=cfg.k, batch_size=cfg.batch_size, img_size=cfg.img_size,
        min_steps=cfg.min_steps,
        train_end=cfg.train_end, val_end=cfg.val_end,
        train_frac=cfg.train_frac, val_frac=cfg.val_frac,
        cadence_min=cfg.cadence_min, night_ghi_thresh=cfg.night_ghi_thresh,
        cache_dir=cfg.cache_dir, num_workers=cfg.num_workers,
        use_satellite=not args.no_sat,
    )
    print("Pipeline:", splits.meta)

    # Proposed is excluded in baselines-only runs: --no-sat lacks the satellite branch;
    # --skip-proposed trains it separately (Colab) on the same splits.
    baselines_only = args.no_sat or args.skip_proposed
    models = [m for m in MODELS if m != "Proposed"] if baselines_only else MODELS

    results = {}
    skipped = {}
    for name in models:
        seed_everything(cfg.seed)  # same init seed per model for fair comparison
        try:
            model = build(name, cfg)
            # Eval-only models (Persistence/ARIMA) have no params -> skip Adam.
            if is_eval_only(model):
                print(f"\n=== Evaluating {name} (eval-only, no training) ===")
                model.to(cfg.device)
            else:
                print(f"\n=== Training {name} ===")
                fit(model, splits, cfg, verbose=False)
            # ARIMA fits one model per window -> subsample + progress bar so it
            # finishes in minutes instead of hours.
            mw = 2000 if name == "ARIMA" else None
            metrics = evaluate_model(model, splits.test_loader, splits.scalers, cfg.device,
                                     cfg.mape_min_value, max_windows=mw, progress=(name == "ARIMA"))
        except ImportError as e:  # optional dep missing (e.g. statsmodels for ARIMA)
            skipped[name] = str(e)
            print(f"SKIP {name}: {e}")
            continue
        results[name] = metrics["overall"]
        print(f"{name}: MAE={metrics['overall']['mae']:.4f} "
              f"RMSE={metrics['overall']['rmse']:.4f} MAPE={metrics['overall']['mape']:.2f}%")

    table = build_benchmark(results, scalers=splits.scalers)  # kW + normalized [0,1] cols
    ordering = relative_ordering(results)

    ablation_md = ""
    if args.ablation and not args.no_sat:
        ablation = run_ablation(cfg, splits)
        ablation_md = (
            "\n\n## Ablation (proposed model arms, identical data/seed)\n\n"
            f"{build_ablation_table(ablation)}\n"
        )

    report = (
        "# DKASC Alice Springs Benchmark (G1 rigorous ranking)\n\n"
        f"Config: `{args.config}` | split: {splits.meta.get('split_mode')} | "
        f"PV units: {splits.meta.get('pv_units')}\n\n"
        f"splits.meta fingerprint (same-condition audit): `{splits.meta}`\n\n"
        f"{table}\n\n"
        f"**Relative ordering (best->worst MAE):** {' < '.join(ordering)}\n"
        + ("".join(f"\n> SKIPPED {n}: {m}\n" for n, m in skipped.items()) if skipped else "")
        + f"{ablation_md}\n"
        "> Metrics are our own on the co-located DKASC pipeline (same split/seed/"
        "scaler for every model). Published-paper tables are intentionally not "
        "compared head-to-head here; see the literature section for indicative (G2) "
        "DKASC numbers with explicit mismatch caveats.\n"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark-dkasc-ranking-and-ablation-report.md"
    out_path.write_text(report)
    print(f"\n{table}\n\nReport -> {out_path}")


if __name__ == "__main__":
    main()
