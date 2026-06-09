"""Assemble a benchmark table comparing measured metrics vs paper Table 6.

Reproduction note (see docs/assumptions.md A0): absolute reproduction of the paper
numbers is NOT expected on the faithful-to-paper mismatched data. The useful
signal is RELATIVE ordering among our own models, not the gap to paper absolutes.

Scale note: the paper reports MAE/RMSE on Min-Max **normalized** PV in [0,1], while
our pipeline computes them in **original units (MW)** via inverse-transform. The two
are therefore NOT directly comparable. Pass the PV scaler (or its range) to
``build_benchmark`` to also emit normalized columns:

    MAE_norm = MAE_MW / (pv_max - pv_min)   # exact: Min-Max scaling is linear

MAPE% is scale-invariant and already comparable to the paper as-is.
"""

from __future__ import annotations

# Paper Table 6 targets: model -> (MAE, RMSE, MAPE%) on normalized [0,1] PV.
PAPER_TABLE6: dict[str, tuple[float, float, float]] = {
    "LSTM": (0.128, 0.201, 9.84),
    "GRU": (0.121, 0.192, 9.17),
    "Transformer": (0.109, 0.174, 7.96),
    "Temporal-GNN": (0.097, 0.161, 7.11),
    "Proposed": (0.082, 0.143, 5.94),
}


def pv_range(scalers_or_range) -> float | None:
    """Extract the PV Min-Max range (max-min) used to normalize the metrics.

    Accepts a ``Scalers`` instance, a fitted sklearn ``MinMaxScaler`` (the
    ``pv_scaler``), or a plain float range. Returns None if nothing usable.
    """
    if scalers_or_range is None:
        return None
    if isinstance(scalers_or_range, (int, float)):
        return float(scalers_or_range)
    pv = getattr(scalers_or_range, "pv_scaler", scalers_or_range)
    return float(pv.data_max_[0] - pv.data_min_[0])


def build_benchmark(results: dict[str, dict], scalers=None) -> str:
    """results: {model_name: overall_metrics_dict (mae/rmse/mape)} -> markdown table.

    model_name should match a PAPER_TABLE6 key to show the paper column. When
    ``scalers`` (a Scalers / pv_scaler / float range) is given, MAE/RMSE are also
    reported on the normalized [0,1] scale so they line up with the paper column;
    the original-unit (MW) values are kept as trailing reference columns.
    """
    rng = pv_range(scalers)
    if rng is None:
        # Original-units-only table (paper columns differ in scale — see module docstring).
        lines = [
            "| Model | MAE (ours) | MAE (paper) | RMSE (ours) | RMSE (paper) | MAPE% (ours) | MAPE% (paper) |",
            "|-------|-----------|-------------|-------------|--------------|--------------|---------------|",
        ]
        for name, m in results.items():
            paper = PAPER_TABLE6.get(name)
            pm = f"{paper[0]:.3f}" if paper else "-"
            pr = f"{paper[1]:.3f}" if paper else "-"
            pp = f"{paper[2]:.2f}" if paper else "-"
            lines.append(
                f"| {name} | {m['mae']:.3f} | {pm} | {m['rmse']:.3f} | {pr} | {m['mape']:.2f} | {pp} |"
            )
        return "\n".join(lines)

    # Normalized table: MAE/RMSE divided by the PV range -> directly comparable to paper.
    lines = [
        "| Model | MAE [0,1] | MAE (paper) | RMSE [0,1] | RMSE (paper) | MAPE% | MAPE% (paper) | MAE (MW) | RMSE (MW) |",
        "|-------|-----------|-------------|------------|--------------|-------|---------------|----------|-----------|",
    ]
    for name, m in results.items():
        paper = PAPER_TABLE6.get(name)
        pm = f"{paper[0]:.3f}" if paper else "-"
        pr = f"{paper[1]:.3f}" if paper else "-"
        pp = f"{paper[2]:.2f}" if paper else "-"
        lines.append(
            f"| {name} | {m['mae'] / rng:.3f} | {pm} | {m['rmse'] / rng:.3f} | {pr} | "
            f"{m['mape']:.2f} | {pp} | {m['mae']:.2f} | {m['rmse']:.2f} |"
        )
    return "\n".join(lines)


def relative_ordering(results: dict[str, dict]) -> list[str]:
    """Model names sorted best (lowest MAE) -> worst, for the ordering check.

    Scale-independent: dividing every MAE by the same PV range preserves order.
    """
    return sorted(results, key=lambda k: results[k]["mae"])
