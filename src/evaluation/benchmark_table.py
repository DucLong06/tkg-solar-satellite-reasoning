"""Assemble a benchmark table comparing measured metrics vs paper Table 6.

Reproduction note (see docs/assumptions.md A0): absolute reproduction of the paper
numbers is NOT expected on the faithful-to-paper mismatched data. The useful
signal is RELATIVE ordering among our own models, not the gap to paper absolutes.
"""

from __future__ import annotations

# Paper Table 6 targets: model -> (MAE, RMSE, MAPE%)
PAPER_TABLE6: dict[str, tuple[float, float, float]] = {
    "LSTM": (0.128, 0.201, 9.84),
    "GRU": (0.121, 0.192, 9.17),
    "Transformer": (0.109, 0.174, 7.96),
    "Temporal-GNN": (0.097, 0.161, 7.11),
    "Proposed": (0.082, 0.143, 5.94),
}


def build_benchmark(results: dict[str, dict]) -> str:
    """results: {model_name: overall_metrics_dict (mae/rmse/mape)} -> markdown table.

    model_name should match a PAPER_TABLE6 key to show the paper column.
    """
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


def relative_ordering(results: dict[str, dict]) -> list[str]:
    """Model names sorted best (lowest MAE) -> worst, for the ordering check."""
    return sorted(results, key=lambda k: results[k]["mae"])
