#!/usr/bin/env python3
"""
Standalone re-evaluation: recompute the model comparison table and
figures from an already-saved outputs/forecasts/all_forecasts.csv,
without re-fitting any models. Useful for regenerating figures/metrics
after tweaking evaluation.py or plotting.py, without paying the cost of
re-running SARIMAX/XGBoost/Chronos.

Usage:
    python scripts/run_pipeline.py     # first, to produce all_forecasts.csv
    python scripts/evaluate_models.py  # then, to re-evaluate/re-plot only
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from appliance_energy import config, data as D, evaluation as E, plotting as P
from appliance_energy.models import benchmarks as B


def main() -> None:
    config.ensure_directories()

    forecast_path = config.FORECAST_DIR / "all_forecasts.csv"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"{forecast_path} not found - run scripts/run_pipeline.py first."
        )

    forecast_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    test = forecast_df["actual"]

    hourly = D.load_hourly_data()
    y = hourly[config.TARGET]
    train = y.iloc[: -config.TEST_STEPS]

    model_cols = [c for c in forecast_df.columns if c != "actual"]
    results = [
        E.evaluate_forecast(name, test, forecast_df[name], train) for name in model_cols
    ]
    metrics_df = E.summarise_metrics(results)

    best_benchmark_mase = metrics_df.loc[
        metrics_df["model"].isin(B.BENCHMARK_MODELS.keys()), "MASE"
    ].min()
    metrics_df["pct_vs_best_benchmark"] = (
        100 * (metrics_df["MASE"] - best_benchmark_mase) / best_benchmark_mase
    )

    print(metrics_df.round(3))
    metrics_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    fig = P.plot_metric_comparison(metrics_df)
    fig.savefig(config.FIGURE_DIR / "14_metric_comparison.png", dpi=200, bbox_inches="tight")

    print(f"\nSaved {config.METRICS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
