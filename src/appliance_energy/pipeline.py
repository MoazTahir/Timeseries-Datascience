"""
End-to-end pipeline orchestration.

Runs every step of the assignment in order:
    1. Load / clean / resample the dataset.
    2. EDA + stationarity diagnostics.
    3. Benchmark models (Part 3).
    4. SARIMAX: grid search, fit, residual diagnostics, rolling forecast (Part 4).
    5. Covariate feature table (Part 5).
    6. Feature-based ML model + ablation (Part 6).
    7. Foundation model (Part 7).
    8. Consolidated evaluation, comparison table, and figures (Part 8).

``scripts/run_pipeline.py`` is a thin CLI wrapper around
:func:`run_pipeline`, so the whole analysis reproduces with:

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

from . import config, data as D, evaluation as E, features as F, plotting as P
from .models import benchmarks as B, feature_models as FM, sarimax as S


def _log_step(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def _save_fig(fig, path) -> None:
    """
    Save a figure and close it immediately.

    The pipeline generates ~14 figures in a single long-running process;
    matplotlib keeps every created figure alive in memory until it is
    explicitly closed, so without this the whole run's peak memory usage
    grows with every plot produced (compounding badly once the
    memory-hungry XGBoost/Chronos steps run later in the same process).
    """
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    force_download: bool = False,
    use_cached_grid_search: bool = True,
    include_foundation_model: bool = True,
) -> dict:
    """
    Run the full analysis pipeline and write all outputs to outputs/.

    Returns a dict with the key intermediate objects (useful for
    interactive/notebook use): hourly data, forecasts, metrics table,
    SARIMAX fit, feature-model, grid search table.
    """
    t_start = time.time()
    config.ensure_directories()

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    _log_step("Step 1/8: Loading and preparing data")
    hourly = D.load_hourly_data(force_download=force_download)
    y = hourly[config.TARGET]

    train = y.iloc[: -config.TEST_STEPS]
    test = y.iloc[-config.TEST_STEPS :]

    print(f"Hourly data: {hourly.shape[0]} rows, {hourly.index.min()} to {hourly.index.max()}")
    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} rows)")
    print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} rows)")

    # ----------------------------------------------------------------
    # 2. EDA + stationarity
    # ----------------------------------------------------------------
    _log_step("Step 2/8: EDA and stationarity diagnostics")

    missing_summary = D.report_missing_values(hourly)
    missing_summary.to_csv(config.METRICS_DIR / "missing_value_summary.csv")

    stationarity_table = D.run_stationarity_suite(y, name=config.TARGET)
    stationarity_table.to_csv(config.METRICS_DIR / "stationarity_tests.csv", index=False)
    print(stationarity_table[["series", "test", "p_value", "is_stationary_at_5pct"]])

    fig = P.plot_full_series(y)
    _save_fig(fig, config.FIGURE_DIR / "01_full_series.png")

    fig = P.plot_series_window(y, days=14)
    _save_fig(fig, config.FIGURE_DIR / "02_series_last_14_days.png")

    fig = P.plot_hourly_and_weekday_profiles(y)
    _save_fig(fig, config.FIGURE_DIR / "03_hourly_weekday_profiles.png")

    decomposition = D.decompose_series(y, period=config.DAILY_PERIOD, method="stl")
    fig = P.plot_decomposition(decomposition, title="STL decomposition (daily period)")
    _save_fig(fig, config.FIGURE_DIR / "04_stl_decomposition.png")

    fig = P.plot_acf_pacf(y, lags=72, title_suffix="- Appliances (level)")
    _save_fig(fig, config.FIGURE_DIR / "05_acf_pacf_level.png")

    fig = P.plot_acf_pacf(
        y.diff(config.DAILY_PERIOD).dropna(), lags=72, title_suffix="- seasonal diff (lag 24)"
    )
    _save_fig(fig, config.FIGURE_DIR / "06_acf_pacf_seasonal_diff.png")

    corr_cols = [config.TARGET] + config.SARIMAX_EXOG_COLS + config.INDOOR_TEMP_COLS[:3]
    fig = P.plot_correlation_heatmap(hourly, corr_cols, title="Correlation: target vs. key covariates")
    _save_fig(fig, config.FIGURE_DIR / "07_correlation_heatmap.png")

    # ----------------------------------------------------------------
    # 3. Benchmarks
    # ----------------------------------------------------------------
    _log_step("Step 3/8: Benchmark models")
    forecasts = B.run_all_benchmarks(y)
    for name in forecasts:
        print(f"  {name}: done")

    # ----------------------------------------------------------------
    # 4. SARIMAX
    # ----------------------------------------------------------------
    _log_step("Step 4/8: SARIMAX grid search and rolling forecast")

    grid_search_path = config.METRICS_DIR / "sarima_grid_search.csv"
    if use_cached_grid_search and grid_search_path.exists():
        print(f"Loading cached grid search results from {grid_search_path}")
        grid_df = pd.read_csv(grid_search_path)
    else:
        grid_df = S.grid_search_sarima(train, verbose=True)
        grid_df.to_csv(grid_search_path, index=False)

    best_order = S.best_order_from_grid(grid_df)
    print(f"Selected SARIMA order (p,d,q) = {best_order}, seasonal_order = {config.SEASONAL_ORDER}")

    # Target-only SARIMA
    sarimax_uni_forecast, sarimax_uni_ci, sarimax_uni_fit = S.rolling_forecast_sarimax(
        y, order=best_order, exog=None
    )
    forecasts["sarimax_univariate"] = sarimax_uni_forecast.rename("sarimax_univariate")

    # SARIMAX with exogenous weather covariates (conditional forecast -
    # see Part 9 Q5: uses realised future exog values from the test set)
    exog = hourly[config.SARIMAX_EXOG_COLS]
    sarimax_exog_forecast, sarimax_exog_ci, sarimax_exog_fit = S.rolling_forecast_sarimax(
        y, order=best_order, exog=exog
    )
    forecasts["sarimax"] = sarimax_exog_forecast.rename("sarimax")

    # Residual diagnostics on the initial fit
    residuals = sarimax_exog_fit.resid
    fig = P.plot_residual_diagnostics(residuals, title="SARIMAX residual diagnostics (with exog)")
    _save_fig(fig, config.FIGURE_DIR / "08_sarimax_residual_diagnostics.png")

    ljung_box = S.ljung_box_test(residuals)
    ljung_box.to_csv(config.METRICS_DIR / "sarimax_ljung_box.csv")
    print("Ljung-Box test on SARIMAX residuals:")
    print(ljung_box)

    fig = P.plot_sarima_grid_heatmap(grid_df, d_value=best_order[1])
    _save_fig(fig, config.FIGURE_DIR / "09_sarima_aic_heatmap.png")

    # ----------------------------------------------------------------
    # 5+6. Feature table + feature-based ML model
    # ----------------------------------------------------------------
    _log_step("Step 5-6/8: Feature engineering and feature-based ML model")

    feature_table = F.make_feature_table(hourly)
    feature_table.to_csv(config.INTERIM_DATA_DIR / "feature_table.csv")

    xgb_forecast, xgb_model, xgb_cols = FM.rolling_forecast_feature_model(
        feature_table, FM.fit_xgboost
    )
    forecasts["feature_model"] = xgb_forecast.rename("feature_model")

    histgb_forecast, histgb_model, _ = FM.rolling_forecast_feature_model(
        feature_table, FM.fit_histgb
    )
    forecasts["feature_model_histgb"] = histgb_forecast.rename("feature_model_histgb")

    importance = FM.feature_importance(xgb_model, xgb_cols)
    importance.to_csv(config.METRICS_DIR / "feature_importance.csv")
    fig = P.plot_feature_importance(importance, title="XGBoost feature importance")
    _save_fig(fig, config.FIGURE_DIR / "10_feature_importance.png")

    print("Running feature-group ablation study ...")
    ablation_df = FM.run_feature_group_ablation(hourly, FM.fit_xgboost, train, test)
    ablation_df.to_csv(config.METRICS_DIR / "feature_ablation.csv", index=False)
    print(ablation_df)

    # ----------------------------------------------------------------
    # 7. Foundation model
    # ----------------------------------------------------------------
    if include_foundation_model:
        _log_step("Step 7/8: Foundation model (Chronos, zero-shot)")

        # Run as a separate subprocess rather than calling
        # FD.rolling_forecast_foundation(y) in-process: XGBoost has
        # already been fit above (Step 5-6), and its OpenMP thread pool
        # deadlocks with PyTorch's thread pool if both are initialised in
        # the same process on macOS. See scripts/run_foundation_model.py
        # for the full explanation.
        subprocess.run(
            [sys.executable, str(config.PROJECT_ROOT / "scripts" / "run_foundation_model.py")],
            check=True,
        )

        foundation_path = config.FORECAST_DIR / "foundation_forecast.csv"
        foundation_result = pd.read_csv(foundation_path, index_col=0, parse_dates=True)
        forecasts["foundation_model"] = foundation_result["foundation_model"]
        # foundation_result also carries foundation_lower/foundation_upper
        # quantile columns (Chronos's 10-90% interval); plotted alongside
        # the SARIMAX confidence interval in notebooks/06_foundation_model.ipynb
        # rather than here, to keep the main comparison figure below readable
        # with only one uncertainty band.

    # ----------------------------------------------------------------
    # 8. Consolidated evaluation
    # ----------------------------------------------------------------
    _log_step("Step 8/8: Evaluation, comparison table, and figures")

    results = [
        E.evaluate_forecast(name, test, pred, train) for name, pred in forecasts.items()
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

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)
    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts.csv")

    main_cols = [
        c
        for c in [
            "mean",
            "naive",
            "seasonal_naive_daily",
            "seasonal_naive_weekly",
            "drift",
            "sarimax",
            "feature_model",
            "foundation_model",
        ]
        if c in forecast_df.columns
    ]
    fig = P.plot_forecast_vs_actual(
        train,
        test,
        forecast_df[main_cols],
        conf_int=sarimax_exog_ci,
        conf_int_model="sarimax",
    )
    _save_fig(fig, config.FIGURE_DIR / "11_forecast_comparison.png")

    fig = P.plot_forecast_zoom(test, forecast_df[main_cols], days=4)
    _save_fig(fig, config.FIGURE_DIR / "12_forecast_comparison_zoom.png")

    fig = P.plot_error_diagnostics(test, forecast_df[main_cols])
    _save_fig(fig, config.FIGURE_DIR / "13_error_diagnostics.png")

    fig = P.plot_metric_comparison(metrics_df)
    _save_fig(fig, config.FIGURE_DIR / "14_metric_comparison.png")

    elapsed = time.time() - t_start
    _log_step(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"Forecasts:  {config.FORECAST_DIR / 'all_forecasts.csv'}")
    print(f"Metrics:    {config.METRICS_DIR / 'model_comparison.csv'}")
    print(f"Figures:    {config.FIGURE_DIR}")

    return {
        "hourly": hourly,
        "train": train,
        "test": test,
        "forecasts": forecasts,
        "forecast_df": forecast_df,
        "metrics_df": metrics_df,
        "grid_df": grid_df,
        "best_order": best_order,
        "sarimax_fit": sarimax_exog_fit,
        "xgb_model": xgb_model,
        "ablation_df": ablation_df,
    }
