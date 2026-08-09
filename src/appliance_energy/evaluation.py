"""
Evaluation metrics and the rolling-origin backtesting harness (Part 2/8).

Every model in this project is evaluated the same way: walk forward
across the 14-day test period in HORIZON-sized (24-hour) blocks, each
time forecasting from a fresh origin using only data available up to
that origin. This directly implements the "24 hour forecast horizon"
instruction while still covering the whole held-out test window, and
gives every model the same, fair, leakage-free evaluation protocol.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean signed error: positive means the model over-forecasts on average."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = config.DAILY_PERIOD,
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales the forecast's MAE by the in-sample MAE of a seasonal naive
    forecast (lag = ``seasonality``) computed on the *training* data only.
    MASE < 1 means the model beats seasonal-naive in-sample; MASE is
    scale-free so it is comparable across series/models.
    """
    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values
    )
    scale = seasonal_errors.mean()

    if scale == 0 or np.isnan(scale):
        return np.nan

    return mae(y_true, y_pred) / scale


def evaluate_forecast(
    name: str, y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series
) -> dict:
    """Compute the full metric set (MAE, RMSE, MASE, Bias) for one model."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).reindex(y_true.index).astype(float)

    valid = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true.loc[valid], y_pred.loc[valid]

    return {
        "model": name,
        "n_obs": int(valid.sum()),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train),
        "Bias": bias(y_true, y_pred),
    }


def summarise_metrics(results: list[dict]) -> pd.DataFrame:
    """Build the comparison table, ranked by MASE, with %-worse-than-best-benchmark."""
    df = pd.DataFrame(results).sort_values("MASE").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Rolling-origin backtesting harness
# ------------------------------------------------------------------

def rolling_origin_splits(
    n_total: int,
    n_test: int = config.TEST_STEPS,
    horizon: int = config.HORIZON,
) -> list[tuple[int, int]]:
    """
    Build (train_end, test_end) index pairs for an expanding-window,
    rolling-origin backtest over the final ``n_test`` observations.

    Each split's training window is [0, train_end) and its forecast
    window is [train_end, train_end + horizon). Consecutive splits move
    the origin forward by exactly ``horizon`` steps, so the splits tile
    the test period with no overlap and no gaps.
    """
    n_origins = n_test // horizon
    first_train_end = n_total - n_test

    splits = []
    for i in range(n_origins):
        train_end = first_train_end + i * horizon
        test_end = train_end + horizon
        splits.append((train_end, test_end))

    return splits


def run_rolling_backtest(
    y: pd.Series,
    forecast_fn: Callable[[pd.Series, int, pd.DatetimeIndex, int], pd.Series],
    n_test: int = config.TEST_STEPS,
    horizon: int = config.HORIZON,
) -> pd.Series:
    """
    Generic rolling-origin backtest driver for models whose forecast
    function only needs the target history (benchmarks, univariate
    SARIMA, foundation model).

    Parameters
    ----------
    y:
        Full target series (train + test).
    forecast_fn:
        Callable ``forecast_fn(y_train, horizon, index, origin_i) -> pd.Series``
        returning a length-``horizon`` forecast for the given origin.
        ``origin_i`` (0-indexed rolling-window number) is passed through
        so model-specific state (e.g. a SARIMAX fit) can be cached/updated
        by the caller if desired via closures.
    n_test, horizon:
        Backtest configuration; see :func:`rolling_origin_splits`.

    Returns
    -------
    A single concatenated series of forecasts spanning the whole test
    period, indexed to match ``y``.
    """
    splits = rolling_origin_splits(len(y), n_test=n_test, horizon=horizon)

    all_forecasts = []
    for origin_i, (train_end, test_end) in enumerate(splits):
        y_train = y.iloc[:train_end]
        index = y.index[train_end:test_end]

        pred = forecast_fn(y_train, horizon, index, origin_i)
        all_forecasts.append(pd.Series(pred, index=index))

    return pd.concat(all_forecasts).sort_index()
