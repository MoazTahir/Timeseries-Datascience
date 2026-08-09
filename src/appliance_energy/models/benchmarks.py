"""
Benchmark forecasting models (Part 3): mean, naive, daily seasonal naive,
weekly seasonal naive, and drift.

Each ``*_forecast`` function has the signature
``(y_train, horizon, index) -> pd.Series`` so they plug directly into
:func:`appliance_energy.evaluation.run_rolling_backtest`.
"""

from __future__ import annotations

import pandas as pd

from .. import config


def mean_forecast(y_train: pd.Series, horizon: int, index: pd.DatetimeIndex) -> pd.Series:
    """Flat forecast at the training-set mean. The weakest sensible baseline."""
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train: pd.Series, horizon: int, index: pd.DatetimeIndex) -> pd.Series:
    """Flat forecast at the last observed value ("tomorrow = today, right now")."""
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(
    y_train: pd.Series, horizon: int, index: pd.DatetimeIndex, seasonality: int
) -> pd.Series:
    """
    Recursive seasonal naive forecast: each forecast step repeats the
    value observed ``seasonality`` steps earlier, extending the history
    with its own forecasts as it goes (needed once the horizon exceeds
    the seasonal period so later steps still have a "one season ago"
    value to copy).
    """
    history = list(y_train.values)
    values = []

    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    return pd.Series(values, index=index)


def seasonal_naive_daily(y_train: pd.Series, horizon: int, index: pd.DatetimeIndex) -> pd.Series:
    return seasonal_naive_forecast(y_train, horizon, index, config.DAILY_PERIOD).rename(
        "seasonal_naive_daily"
    )


def seasonal_naive_weekly(y_train: pd.Series, horizon: int, index: pd.DatetimeIndex) -> pd.Series:
    return seasonal_naive_forecast(y_train, horizon, index, config.WEEKLY_PERIOD).rename(
        "seasonal_naive_weekly"
    )


def drift_forecast(y_train: pd.Series, horizon: int, index: pd.DatetimeIndex) -> pd.Series:
    """
    Extrapolate the straight line joining the first and last training
    observations. Captures a linear trend but nothing seasonal.
    """
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
    values = [y_train.iloc[-1] + slope * step for step in range(1, horizon + 1)]
    return pd.Series(values, index=index, name="drift")


BENCHMARK_MODELS = {
    "mean": mean_forecast,
    "naive": naive_forecast,
    "seasonal_naive_daily": seasonal_naive_daily,
    "seasonal_naive_weekly": seasonal_naive_weekly,
    "drift": drift_forecast,
}


def run_all_benchmarks(y: pd.Series) -> dict[str, pd.Series]:
    """
    Run every benchmark through the rolling-origin backtest and return a
    dict of {model_name: full-test-period forecast series}.
    """
    from ..evaluation import run_rolling_backtest

    forecasts = {}
    for name, fn in BENCHMARK_MODELS.items():
        wrapped = lambda y_train, horizon, index, origin_i, fn=fn: fn(y_train, horizon, index)
        forecasts[name] = run_rolling_backtest(y, wrapped)

    return forecasts
