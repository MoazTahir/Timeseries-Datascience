"""Tests for src/appliance_energy/models/benchmarks.py."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy.models import benchmarks as B


@pytest.fixture
def toy_series():
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    values = np.arange(len(index), dtype=float)
    return pd.Series(values, index=index)


def test_forecast_length_matches_horizon(toy_series):
    horizon = 24
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")

    for fn in [B.mean_forecast, B.naive_forecast, B.seasonal_naive_daily, B.drift_forecast]:
        forecast = fn(toy_series, horizon, index)
        assert len(forecast) == horizon
        assert forecast.index.equals(index)


def test_mean_forecast_is_flat_at_training_mean(toy_series):
    horizon = 5
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.mean_forecast(toy_series, horizon, index)

    assert (forecast == toy_series.mean()).all()


def test_naive_forecast_repeats_last_value(toy_series):
    horizon = 5
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.naive_forecast(toy_series, horizon, index)

    assert (forecast == toy_series.iloc[-1]).all()


def test_seasonal_naive_daily_repeats_value_from_24h_ago(toy_series):
    horizon = 24
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.seasonal_naive_daily(toy_series, horizon, index)

    # first forecast step should equal the value 24 hours before the origin
    assert forecast.iloc[0] == toy_series.iloc[-24]


def test_seasonal_naive_handles_horizon_longer_than_one_season(toy_series):
    """
    With a 48-hour horizon and daily (24h) seasonality, step 25 must copy
    step 1's *forecast* (one season back from step 25), not raw history -
    exercises the recursive extension logic.
    """
    horizon = 48
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.seasonal_naive_daily(toy_series, horizon, index)

    assert forecast.iloc[24] == pytest.approx(forecast.iloc[0])


def test_drift_forecast_is_linear(toy_series):
    horizon = 5
    index = pd.date_range(toy_series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.drift_forecast(toy_series, horizon, index)

    diffs = np.diff(forecast.values)
    assert np.allclose(diffs, diffs[0])


def test_perfect_linear_series_drift_forecast_is_exact():
    index = pd.date_range("2024-01-01", periods=100, freq="h")
    values = np.arange(100, dtype=float)  # slope exactly 1 per step
    series = pd.Series(values, index=index)

    horizon = 10
    future_index = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = B.drift_forecast(series, horizon, future_index)

    expected = values[-1] + np.arange(1, horizon + 1)
    assert np.allclose(forecast.values, expected)
