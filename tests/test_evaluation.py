"""Tests for src/appliance_energy/evaluation.py."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import evaluation as E


@pytest.fixture
def synthetic_train():
    """
    A synthetic seasonal series (period 24) with a bit of noise, long
    enough to compute a stable MASE scale.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    hours = index.hour.values
    seasonal = 50 + 30 * np.sin(2 * np.pi * hours / 24)
    noise = rng.normal(0, 2, size=len(index))
    return pd.Series(seasonal + noise, index=index)


def test_mase_is_zero_for_a_perfect_forecast(synthetic_train):
    y_true = synthetic_train.iloc[-48:]
    y_pred = y_true.copy()

    result = E.mase(y_true, y_pred, y_train=synthetic_train.iloc[:-48])
    assert result == pytest.approx(0.0, abs=1e-9)


def test_mase_is_one_for_a_seasonal_naive_forecast_matching_the_scale(synthetic_train):
    """
    MASE compares the forecast's MAE to the in-sample seasonal-naive MAE
    from the *training* set. A held-out seasonal-naive forecast on a
    stationary seasonal process should therefore land close to (not
    necessarily exactly) 1.0, since the test-period seasonal-naive error
    is drawn from the same noise-generating process as the training-period
    scale.
    """
    train = synthetic_train.iloc[:-48]
    test = synthetic_train.iloc[-48:]

    seasonal_naive_pred = synthetic_train.shift(24).iloc[-48:]

    result = E.mase(test, seasonal_naive_pred, y_train=train, seasonality=24)
    assert 0.5 < result < 1.5


def test_rmse_penalises_large_errors_more_than_mae():
    y_true = pd.Series([10, 10, 10, 10])
    y_pred_small_errors = pd.Series([12, 8, 12, 8])
    y_pred_one_big_error = pd.Series([10, 10, 10, 18])

    # Same total absolute error (8, mean absolute error 2.0) split differently.
    assert E.mae(y_true, y_pred_small_errors) == pytest.approx(
        E.mae(y_true, y_pred_one_big_error)
    )
    assert E.rmse(y_true, y_pred_one_big_error) > E.rmse(y_true, y_pred_small_errors)


def test_bias_sign_convention():
    y_true = pd.Series([10, 10, 10])
    over_forecast = pd.Series([12, 12, 12])
    under_forecast = pd.Series([8, 8, 8])

    assert E.bias(y_true, over_forecast) > 0
    assert E.bias(y_true, under_forecast) < 0


def test_rolling_origin_splits_tile_the_test_period_without_gaps_or_overlap():
    splits = E.rolling_origin_splits(n_total=1000, n_test=336, horizon=24)

    assert len(splits) == 336 // 24
    assert splits[0][0] == 1000 - 336
    for (a_start, a_end), (b_start, b_end) in zip(splits, splits[1:]):
        assert a_end == b_start  # no gap, no overlap
    assert splits[-1][1] == 1000


def test_run_rolling_backtest_returns_forecast_covering_whole_test_period(synthetic_train):
    def naive_fn(y_train, horizon, index, origin_i):
        return pd.Series(y_train.iloc[-1], index=index)

    forecast = E.run_rolling_backtest(synthetic_train, naive_fn, n_test=48, horizon=24)

    assert len(forecast) == 48
    assert forecast.index.equals(synthetic_train.index[-48:])
