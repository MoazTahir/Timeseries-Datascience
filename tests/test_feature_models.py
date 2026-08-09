"""
Tests for src/appliance_energy/models/feature_models.py, focused on the
recursive multi-step forecasting logic, which is the module's main
leakage-safety risk: lags/rolling windows shorter than the horizon must
be filled with the model's own prior predictions, never with future
actuals.
"""

import numpy as np
import pandas as pd
import pytest

from appliance_energy.models import feature_models as FM


class _EchoLag1Model:
    """
    A stub 'model' whose prediction is always exactly the lag_1 feature
    value. If recursive_forecast leaked future actuals instead of using
    its own predictions, this model's forecast trajectory would jump to
    the (unavailable) true future values instead of propagating a single
    seed value forward.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1"].to_numpy()


def test_recursive_forecast_uses_predictions_not_future_actuals():
    history = pd.Series(
        [1.0] * 200,
        index=pd.date_range("2024-01-01", periods=200, freq="h"),
    )
    # Seed the most recent value distinctly so we can trace it forward.
    history.iloc[-1] = 42.0

    future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=5, freq="h")
    # exog_future deliberately contains no information the echo model uses.
    exog_future = pd.DataFrame({"hour": future_index.hour}, index=future_index)

    feature_cols = ["lag_1", "hour"]

    forecast = FM.recursive_forecast(
        model=_EchoLag1Model(),
        history=history,
        exog_future=exog_future,
        horizon=5,
        feature_cols=feature_cols,
        lags=[1],
        windows=[],
    )

    # Step 1 should echo the seed value (42.0); every subsequent step
    # should echo the *previous forecast step*, not a future actual
    # (there are none available), proving the recursion feeds predictions
    # forward rather than reading ahead.
    assert forecast.iloc[0] == pytest.approx(42.0)
    assert (forecast.to_numpy() == 42.0).all()


def test_recursive_forecast_respects_feature_column_order():
    history = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2024-01-01", periods=50, freq="h"),
    )
    future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=3, freq="h")
    exog_future = pd.DataFrame({"hour": future_index.hour, "dummy": 0}, index=future_index)

    seen_columns = []

    class _RecordingModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            seen_columns.append(list(X.columns))
            return np.zeros(len(X))

    feature_cols = ["dummy", "lag_1", "hour"]

    FM.recursive_forecast(
        model=_RecordingModel(),
        history=history,
        exog_future=exog_future,
        horizon=3,
        feature_cols=feature_cols,
        lags=[1],
        windows=[],
    )

    assert all(cols == feature_cols for cols in seen_columns)


def test_recursive_forecast_output_length_matches_horizon():
    history = pd.Series(
        np.ones(200),
        index=pd.date_range("2024-01-01", periods=200, freq="h"),
    )
    future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    exog_future = pd.DataFrame({"hour": future_index.hour}, index=future_index)

    forecast = FM.recursive_forecast(
        model=_EchoLag1Model(),
        history=history,
        exog_future=exog_future,
        horizon=24,
        feature_cols=["lag_1", "hour"],
        lags=[1],
        windows=[],
    )

    assert len(forecast) == 24
    assert forecast.index.equals(future_index)
