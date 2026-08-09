"""
Tests for src/appliance_energy/features.py.

These specifically target the data-leakage failure modes the assignment
brief calls out: lag/rolling features must never expose the current or
a future value of the target.
"""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import features as F


@pytest.fixture
def toy_df():
    """
    A small, fully deterministic hourly series (values = index position)
    so leakage is trivial to detect: lag_k at row i must equal i - k
    exactly, and any off-by-one bug would show up immediately.
    """
    index = pd.date_range("2024-01-01", periods=300, freq="h")
    values = np.arange(300, dtype=float)
    return pd.DataFrame({"Appliances": values}, index=index)


def test_lag_features_reference_only_past_values(toy_df):
    out = F.add_lag_features(toy_df, target="Appliances", lags=[1, 24])

    valid = out.dropna(subset=["lag_1", "lag_24"])
    assert (valid["lag_1"] == valid["Appliances"] - 1).all()
    assert (valid["lag_24"] == valid["Appliances"] - 24).all()


def test_rolling_features_exclude_current_value(toy_df):
    out = F.add_rolling_features(toy_df, target="Appliances", windows=[3])

    # roll_mean_3 at row i should be the mean of rows i-3, i-2, i-1 -
    # i.e. strictly before i - never including Appliances[i] itself.
    row = out.iloc[10]
    expected_mean = toy_df["Appliances"].iloc[7:10].mean()
    assert row["roll_mean_3"] == pytest.approx(expected_mean)
    assert row["roll_mean_3"] != pytest.approx(toy_df["Appliances"].iloc[10])


def test_time_features_are_deterministic_functions_of_index(toy_df):
    out = F.add_time_features(toy_df)

    assert (out["hour"] == out.index.hour).all()
    assert (out["dayofweek"] == out.index.dayofweek).all()
    assert set(out["is_weekend"].unique()).issubset({0, 1})
    # cyclic encodings must lie on the unit circle
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)


def test_make_feature_table_has_no_missing_values(toy_df):
    table = F.make_feature_table(toy_df, lags=[1, 2, 24], windows=[3, 6])
    assert table.isna().sum().sum() == 0
    assert "Appliances" in table.columns


def test_make_feature_table_respects_requested_groups(toy_df):
    table = F.make_feature_table(toy_df, lags=[1], windows=[3], groups=["lags"])
    assert "lag_1" in table.columns
    assert "hour" not in table.columns
    assert "roll_mean_3" not in table.columns
