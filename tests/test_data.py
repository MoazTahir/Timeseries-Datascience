"""Tests for src/appliance_energy/data.py."""

import numpy as np
import pandas as pd
import pytest

from appliance_energy import data as D


@pytest.fixture
def raw_like_df():
    """A small 10-minute-frequency dataframe shaped like the raw UCI data."""
    index = pd.date_range("2024-01-01", periods=6 * 24 * 3, freq="10min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Appliances": rng.uniform(10, 200, size=len(index)),
            "T1": rng.uniform(18, 24, size=len(index)),
            "RH_1": rng.uniform(30, 60, size=len(index)),
            "rv1": rng.uniform(0, 50, size=len(index)),
            "rv2": rng.uniform(0, 50, size=len(index)),
        },
        index=index,
    )


def test_clean_raw_data_drops_redundant_columns(raw_like_df):
    cleaned = D.clean_raw_data(raw_like_df)
    assert "rv1" not in cleaned.columns
    assert "rv2" not in cleaned.columns
    assert "Appliances" in cleaned.columns


def test_clean_raw_data_has_no_missing_target(raw_like_df):
    raw_like_df.loc[raw_like_df.index[5], "Appliances"] = np.nan
    cleaned = D.clean_raw_data(raw_like_df)
    assert cleaned["Appliances"].isna().sum() == 0
    assert len(cleaned) == len(raw_like_df) - 1


def test_resample_hourly_produces_hourly_index_with_no_gaps(raw_like_df):
    cleaned = D.clean_raw_data(raw_like_df)
    hourly = D.resample_hourly(cleaned)

    assert hourly.index.freqstr in ("h", "H")
    assert hourly.isna().sum().sum() == 0
    # 3 days of hourly data (allowing edge trimming from interpolate/dropna)
    assert 24 * 2 <= len(hourly) <= 24 * 3


def test_adf_and_kpss_return_expected_fields():
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    rng = np.random.default_rng(0)
    white_noise = pd.Series(rng.normal(0, 1, size=len(index)), index=index)

    adf_result = D.adf_test(white_noise, name="white_noise")
    kpss_result = D.kpss_test(white_noise, name="white_noise")

    assert "p_value" in adf_result
    assert "p_value" in kpss_result
    # White noise should be judged stationary by both tests.
    assert bool(adf_result["is_stationary_at_5pct"]) is True
    assert bool(kpss_result["is_stationary_at_5pct"]) is True
