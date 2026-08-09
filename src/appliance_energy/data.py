"""
Data acquisition and preparation (Part 1 of the assignment).

Responsibilities:
    - download the raw Appliances Energy Prediction dataset from UCI
    - parse the timestamp and set it as a DatetimeIndex
    - check / report missing values
    - resample the 10-minute data to hourly means
    - run the full battery of stationarity diagnostics (ADF, KPSS,
      ACF/PACF, seasonal decomposition, differencing)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import urllib.request
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config


# ------------------------------------------------------------------
# Download and load
# ------------------------------------------------------------------

def download_raw_data(force: bool = False) -> pd.DataFrame:
    """
    Download the raw 10-minute Appliances Energy Prediction dataset from
    the UCI repository and cache it under data/raw/.

    Parameters
    ----------
    force:
        If True, re-download even if a cached copy already exists.

    Returns
    -------
    Raw dataframe, unmodified apart from parsing ``date`` as a
    DatetimeIndex.
    """
    config.ensure_directories()

    if force or not config.RAW_DATA_PATH.exists():
        print(f"Downloading data from {config.RAW_DATA_URL} ...")
        urllib.request.urlretrieve(config.RAW_DATA_URL, config.RAW_DATA_PATH)
        print(f"Saved raw data to {config.RAW_DATA_PATH}")
    else:
        print(f"Using cached raw data at {config.RAW_DATA_PATH}")

    df = pd.read_csv(config.RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary table of missing values per column."""
    n_missing = df.isna().sum()
    pct_missing = 100 * n_missing / len(df)

    summary = (
        pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
        .sort_values("n_missing", ascending=False)
    )

    return summary


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all columns to numeric, drop the two constant/near-useless
    columns from the original dataset (`rv1`, `rv2` are random noise
    variables included by the original authors as a sanity check, and
    `NSM`/`WeekStatus`/`Day_of_week` duplicate information already
    derivable from the timestamp index), and drop rows with a missing
    target.
    """
    out = df.copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    redundant_cols = [
        c for c in ["rv1", "rv2", "NSM", "WeekStatus", "Day_of_week"]
        if c in out.columns
    ]
    out = out.drop(columns=redundant_cols)

    out = out.dropna(subset=[config.TARGET])

    return out


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the 10-minute data up to hourly means (Part 1).

    Small gaps introduced by resampling (e.g. an hour with no 10-minute
    readings) are filled by time-weighted interpolation; any residual
    missing rows at the very start/end of the series are dropped.
    """
    hourly = df.resample(config.RESAMPLE_FREQ).mean()
    hourly = hourly.interpolate(method="time")
    hourly = hourly.dropna()

    return hourly


def load_hourly_data(force_download: bool = False) -> pd.DataFrame:
    """
    End-to-end loader: download raw data (if needed), clean it, resample
    to hourly, cache the processed series, and return it.
    """
    config.ensure_directories()

    if not force_download and config.HOURLY_DATA_PATH.exists():
        print(f"Loading cached hourly data from {config.HOURLY_DATA_PATH}")
        hourly = pd.read_csv(config.HOURLY_DATA_PATH, index_col=0, parse_dates=True)
        return hourly

    raw = download_raw_data(force=force_download)
    clean = clean_raw_data(raw)
    hourly = resample_hourly(clean)

    hourly.to_csv(config.HOURLY_DATA_PATH)
    print(f"Saved hourly data ({hourly.shape[0]} rows) to {config.HOURLY_DATA_PATH}")

    return hourly


# ------------------------------------------------------------------
# Stationarity diagnostics (Part 1)
# ------------------------------------------------------------------

def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Augmented Dickey-Fuller test.

    H0: the series has a unit root (is non-stationary).
    A small p-value (< 0.05) lets us reject H0, i.e. evidence of
    stationarity.
    """
    series = series.dropna()
    stat, pvalue, used_lag, n_obs, crit_values, _ = adfuller(series, autolag="AIC")

    return {
        "series": name,
        "test": "ADF",
        "statistic": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "crit_1pct": crit_values["1%"],
        "crit_5pct": crit_values["5%"],
        "crit_10pct": crit_values["10%"],
        "is_stationary_at_5pct": pvalue < 0.05,
    }


def kpss_test(series: pd.Series, name: str = "", regression: str = "c") -> dict:
    """
    KPSS test.

    H0: the series is (trend-)stationary. A small p-value (< 0.05) lets
    us reject H0, i.e. evidence of non-stationarity. Used alongside the
    ADF test since the two tests have opposite null hypotheses -
    agreement between them gives more confidence in the conclusion.
    """
    series = series.dropna()
    stat, pvalue, n_lags, crit_values = kpss(series, regression=regression, nlags="auto")

    return {
        "series": name,
        "test": "KPSS",
        "statistic": stat,
        "p_value": pvalue,
        "n_lags": n_lags,
        "crit_1pct": crit_values["1%"],
        "crit_5pct": crit_values["5%"],
        "crit_10pct": crit_values["10%"],
        "is_stationary_at_5pct": pvalue >= 0.05,
    }


def run_stationarity_suite(series: pd.Series, name: str = "series") -> pd.DataFrame:
    """
    Run ADF + KPSS on the raw series, on the first difference, and on the
    seasonal (24-hour) difference, so both trend-type and seasonal-type
    non-stationarity are diagnosed. Returns a single tidy table.
    """
    series = series.astype(float)

    variants = {
        f"{name} (level)": series,
        f"{name} (first diff)": series.diff().dropna(),
        f"{name} (seasonal diff, lag={config.DAILY_PERIOD})": series.diff(
            config.DAILY_PERIOD
        ).dropna(),
        f"{name} (first + seasonal diff)": series.diff(config.DAILY_PERIOD)
        .diff()
        .dropna(),
    }

    rows = []
    for label, s in variants.items():
        rows.append(adf_test(s, name=label))
        rows.append(kpss_test(s, name=label))

    return pd.DataFrame(rows)


def compute_acf_pacf(series: pd.Series, nlags: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Return (ACF, PACF) values up to ``nlags`` for plotting/diagnosis."""
    series = series.dropna()
    acf_vals = acf(series, nlags=nlags, fft=True)
    pacf_vals = pacf(series, nlags=nlags)
    return acf_vals, pacf_vals


def decompose_series(
    series: pd.Series, period: int = config.DAILY_PERIOD, method: str = "stl"
):
    """
    Decompose the series into trend, seasonal, and residual components.

    method="stl" uses STL (robust to non-sinusoidal/changing seasonal
    shape, generally preferred for real-world data); method="classical"
    uses the simpler additive seasonal_decompose taught as a baseline.
    """
    series = series.astype(float).dropna()

    if method == "stl":
        result = STL(series, period=period, robust=True).fit()
    else:
        result = seasonal_decompose(series, model="additive", period=period)

    return result
