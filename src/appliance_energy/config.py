"""
Central configuration for the appliance energy forecasting project.

This module defines the forecasting problem (Part 2 of the assignment):
target variable, forecast horizon, train/test split, seasonal periods,
and file-system locations. All other modules import their constants
from here so the problem definition only lives in one place.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)
RAW_DATA_PATH = RAW_DATA_DIR / "energydata_complete.csv"
HOURLY_DATA_PATH = PROCESSED_DATA_DIR / "appliance_hourly.csv"

ALL_DIRS = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODEL_DIR,
]


def ensure_directories() -> None:
    """Create every project directory the pipeline writes to, if missing."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Forecasting problem definition (Part 2)
# ------------------------------------------------------------------

# Target variable: total appliance energy use (Wh), aggregated to hourly means.
TARGET = "Appliances"

# Resampling frequency. The raw data are sampled every 10 minutes; the
# assignment recommends resampling to hourly values to keep SARIMAX and
# the grid search tractable while still resolving daily/weekly seasonality.
RESAMPLE_FREQ = "h"

# Seasonal periods once resampled to hourly data.
DAILY_PERIOD = 24        # 24 hours in a day
WEEKLY_PERIOD = 24 * 7   # 168 hours in a week

# Forecast horizon: forecast the next 24 hours of appliance energy use.
HORIZON = 24

# Test period: the final 14 days of the (hourly) series are held out.
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * DAILY_PERIOD  # 336 hourly observations

# Number of rolling-origin windows spanning the test period at HORIZON steps
# each. With a 14-day test set and a 24-hour horizon this gives 14 origins,
# i.e. the model re-forecasts "the next day" once per day across the held-out
# fortnight rather than making a single 336-step-ahead forecast. This keeps
# every model evaluated on genuine 24-hour-ahead forecasts (as instructed)
# while still covering the full 14-day test window.
N_ROLLING_ORIGINS = TEST_STEPS // HORIZON

RANDOM_STATE = 0

# ------------------------------------------------------------------
# Candidate exogenous / covariate columns
# ------------------------------------------------------------------

# Sensor + weather columns available in the raw dataset that are candidate
# exogenous regressors for SARIMAX and features for the ML model.
INDOOR_TEMP_COLS = [f"T{i}" for i in range(1, 10)]
INDOOR_HUMIDITY_COLS = [f"RH_{i}" for i in range(1, 10)]
OUTDOOR_WEATHER_COLS = [
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]

# Exogenous variables used for the SARIMAX model. Kept small and physically
# motivated (outdoor conditions that plausibly drive heating/cooling and
# appliance use) rather than throwing in every sensor column, which would
# risk overfitting/multicollinearity (many T*/RH_* indoor sensors are highly
# correlated with each other and with Appliances itself).
SARIMAX_EXOG_COLS = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]

# Lags (in hours) used for the feature-based model's lag features.
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]

# Rolling window sizes (in hours) used for rolling mean/std features.
ROLLING_WINDOWS = [3, 6, 12, 24, 168]

# SARIMA/SARIMAX AIC grid-search ranges (Part 4): p in [0,6], d in [0,2],
# q in [0,6], as specified in the assignment brief.
P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)

# Seasonal order used for SARIMAX. Chosen by reasoning from the seasonal
# decomposition / ACF analysis (see notebooks/02) rather than grid-searched
# jointly with (p, d, q): a joint search of P, D, Q at 3 candidate values
# each would multiply the 147-model (p, d, q) grid by up to 27x, which is
# computationally unreasonable for a statsmodels state-space model fit on
# ~3000 hourly points. D=1 seasonal differencing removes the strong daily
# cycle visible in the ACF; P=Q=1 captures the remaining seasonal
# autocorrelation at lag 24 parsimoniously.
SEASONAL_ORDER = (1, 1, 1, DAILY_PERIOD)
