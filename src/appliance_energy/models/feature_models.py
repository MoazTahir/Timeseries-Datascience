"""
Feature-based machine-learning model (Part 6): XGBoost (primary) and
HistGradientBoostingRegressor (lightweight comparison), plus a
feature-group ablation study (Part 9, Q3).

Multi-step forecasting strategy
--------------------------------
A 24-hour-ahead forecast cannot simply call ``model.predict`` on the raw
test rows, because lag_1..lag_12 and the short rolling windows
(roll_*_3/6/12) for hours 2..24 of the horizon depend on Appliances
values that have not been observed yet at the forecast origin - they
fall *inside* the forecast horizon itself. (lag_24, lag_48, lag_168 and
roll_*_24/168 are always safe: they reference >= 24h in the past, which
is always outside a 24h horizon.)

``recursive_forecast`` therefore steps through the horizon one hour at a
time: it builds each hour's feature row from the best information
available (real history plus the model's own predictions for earlier
hours in the same horizon), predicts, appends that prediction to the
working history, and moves on. This is the textbook "recursive
multi-step forecasting" strategy and is the only leakage-safe way to
use short lags/rolling windows beyond a 1-step horizon.

Sensor/weather columns are treated as exogenous "known" inputs taken
from the realised test-set values (a *conditional* forecast - see
Part 9, Q5) - time features are always genuinely known in advance.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

from .. import config, features as F


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    # n_jobs is capped (not -1) rather than claiming every core: this
    # process also loads PyTorch for the foundation model later, and an
    # XGBoost OpenMP thread pool left holding all cores causes severe
    # thread-contention thrashing once PyTorch tries to start its own
    # pool in the same process (observed: ~1 minute of work taking 20+
    # minutes as a result).
    n_jobs = max(1, (os.cpu_count() or 4) // 2)
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config.RANDOM_STATE,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def fit_histgb(X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


FITTERS = {
    "xgboost": fit_xgboost,
    "histgb": fit_histgb,
}


def recursive_forecast(
    model,
    history: pd.Series,
    exog_future: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    lags: list[int] = config.LAG_HOURS,
    windows: list[int] = config.ROLLING_WINDOWS,
    target: str = config.TARGET,
) -> pd.Series:
    """
    Recursive multi-step forecast for one rolling-origin window.

    Parameters
    ----------
    model:
        A fitted regressor with .predict(X) -> array.
    history:
        Series of *actual* target values up to (not including) the
        forecast origin. Must contain at least max(lags/windows) points.
    exog_future:
        Dataframe indexed by the horizon's timestamps, containing every
        non-target feature that is NOT derived from the target itself
        (time features + sensor/weather columns). These are treated as
        known at forecast time (see module docstring re: conditional
        forecast for sensor/weather).
    feature_cols:
        Exact column order the model was trained on.

    Returns
    -------
    pd.Series of length ``horizon``, indexed like ``exog_future``.
    """
    working_history = history.copy()
    predictions = []

    for ts in exog_future.index:
        row = {}

        for lag in lags:
            row[f"lag_{lag}"] = working_history.iloc[-lag]

        shifted = working_history  # already "up to but not including now"
        for window in windows:
            recent = shifted.iloc[-window:]
            row[f"roll_mean_{window}"] = recent.mean()
            row[f"roll_std_{window}"] = recent.std()

        for col in exog_future.columns:
            row[col] = exog_future.loc[ts, col]

        X_row = pd.DataFrame([row], index=[ts])[feature_cols]
        pred = float(model.predict(X_row)[0])
        predictions.append(pred)

        working_history.loc[ts] = pred

    return pd.Series(predictions, index=exog_future.index, name="feature_model")


def rolling_forecast_feature_model(
    df: pd.DataFrame,
    fit_fn,
    groups: list[str] | None = None,
    n_test: int = config.TEST_STEPS,
    horizon: int = config.HORIZON,
    target: str = config.TARGET,
    refit_each_origin: bool = False,
):
    """
    Rolling-origin 24-hour-ahead feature-model forecast across the test
    period.

    The model is trained once on the initial training window (fast:
    gradient-boosted trees train in seconds, so ``refit_each_origin=True``
    is also offered for a more realistic "retrain daily" variant, but
    defaults to False to keep the primary run fast and directly
    comparable to the SARIMAX rolling protocol).

    Parameters
    ----------
    df: full feature table (output of features.make_feature_table),
        target column + all engineered/raw feature columns, covering
        the whole series (train + test).
    fit_fn: one of FITTERS.values(), e.g. fit_xgboost.
    groups: feature groups to include (see features.FEATURE_GROUPS);
        None = all groups. Used for the ablation study.

    Returns
    -------
    (forecast, fitted_model, feature_cols)
    """
    from ..evaluation import rolling_origin_splits

    if groups is None:
        groups = list(F.FEATURE_GROUPS.keys())

    feature_cols = [c for c in df.columns if c != target]
    time_cols = [c for c in F.FEATURE_GROUPS["time"] if c in feature_cols]
    lag_cols = [c for c in F.FEATURE_GROUPS["lags"] if c in feature_cols]
    roll_cols = [c for c in F.FEATURE_GROUPS["rolling"] if c in feature_cols]
    sensor_weather_cols = [
        c
        for c in feature_cols
        if c not in time_cols and c not in lag_cols and c not in roll_cols
    ]

    splits = rolling_origin_splits(len(df), n_test=n_test, horizon=horizon)

    all_preds = []
    model = None

    for origin_i, (train_end, test_end) in enumerate(splits):
        train_df = df.iloc[:train_end]
        X_train = train_df[feature_cols]
        y_train = train_df[target]

        if model is None or refit_each_origin:
            model = fit_fn(X_train, y_train)

        index = df.index[train_end:test_end]
        # exog_future: time + sensor/weather columns for the horizon,
        # taken from realised values (conditional forecast, see docstring).
        exog_future = df.loc[index, time_cols + sensor_weather_cols]

        # history of actual target values available up to the origin
        history = df[target].iloc[:train_end]

        pred = recursive_forecast(
            model=model,
            history=history,
            exog_future=exog_future,
            horizon=horizon,
            feature_cols=feature_cols,
        )
        all_preds.append(pred)

    forecast = pd.concat(all_preds).sort_index()
    return forecast, model, feature_cols


def feature_importance(model, feature_cols: list[str]) -> pd.Series:
    """Return a Series of feature importances indexed by feature name, sorted desc."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model has no feature_importances_ attribute")

    return pd.Series(importances, index=feature_cols).sort_values(ascending=False)


def run_feature_group_ablation(
    hourly: pd.DataFrame,
    fit_fn,
    y_train: pd.Series,
    test: pd.Series,
    ablation_groups: list[list[str]] | None = None,
) -> pd.DataFrame:
    """
    Fit + rolling-forecast the feature model on progressively richer
    feature sets and report metrics for each, to directly answer
    "which feature groups appear most useful" (Part 9, Q3).

    Default ablation ladder: lags -> +rolling -> +time -> +sensor+weather.
    """
    from ..evaluation import evaluate_forecast

    if ablation_groups is None:
        ablation_groups = [
            ["lags"],
            ["lags", "rolling"],
            ["lags", "rolling", "time"],
            ["lags", "rolling", "time", "sensor", "weather"],
        ]

    rows = []
    for groups in ablation_groups:
        table = F.make_feature_table(hourly, groups=groups)
        forecast, _, _ = rolling_forecast_feature_model(table, fit_fn, groups=groups)
        label = "+".join(groups)
        rows.append(evaluate_forecast(label, test, forecast, y_train))

    return pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)
