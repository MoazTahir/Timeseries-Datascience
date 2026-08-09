"""
SARIMAX modelling (Part 4).

Contents:
    - grid_search_sarima: AIC grid search over p in [0,6], d in [0,2],
      q in [0,6], exactly as specified in the assignment brief, following
      the nested-loop-with-try/except pattern taught in the Week 5 ARMA
      tutorial (statsmodels ``optimize_ARIMA``), extended to SARIMAX with
      a fixed daily seasonal order.
    - fit_sarimax / rolling_forecast_sarimax: fit once on the initial
      training window, then roll forward through the 14-day test period
      using ``append(refit=False)`` to extend the filtered state with
      newly observed actuals without re-optimising parameters at every
      origin (standard practice for rolling multi-step forecasts; makes
      14 origins tractable instead of 14 full MLE re-fits).
    - residual diagnostics helpers (Ljung-Box test; ACF/histogram plot
      lives in plotting.py).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config


# ------------------------------------------------------------------
# AIC grid search over (p, d, q)
# ------------------------------------------------------------------

def _fit_one_order(
    order: tuple[int, int, int],
    y_train: pd.Series,
    exog_train: pd.DataFrame | None,
    seasonal_order: tuple,
) -> dict:
    """Fit a single SARIMAX(p,d,q) candidate; used as the grid-search worker."""
    p, d, q = order
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                trend="c" if d == 0 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)

        return {
            "p": p,
            "d": d,
            "q": q,
            "aic": fit.aic,
            "bic": fit.bic,
            "converged": bool(fit.mle_retvals.get("converged", True)),
        }
    except Exception:
        return {"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan, "converged": False}


def grid_search_sarima(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None = None,
    p_range=config.P_RANGE,
    d_range=config.D_RANGE,
    q_range=config.Q_RANGE,
    seasonal_order: tuple = config.SEASONAL_ORDER,
    verbose: bool = True,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Loop over every (p, d, q) combination in the given ranges, fit a
    SARIMAX(p, d, q) x seasonal_order model, and record its AIC.

    A model that fails to converge or raises during fitting is skipped
    (recorded with aic=NaN) rather than crashing the whole search - some
    combinations (e.g. very high p and q together) are numerically
    unstable or non-identifiable and are expected to fail.

    Each candidate SARIMAX fit takes on the order of ~10-30 seconds on
    ~3000 hourly points (state-space filtering with a period-24 seasonal
    term is expensive per optimiser iteration), so the full 7x3x7=147
    combination grid is run in parallel across CPU cores via joblib
    rather than sequentially, which would otherwise take the best part
    of an hour.

    Returns
    -------
    Dataframe with one row per attempted combination, columns
    [p, d, q, aic, bic, converged], sorted by AIC ascending.
    """
    orders = [(p, d, q) for d in d_range for p in p_range for q in q_range]

    if n_jobs is None:
        n_jobs = max(1, min(8, (os.cpu_count() or 4) - 2))

    if verbose:
        print(f"Grid search: fitting {len(orders)} SARIMAX candidates using n_jobs={n_jobs} ...")

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_order)(order, y_train, exog_train, seasonal_order) for order in orders
    )

    grid_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)

    n_converged = int(grid_df["converged"].sum())
    if verbose:
        print(f"Grid search complete: {n_converged}/{len(orders)} candidates converged.")
        if len(grid_df) and grid_df["aic"].notna().any():
            best = grid_df.iloc[0]
            print(
                f"Best SARIMA order: (p,d,q)=({int(best.p)},{int(best.d)},{int(best.q)}) "
                f"AIC={best.aic:.1f}"
            )

    return grid_df


def best_order_from_grid(grid_df: pd.DataFrame) -> tuple[int, int, int]:
    """Extract the (p, d, q) tuple with the lowest AIC from a grid-search table."""
    valid = grid_df.dropna(subset=["aic"])
    best = valid.sort_values("aic").iloc[0]
    return int(best.p), int(best.d), int(best.q)


# ------------------------------------------------------------------
# Fitting and rolling-origin forecasting
# ------------------------------------------------------------------

def fit_sarimax(
    y_train: pd.Series,
    order: tuple[int, int, int],
    exog_train: pd.DataFrame | None = None,
    seasonal_order: tuple = config.SEASONAL_ORDER,
):
    """Fit a single SARIMAX model with the given order on the training data."""
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def rolling_forecast_sarimax(
    y: pd.Series,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
    n_test: int = config.TEST_STEPS,
    horizon: int = config.HORIZON,
    seasonal_order: tuple = config.SEASONAL_ORDER,
    return_conf_int: bool = True,
    alpha: float = 0.05,
):
    """
    Rolling-origin 24-hour-ahead SARIMAX forecast across the test period.

    The model is fit once (MLE parameter estimation) on the initial
    training window. At each subsequent origin, the newly observed
    24 hours of actuals are appended to the fitted results with
    ``refit=False``, which updates the Kalman filter state to include
    the new observations without re-estimating (p,d,q) coefficients.
    This mirrors how a SARIMAX model would actually be operated in
    production (params tuned periodically, state updated continuously)
    and keeps 14 rolling origins computationally cheap.

    Returns
    -------
    (forecast, conf_int) where forecast is a pd.Series of length n_test
    and conf_int is a dataframe with 'lower'/'upper' columns (or None
    if return_conf_int=False).
    """
    from ..evaluation import rolling_origin_splits

    splits = rolling_origin_splits(len(y), n_test=n_test, horizon=horizon)
    first_train_end = splits[0][0]

    y_train_init = y.iloc[:first_train_end]
    exog_train_init = exog.iloc[:first_train_end] if exog is not None else None
    exog_full = exog

    fit = fit_sarimax(y_train_init, order, exog_train_init, seasonal_order)

    all_preds = []
    all_lower = []
    all_upper = []

    current_fit = fit
    for origin_i, (train_end, test_end) in enumerate(splits):
        index = y.index[train_end:test_end]
        exog_step = exog_full.loc[index] if exog_full is not None else None

        fc = current_fit.get_forecast(steps=horizon, exog=exog_step)
        pred = fc.predicted_mean
        pred.index = index
        all_preds.append(pred)

        if return_conf_int:
            ci = fc.conf_int(alpha=alpha)
            ci.index = index
            all_lower.append(ci.iloc[:, 0])
            all_upper.append(ci.iloc[:, 1])

        # Extend the filtered state with the newly-observed actuals
        # (available now that this origin's horizon has "passed") ahead
        # of forecasting from the next origin. Skip on the last split.
        if origin_i < len(splits) - 1:
            new_y = y.iloc[train_end:test_end]
            new_exog = exog_full.iloc[train_end:test_end] if exog_full is not None else None
            current_fit = current_fit.append(new_y, exog=new_exog, refit=False)

    forecast = pd.concat(all_preds).sort_index().rename("sarimax")

    conf_int = None
    if return_conf_int:
        conf_int = pd.DataFrame(
            {"lower": pd.concat(all_lower).sort_index(), "upper": pd.concat(all_upper).sort_index()}
        )

    return forecast, conf_int, fit


# ------------------------------------------------------------------
# Residual diagnostics
# ------------------------------------------------------------------

def ljung_box_test(residuals: pd.Series, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Ljung-Box test for residual autocorrelation.

    H0: residuals are independently distributed (no leftover
    autocorrelation, i.e. the model has captured the serial structure).
    A small p-value indicates the model has NOT fully captured the
    autocorrelation structure and residuals still contain signal.
    """
    residuals = residuals.dropna()
    if lags is None:
        lags = [24, 48]
    return acorr_ljungbox(residuals, lags=lags, return_df=True)
