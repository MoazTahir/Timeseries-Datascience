"""
Tests for src/appliance_energy/models/sarimax.py.

Deliberately does not fit a real SARIMAX model (too slow for a unit
test); focuses on the pure-logic pieces: order selection from a grid
search table and the Ljung-Box wrapper's output shape.
"""

import numpy as np
import pandas as pd

from appliance_energy.models import sarimax as S


def test_best_order_from_grid_picks_lowest_aic():
    grid_df = pd.DataFrame(
        {
            "p": [0, 1, 2, 3],
            "d": [1, 1, 1, 1],
            "q": [0, 1, 0, 1],
            "aic": [500.0, 480.0, np.nan, 490.0],
            "bic": [510.0, 495.0, np.nan, 505.0],
            "converged": [True, True, False, True],
        }
    )

    order = S.best_order_from_grid(grid_df)
    assert order == (1, 1, 1)


def test_best_order_from_grid_ignores_nan_aic_rows():
    grid_df = pd.DataFrame(
        {
            "p": [5, 6],
            "d": [2, 2],
            "q": [5, 6],
            "aic": [np.nan, 100.0],
            "bic": [np.nan, 110.0],
            "converged": [False, True],
        }
    )

    order = S.best_order_from_grid(grid_df)
    assert order == (6, 2, 6)


def test_ljung_box_test_returns_one_row_per_lag():
    rng = np.random.default_rng(0)
    residuals = pd.Series(rng.normal(0, 1, size=500))

    result = S.ljung_box_test(residuals, lags=[24, 48])

    assert len(result) == 2
    assert "lb_pvalue" in result.columns
