"""
Time-series foundation model (Part 7): Chronos.

We use Amazon's Chronos-Bolt (chronos-forecasting package), a
pretrained probabilistic time-series foundation model, in pure
zero-shot mode - the model is never fine-tuned or shown any appliance
energy data during training, only at inference time as a numeric
context window. This is the "easiest to use in class" option the
assignment brief points to: it runs fully locally on CPU (no API key,
unlike TimeGPT which needs a Nixtla account), unlike TimesFM which has
heavier setup requirements.

Chronos-Bolt is target-only: it forecasts the univariate Appliances
series directly from its own history and has no mechanism to condition
on exogenous covariates (indoor/outdoor sensors, time features). This
is an important, explicit limitation for the Part 9 discussion of
whether covariates matter and whether the foundation model is
"worth" its complexity.

If torch/chronos-forecasting is unavailable, or the pretrained weights
cannot be downloaded (e.g. no internet at grading time), forecasting
falls back to the daily seasonal-naive benchmark with a clear warning,
so the rest of the pipeline still runs end-to-end.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

from .. import config

CHRONOS_MODEL_NAME = "amazon/chronos-bolt-tiny"

_pipeline_cache = {}


def _load_pipeline():
    """
    Lazily load (and cache) the Chronos pipeline so it is only downloaded
    once per process.

    Tries a fully offline load first (``local_files_only=True``): once the
    weights are cached under ~/.cache/huggingface, this never touches the
    network at all, which avoids a real failure mode observed during
    development - huggingface_hub's routine "is there a newer version?"
    metadata check can hang indefinitely on a slow/unstable connection even
    though the model is already cached locally and no download is actually
    needed. Only falls back to a normal (network-enabled) load if nothing
    is cached yet.
    """
    if "pipeline" in _pipeline_cache:
        return _pipeline_cache["pipeline"]

    import torch
    from chronos import BaseChronosPipeline

    # Explicitly cap PyTorch's thread pool. It is sized lazily on first
    # use, so if XGBoost's OpenMP thread pool (see feature_models.py) is
    # still holding every core when torch starts up in the same process,
    # the two compete for CPU in a way that can make a ~1-minute workload
    # take 20+ minutes on macOS. scripts/run_pipeline.py also sets the
    # OMP/MKL env vars before either library is imported; this is a
    # second, explicit line of defence for anyone calling this module
    # directly (e.g. from a notebook) without going through that script.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL_NAME,
            device_map="cpu",
            torch_dtype="float32",
            local_files_only=True,
        )
    except Exception:
        pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL_NAME,
            device_map="cpu",
            torch_dtype="float32",
        )
    _pipeline_cache["pipeline"] = pipeline
    return pipeline


def chronos_available() -> bool:
    try:
        _load_pipeline()
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Chronos foundation model unavailable ({exc}); will fall back.")
        return False


def chronos_forecast(
    y_train: pd.Series,
    horizon: int,
    index: pd.DatetimeIndex,
    context_length: int = 512,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Zero-shot Chronos forecast for one rolling-origin window.

    Parameters
    ----------
    y_train:
        Target history up to the forecast origin. Only the last
        ``context_length`` points are given to the model as context
        (Chronos-Bolt's practical/effective context window).
    horizon:
        Number of steps ahead to forecast.

    Returns
    -------
    (median_forecast, quantile_df) where quantile_df has columns for
    the 10th and 90th percentile (used as an approximate uncertainty
    band, analogous to the SARIMAX confidence interval).
    """
    import torch

    pipeline = _load_pipeline()

    context = torch.tensor(y_train.values[-context_length:], dtype=torch.float32)

    quantiles, mean = pipeline.predict_quantiles(
        inputs=context,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    quantiles = quantiles[0].numpy()  # shape (horizon, 3)

    median_forecast = pd.Series(quantiles[:, 1], index=index, name="foundation_model")
    quantile_df = pd.DataFrame(
        {"lower": quantiles[:, 0], "upper": quantiles[:, 2]}, index=index
    )

    return median_forecast, quantile_df


def rolling_forecast_foundation(
    y: pd.Series,
    n_test: int = config.TEST_STEPS,
    horizon: int = config.HORIZON,
    context_length: int = 512,
):
    """
    Rolling-origin 24-hour-ahead Chronos forecast across the test period.
    Falls back to daily seasonal-naive (with a warning) if Chronos is
    unavailable in the current environment.

    Returns
    -------
    (forecast, quantile_df, used_fallback: bool)
    """
    from ..evaluation import rolling_origin_splits

    if not chronos_available():
        from .benchmarks import seasonal_naive_forecast

        splits = rolling_origin_splits(len(y), n_test=n_test, horizon=horizon)
        preds = []
        for train_end, test_end in splits:
            y_train = y.iloc[:train_end]
            idx = y.index[train_end:test_end]
            preds.append(
                seasonal_naive_forecast(y_train, horizon, idx, config.DAILY_PERIOD)
            )
        forecast = pd.concat(preds).sort_index().rename("foundation_model")
        return forecast, None, True

    splits = rolling_origin_splits(len(y), n_test=n_test, horizon=horizon)

    all_preds = []
    all_lower = []
    all_upper = []

    for origin_i, (train_end, test_end) in enumerate(splits):
        y_train = y.iloc[:train_end]
        index = y.index[train_end:test_end]

        pred, qdf = chronos_forecast(y_train, horizon, index, context_length=context_length)
        all_preds.append(pred)
        all_lower.append(qdf["lower"])
        all_upper.append(qdf["upper"])
        print(f"  Chronos rolling forecast: origin {origin_i + 1}/{len(splits)} done")

    forecast = pd.concat(all_preds).sort_index().rename("foundation_model")
    quantile_df = pd.DataFrame(
        {"lower": pd.concat(all_lower).sort_index(), "upper": pd.concat(all_upper).sort_index()}
    )

    return forecast, quantile_df, False
