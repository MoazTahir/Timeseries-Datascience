#!/usr/bin/env python3
"""
Run the Chronos foundation-model rolling forecast (Part 7) and save its
output to outputs/forecasts/foundation_forecast.csv.

Why this runs as its own process
---------------------------------
On macOS, XGBoost's OpenMP thread pool and PyTorch's thread pool
deadlock when both libraries are initialised in the same Python process
(observed during development: a ~4-second workload hanging indefinitely
once XGBoost had already been fit earlier in the same process - thread
count env vars alone did not resolve it). Rather than depend on a
fragile fix for a third-party threading interaction, `pipeline.py`
invokes this script as a fresh subprocess for the foundation-model step,
so PyTorch never has to share a process with XGBoost's thread pool.

Usage:
    python scripts/run_foundation_model.py
"""

import os

# Must be set before torch is imported (either here or transitively).
# Works around a libomp crash (SIGSEGV inside __kmp_suspend_64 during a
# PyTorch op) seen when more than one OpenMP runtime ends up loaded in
# the same process on macOS - a known PyTorch/macOS issue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appliance_energy import config, data as D
from appliance_energy.models import foundation as FD


def main() -> None:
    config.ensure_directories()

    hourly = D.load_hourly_data()
    y = hourly[config.TARGET]

    forecast, quantile_df, used_fallback = FD.rolling_forecast_foundation(y)

    out = forecast.to_frame("foundation_model")
    if quantile_df is not None:
        out["foundation_lower"] = quantile_df["lower"]
        out["foundation_upper"] = quantile_df["upper"]

    out_path = config.FORECAST_DIR / "foundation_forecast.csv"
    out.to_csv(out_path)

    print(f"used_fallback={used_fallback}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
