#!/usr/bin/env python3
"""
Download and prepare the Appliances Energy Prediction dataset only
(Part 1), without running the rest of the pipeline. Useful for a quick
first check that data access and resampling work in a fresh environment.

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appliance_energy import data as D


def main() -> None:
    hourly = D.load_hourly_data(force_download=True)
    print(f"\nHourly data shape: {hourly.shape}")
    print(f"Date range: {hourly.index.min()} to {hourly.index.max()}")
    print("\nMissing values:")
    print(D.report_missing_values(hourly))


if __name__ == "__main__":
    main()
