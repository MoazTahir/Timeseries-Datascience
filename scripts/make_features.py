#!/usr/bin/env python3
"""
Build the full covariate feature table (Part 5) from the processed
hourly data and cache it to data/interim/feature_table.csv, without
running the rest of the pipeline.

Usage:
    python scripts/make_features.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appliance_energy import config, data as D, features as F


def main() -> None:
    config.ensure_directories()
    hourly = D.load_hourly_data()

    table = F.make_feature_table(hourly)
    out_path = config.INTERIM_DATA_DIR / "feature_table.csv"
    table.to_csv(out_path)

    print(f"Feature table shape: {table.shape}")
    print(f"Columns: {table.columns.tolist()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
