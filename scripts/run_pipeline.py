#!/usr/bin/env python3
"""
Main pipeline entry point.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --force-download
    python scripts/run_pipeline.py --refit-grid-search
    python scripts/run_pipeline.py --skip-foundation-model
"""

import argparse
import os
import sys
from pathlib import Path

# Cap every BLAS/OpenMP thread pool *before* numpy/xgboost/torch are
# imported (they size their pools from these env vars at import/first-use
# time, so setting them later has no effect). Without this, XGBoost's
# OpenMP thread pool and PyTorch's own thread pool both try to claim all
# CPU cores in the same process, which causes severe thread-contention
# thrashing on macOS (observed: the Chronos step taking 20+ minutes
# instead of ~1 minute for 14 rolling forecasts).
_N_THREADS = str(max(1, (os.cpu_count() or 4) // 2))
for _env_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_env_var, _N_THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Works around a separate libomp crash (SIGSEGV inside __kmp_suspend_64
# during a PyTorch op) seen when more than one OpenMP runtime ends up
# loaded in the same process on macOS - a known PyTorch/macOS issue, not
# specific to this project, that this env var is the standard mitigation
# for.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from appliance_energy.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full appliance energy forecasting pipeline.")
    parser.add_argument(
        "--force-download", action="store_true", help="Re-download the raw dataset even if cached."
    )
    parser.add_argument(
        "--refit-grid-search",
        action="store_true",
        help="Re-run the SARIMA AIC grid search instead of using cached results.",
    )
    parser.add_argument(
        "--skip-foundation-model",
        action="store_true",
        help="Skip the Chronos foundation model step (useful offline).",
    )
    args = parser.parse_args()

    run_pipeline(
        force_download=args.force_download,
        use_cached_grid_search=not args.refit_grid_search,
        include_foundation_model=not args.skip_foundation_model,
    )


if __name__ == "__main__":
    main()
