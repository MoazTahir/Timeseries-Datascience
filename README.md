# Appliance Energy Forecasting

A reproducible time-series forecasting pipeline for household appliance energy use, built for the Data Analysis with AI (Semester C) Time Series Case Study assignment.

The project compares five simple benchmark forecasts, a SARIMAX model, a feature-based gradient-boosting model (XGBoost / HistGradientBoosting), and a zero-shot time-series foundation model (Amazon Chronos-Bolt) on the same rolling-origin, 24-hour-ahead forecasting task, using the **Appliances Energy Prediction** dataset (indoor/outdoor sensor readings alongside appliance energy use).

## Project aim

Forecast the next 24 hours of household appliance energy use and evaluate whether increasingly complex models actually improve on simple benchmarks. The six questions this project answers (Part 9 of the assignment, answered in full with real numbers in `notebooks/07_model_comparison.ipynb`):

1. Which benchmark model is strongest, and what does that reveal about the structure of appliance energy use?
2. Does SARIMAX improve on the strongest seasonal benchmark?
3. Does the feature-based model improve as lag, rolling-window, time, and sensor/weather features are added? Which feature groups actually help?
4. Does the foundation model outperform the simpler benchmark, SARIMAX, and feature-based models - and is any improvement worth the extra complexity?
5. Which covariates would genuinely be known at the forecast origin (true forecast vs. conditional forecast)?
6. Which model is recommended for practical smart-home energy forecasting, and why?

## Headline result

Run on the actual 14-day held-out test period (see `outputs/metrics/model_comparison.csv`):

| Rank | Model | MASE | RMSE |
|---|---|---|---|
| 1 | **Chronos (zero-shot foundation model)** | **0.649** | 68.3 |
| 2 | SARIMAX (target-only) | 0.676 | 64.7 |
| 3 | SARIMAX (+ exogenous weather) | 0.687 | 64.8 |
| 4 | Seasonal-naive, weekly (strongest benchmark) | 0.813 | 81.4 |
| 5 | XGBoost (full feature set) | 0.849 | 65.6 |
| 6 | HistGradientBoosting (full feature set) | 0.863 | 67.0 |
| 7 | Seasonal-naive, daily | 0.904 | 85.6 |
| 8 | Mean | 0.941 | 74.9 |
| 9 | Naive | 1.601 | 110.4 |
| 10 | Drift | 1.606 | 110.7 |

Two results are genuinely counter-intuitive and are discussed at length in the notebooks/report: **adding exogenous weather covariates makes SARIMAX slightly *worse*, not better**, and **the full-featured XGBoost model is the *worst* of four feature-ablation variants** - a reduced `lags + rolling + time` feature set scores MASE 0.756, clearly better than the full-featured 0.849 (see `outputs/metrics/feature_ablation.csv` and notebook 05). Model complexity does not track accuracy monotonically on this dataset.

## Dataset

**Appliances Energy Prediction** (Candanedo, Feldheim & Deramaix, 2017), downloaded directly from the UCI Machine Learning Repository:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
```

Sampled every 10 minutes; resampled to hourly means for this project (Part 1) to keep the SARIMAX grid search and the 24-hour forecast horizon tractable. ~3,290 hourly rows, 2016-01-11 to 2016-05-27.

Target variable: `Appliances` (hourly-mean appliance energy use, Wh).

Other columns: `lights`, indoor temperature/humidity sensors `T1..T9`/`RH_1..RH_9`, and outdoor weather `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`. (The original dataset's `rv1`, `rv2` random-noise sanity-check columns and the redundant `NSM`/`WeekStatus`/`Day_of_week` columns - all directly derivable from the timestamp - are dropped during cleaning.)

## Forecasting problem definition (Part 2)

| | |
|---|---|
| **Target** | `Appliances` |
| **Forecast horizon** | 24 hours |
| **Train / test split** | Final 14 days (336 hourly rows) held out as test |
| **Evaluation protocol** | Rolling-origin: 14 separate 24-hour-ahead forecasts, one per test day, expanding training window |
| **Metrics** | MAE, RMSE, MASE (scaled to the in-sample daily seasonal-naive error), Bias |

A rolling-origin backtest (not a single 336-step-ahead forecast) is used throughout: every model re-forecasts 24 hours ahead once per day across the held-out fortnight, which satisfies the assignment's 24-hour-horizon instruction literally while still covering the whole test period, and mirrors how a deployed forecaster would actually be operated. See `src/appliance_energy/evaluation.py::run_rolling_backtest` / `rolling_origin_splits`.

## Models

### 1. Benchmarks (Part 3)
Mean, naive, daily seasonal-naive (lag 24), weekly seasonal-naive (lag 168), drift. `src/appliance_energy/models/benchmarks.py`.

### 2. SARIMAX (Part 4)
AIC grid search over `p ∈ [0,6]`, `d ∈ [0,2]`, `q ∈ [0,6]` (147 combinations, parallelised across CPU cores via `joblib` - ~13 minutes; run once and cached to `outputs/metrics/sarima_grid_search.csv`), following the nested-loop/try-except/AIC-comparison pattern from the Week 5 ARMA tutorial. Selected order: **(6, 0, 0)** with a fixed seasonal order **(1, 1, 1, 24)** reasoned from the seasonal decomposition/ACF rather than jointly grid-searched (a joint search would multiply the 147-model grid by up to ~27x - see the reasoning in `src/appliance_energy/config.py`).

Fit both target-only and with exogenous weather covariates (`T_out`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`). Rolling-origin forecasts use `.append(refit=False)` to extend the fitted state with each day's newly-observed actuals without re-running the expensive MLE optimisation at every origin. Residual diagnostics (ACF, histogram, Q-Q plot, Ljung-Box test) in `notebooks/04_sarimax_models.ipynb`.

### 3. Feature-based ML (Parts 5-6)
Time features (hour/day-of-week/weekend + cyclic sin/cos encodings), lag features (1,2,3,6,12,24,48,168 hours), rolling mean/std (3,6,12,24,168-hour windows, shifted before rolling), plus the raw sensor/weather columns. `src/appliance_energy/features.py`.

XGBoost (primary) and HistGradientBoostingRegressor (comparison). Because a 24-hour-ahead forecast needs lag/rolling features that fall *inside* the forecast horizon for hours 2-24 (lags shorter than the 24h horizon), forecasting uses **genuine recursive multi-step forecasting**: each hour's features are built from real history plus the model's own predictions for earlier hours in the same horizon (`src/appliance_energy/models/feature_models.py::recursive_forecast`, leakage-safety verified in `tests/test_feature_models.py`).

A feature-group ablation study (`outputs/metrics/feature_ablation.csv`) tests four progressively richer feature sets to identify which groups actually help.

### 4. Foundation model (Part 7)
[Chronos-Bolt](https://arxiv.org/abs/2403.07815) (`amazon/chronos-bolt-tiny`), used **zero-shot**: never fine-tuned or shown any appliance energy data during training, only given a numeric context window at inference time. Chosen over TimeGPT (requires a paid Nixtla API key) and TimesFM (heavier setup) because it runs fully locally on CPU with no external account. Target-only - it has no mechanism to condition on covariates, an explicit limitation discussed in the Part 9 answers.

**Runs as its own subprocess** (`scripts/run_foundation_model.py`), invoked by `pipeline.py` via `subprocess.run`, rather than in-process after the XGBoost step - see "Known issues" below for why.

## Feature availability / conditional vs. true forecasts (Part 9, Q5)

Time-of-day and day-of-week features are always genuinely known in advance. Lag/rolling features built from the target's own past are genuinely available under the recursive strategy above. **Indoor sensor and outdoor weather values are not genuinely known at the forecast origin** - the SARIMAX-exog and full-featured XGBoost models use *realised* test-set values for these, making those specific results **conditional forecasts**, not true unconditional forecasts (explicitly permitted by the assignment brief, but flagged throughout). In this project the distinction turns out to matter less than expected in practice, because both covariate-using variants scored *worse* than their covariate-free counterparts anyway.

## Repository structure

```text
Timeseries-Datascience/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/              # downloaded UCI CSV (gitignored - reproducible via scripts/download_data.py)
│   ├── interim/          # cached feature table
│   └── processed/        # cleaned hourly series
│
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb        # EDA, decomposition, ADF/KPSS/ACF/PACF stationarity suite
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb              # grid search, residual diagnostics, rolling forecast + CI
│   ├── 05_feature_based_models.ipynb        # features, XGBoost/HistGB, ablation study
│   ├── 06_foundation_model.ipynb            # Chronos zero-shot
│   └── 07_model_comparison.ipynb            # full comparison + Part 9 discussion questions
│
├── src/
│   └── appliance_energy/
│       ├── config.py           # forecasting problem definition, all shared constants
│       ├── data.py             # download, clean, resample, stationarity tests
│       ├── features.py         # time/lag/rolling covariate engineering
│       ├── evaluation.py       # MAE/RMSE/MASE/Bias, rolling-origin backtest harness
│       ├── plotting.py         # every figure-producing function
│       ├── pipeline.py         # end-to-end orchestration (used by scripts/run_pipeline.py)
│       └── models/
│           ├── benchmarks.py
│           ├── sarimax.py          # parallelised AIC grid search, rolling forecast, diagnostics
│           ├── feature_models.py   # XGBoost/HistGB, recursive forecasting, ablation
│           └── foundation.py       # Chronos zero-shot + graceful fallback
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py           # main entry point - runs everything
│   ├── run_foundation_model.py   # foundation model step, run as its own subprocess by pipeline.py
│   └── evaluate_models.py        # re-evaluate/re-plot from saved forecasts without refitting
│
├── outputs/
│   ├── figures/           # 14 figures: EDA, diagnostics, forecast comparisons, feature importance
│   ├── forecasts/         # all_forecasts.csv, foundation_forecast.csv
│   ├── metrics/           # model_comparison.csv, sarima_grid_search.csv, feature_ablation.csv, ...
│   └── model_objects/
│
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_evaluation.py
    ├── test_benchmarks.py
    ├── test_sarimax.py
    └── test_feature_models.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
python -m pip install -e .         # editable install so `from appliance_energy import ...` works everywhere
```

Notable dependencies: `statsmodels` (SARIMAX), `xgboost`/`scikit-learn` (feature model), `torch`/`chronos-forecasting` (foundation model - CPU only, no GPU required), `jupyter`/`nbconvert` (notebooks).

## Running the pipeline

```bash
python scripts/run_pipeline.py
```

This downloads/caches the dataset, runs the full stationarity/EDA suite, fits every model class with the rolling-origin backtest described above, and writes forecasts/metrics/figures to `outputs/`. Runs in **~2 minutes** once the dataset and SARIMA grid search are cached (the grid search itself, run once, takes ~13 minutes; `python scripts/run_pipeline.py --refit-grid-search` re-runs it from scratch).

Other flags: `--force-download` (bypass the cached CSV), `--skip-foundation-model` (useful if offline and the Chronos weights aren't cached yet).

Individual steps can also be run standalone: `scripts/download_data.py`, `scripts/make_features.py`, `scripts/run_foundation_model.py`, `scripts/evaluate_models.py` (re-plots/re-scores from an existing `all_forecasts.csv` without refitting anything).

## Outputs

- `outputs/forecasts/all_forecasts.csv` - actual values + every model's forecast, aligned to the test period.
- `outputs/metrics/model_comparison.csv` - MAE/RMSE/MASE/Bias per model, plus `pct_vs_best_benchmark`.
- `outputs/metrics/sarima_grid_search.csv`, `feature_ablation.csv`, `stationarity_tests.csv`, `sarimax_ljung_box.csv`, `feature_importance.csv`.
- `outputs/figures/` - 14 figures covering EDA, decomposition, ACF/PACF (level and differenced), SARIMAX residual diagnostics and AIC heatmap, feature importance, full/zoomed forecast comparisons, error diagnostics, and metric comparison bars.

## Data leakage safeguards

- Lag/rolling features are built with `.shift(...)` before rolling, so a feature row at time *t* never reads the target at *t* or later (verified in `tests/test_features.py`).
- Multi-step feature-model forecasts use genuine recursive forecasting rather than reading ahead for lags shorter than the horizon (verified in `tests/test_feature_models.py`).
- SARIMAX rolling forecasts only ever fit on data available at each origin; state is extended (not re-fit on future data) via `append(refit=False)`.
- Exogenous/sensor covariate use is explicitly flagged as a *conditional* forecast throughout (see Q5 above), never silently presented as a true unconditional forecast.
- Model selection (SARIMA order, feature groups) is done by AIC / rolling-backtest MASE, not chosen post-hoc from final test-set performance.

## Known issues / engineering notes

- **XGBoost + PyTorch deadlock (macOS)**: fitting XGBoost and then loading PyTorch/Chronos in the *same* process caused a severe thread-contention hang (a ~1-minute workload taking 20+ minutes) during development, traced to XGBoost's and PyTorch's OpenMP thread pools both claiming every CPU core. Fixed by running the foundation-model step as an isolated subprocess (`scripts/run_foundation_model.py`) rather than in-process.
- **libomp segfault under Jupyter**: a separate, known PyTorch/macOS issue - a duplicate OpenMP runtime being loaded can segfault inside `libomp.dylib` during a PyTorch op when running under `ipykernel` specifically. Mitigated by setting `KMP_DUPLICATE_LIB_OK=TRUE` and capping thread pools before any heavy imports (see the setup cell in every notebook and the top of `scripts/run_pipeline.py`/`run_foundation_model.py`).
- **matplotlib figure memory**: `pipeline.py` explicitly closes every figure immediately after saving (`_save_fig`) - left open, ~14 uncached figures accumulated enough memory across a single long-running process to cause swapping by the time the memory-hungry Chronos step ran.

## Tests

```bash
pytest
```

28 tests covering: leakage-safety of lag/rolling features and recursive forecasting, MASE/RMSE/bias correctness (including a perfect-forecast MASE=0 check), benchmark forecast correctness (including the seasonal-naive recursive-extension edge case beyond one season), rolling-origin split correctness, data cleaning/resampling, and SARIMA grid-search order selection.

## Reproducibility

- `RANDOM_STATE = 0` set for every stochastic model (XGBoost, HistGB) - see `src/appliance_energy/config.py`.
- The raw dataset is re-downloadable from a fixed UCI URL rather than committed (`data/raw/` is gitignored); processed/interim data and all `outputs/` are committed so results are visible without re-running anything.
- Every advanced model is compared against the strongest *benchmark*, not just against each other (`pct_vs_best_benchmark` column in `model_comparison.csv`).

## References

- Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house.* Energy and Buildings, 140, pp.81-97. (Original dataset.)
- Hyndman, R.J. and Koehler, A.B. (2006). *Another look at measures of forecast accuracy.* International Journal of Forecasting, 22(4), pp.679-688.
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. and Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley.
- Cleveland, R.B., Cleveland, W.S., McRae, J.E. and Terpenning, I. (1990). *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* Journal of Official Statistics, 6(1), pp.3-73.
- Ansari, A.F. et al. (2024). *Chronos: Learning the Language of Time Series.* arXiv:2403.07815.
- Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp.785-794.
