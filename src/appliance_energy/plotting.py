"""
All figure-producing functions for the project.

Every function here returns a matplotlib Figure so callers (scripts,
notebooks) decide whether/where to save it. Keeping plotting logic out
of the modelling modules keeps those testable and this one visual-only.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------
# EDA (Part 1)
# ------------------------------------------------------------------

def plot_full_series(series: pd.Series, title: str = "Appliance energy use (hourly)"):
    fig, ax = plt.subplots(figsize=(14, 5))
    series.plot(ax=ax, linewidth=0.8, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    return fig


def plot_series_window(series: pd.Series, days: int = 14, title: str | None = None):
    """Zoom in on the last ``days`` days to show daily/weekly structure clearly."""
    window = series.tail(days * 24)
    fig, ax = plt.subplots(figsize=(14, 5))
    window.plot(ax=ax, linewidth=1.2, color="tab:blue", marker="o", markersize=2)
    ax.set_title(title or f"Appliance energy use - last {days} days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    return fig


def plot_hourly_and_weekday_profiles(series: pd.Series):
    """Boxplots of usage by hour-of-day and by day-of-week - shows seasonal shape."""
    df = series.to_frame("Appliances")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.boxplot(data=df, x="hour", y="Appliances", ax=axes[0], color="tab:blue")
    axes[0].set_title("Appliance use by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Appliances (Wh)")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sns.boxplot(data=df, x="dayofweek", y="Appliances", ax=axes[1], color="tab:orange")
    axes[1].set_title("Appliance use by day of week")
    axes[1].set_xlabel("Day of week")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_labels)
    axes[1].set_ylabel("")

    fig.tight_layout()
    return fig


def plot_decomposition(decomposition_result, title: str = "Seasonal-trend decomposition"):
    fig = decomposition_result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 72, title_suffix: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF {title_suffix}".strip())
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF {title_suffix}".strip())
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str], title: str = "Correlation matrix"):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Model residual diagnostics (Part 4)
# ------------------------------------------------------------------

def plot_residual_diagnostics(residuals: pd.Series, title: str = "Residual diagnostics"):
    """4-panel residual diagnostic plot: time series, histogram, Q-Q, ACF."""
    from scipy import stats as scipy_stats

    residuals = residuals.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(residuals.index, residuals.values, linewidth=0.8, color="tab:blue")
    axes[0, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Residuals over time")
    # Weekly-spaced ticks with a compact "01 Feb" style label - the
    # default datetime locator/formatter crams in far too many
    # timestamps for a ~3000-hour index and the labels overlap into an
    # unreadable block. A fixed weekly locator plus ConciseDateFormatter
    # keeps a handful of clean, evenly-spaced labels instead.
    axes[0, 0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[0, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[0, 0].xaxis.get_major_locator()))

    axes[0, 1].hist(residuals.values, bins=40, color="tab:blue", edgecolor="white")
    axes[0, 1].set_title("Residual distribution")

    scipy_stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot")

    plot_acf(residuals, lags=48, ax=axes[1, 1])
    axes[1, 1].set_title("Residual ACF")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Forecast comparison plots (Part 8)
# ------------------------------------------------------------------

def plot_forecast_vs_actual(
    train: pd.Series,
    test: pd.Series,
    forecast_df: pd.DataFrame,
    conf_int: pd.DataFrame | None = None,
    conf_int_model: str | None = None,
    context_days: int = 14,
    title: str = "Appliance energy forecasting - all models",
):
    """Full test-period plot of every model's forecast against the actuals."""
    fig, ax = plt.subplots(figsize=(15, 7))

    train.tail(context_days * 24).plot(ax=ax, label="Training data", color="grey", linewidth=1.0)
    test.plot(ax=ax, label="Actual (test)", color="black", linewidth=2.0)

    palette = sns.color_palette("tab10", n_colors=len(forecast_df.columns))
    for col, colour in zip(forecast_df.columns, palette):
        forecast_df[col].plot(ax=ax, label=col, alpha=0.85, linewidth=1.3, color=colour)

    if conf_int is not None and conf_int_model is not None:
        ax.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="tab:blue",
            alpha=0.15,
            label=f"{conf_int_model} 95% CI",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_forecast_zoom(
    test: pd.Series,
    forecast_df: pd.DataFrame,
    days: int = 4,
    title: str = "Forecast comparison - zoomed",
):
    """Zoom into the first few days of the test period for a readable close-up."""
    window_index = test.index[: days * 24]

    fig, ax = plt.subplots(figsize=(14, 6))
    test.loc[window_index].plot(ax=ax, label="Actual", color="black", linewidth=2.2)

    palette = sns.color_palette("tab10", n_colors=len(forecast_df.columns))
    for col, colour in zip(forecast_df.columns, palette):
        forecast_df.loc[window_index, col].plot(
            ax=ax, label=col, alpha=0.85, linewidth=1.4, color=colour
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_error_diagnostics(test: pd.Series, forecast_df: pd.DataFrame):
    """Boxplot of forecast errors per model + error-over-time for the best few models."""
    errors = forecast_df.sub(test, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    errors.boxplot(ax=axes[0], rot=45)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Forecast error distribution by model")
    axes[0].set_ylabel("Forecast error (predicted - actual)")

    abs_errors = errors.abs()
    mean_abs_by_day = abs_errors.groupby(abs_errors.index.date).mean()
    mean_abs_by_day.plot(ax=axes[1], marker="o", markersize=3)
    axes[1].set_title("Mean absolute error per test day")
    axes[1].set_xlabel("Test day")
    axes[1].set_ylabel("Mean |error|")
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame):
    """Bar charts comparing MAE, RMSE, MASE, Bias across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = ["MAE", "RMSE", "MASE", "Bias"]

    ordered = metrics_df.sort_values("MASE")

    for ax, metric in zip(axes.flat, metrics):
        colours = ["tab:red" if v < 0 else "tab:blue" for v in ordered[metric]]
        ax.bar(ordered["model"], ordered[metric], color=colours)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 20, title: str = "Feature importance"):
    top = importance.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * top_n)))
    ax.barh(top.index, top.values, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_sarima_grid_heatmap(grid_df: pd.DataFrame, d_value: int, title: str | None = None):
    """
    Heatmap of AIC over (p, q) for a fixed d, from the SARIMA grid search
    results table (columns: p, d, q, aic).

    ``robust=True`` colour-scales off the 2nd/98th percentile of AIC
    rather than its true min/max: a handful of over-parameterised (p, q)
    combinations are numerically unstable and land AIC values many
    thousands higher than every sensible candidate, which would
    otherwise stretch the colour scale so far that the actually
    converged, informative cells all render as a single flat colour.
    Annotating each cell with its AIC (rounded to the nearest 10) makes
    the figure readable as a real reference table, not just a colour
    gradient.
    """
    subset = grid_df[grid_df["d"] == d_value]
    pivot = subset.pivot(index="p", columns="q", values="aic")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 7},
        cmap="viridis_r",
        robust=True,
        ax=ax,
    )
    ax.set_title(title or f"SARIMA AIC grid (d={d_value})")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    fig.tight_layout()
    return fig


def plot_feature_ablation(ablation_df: pd.DataFrame, title: str = "Feature-group ablation (XGBoost)"):
    """
    MASE per feature-group configuration, from the ablation study
    (columns: model [feature-group label], MASE). ``run_feature_group_ablation``
    returns its rows sorted by MASE ascending, so bars run best-to-worst
    left-to-right; the best (green) bar is never the full-feature-set
    configuration, which is the point being visualised.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = ["tab:green" if v == ablation_df["MASE"].min() else "tab:blue" for v in ablation_df["MASE"]]
    ax.bar(ablation_df["model"], ablation_df["MASE"], color=colours)
    for i, v in enumerate(ablation_df["MASE"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("MASE (lower = better)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_foundation_forecast(
    test: pd.Series,
    foundation_forecast: pd.Series,
    foundation_ci: pd.DataFrame | None = None,
    title: str = "Chronos (zero-shot) rolling forecast - full test period",
):
    """Standalone full-test-period plot of the foundation model forecast with its quantile band."""
    fig, ax = plt.subplots(figsize=(14, 6))
    test.plot(ax=ax, label="Actual", color="black", linewidth=2)
    foundation_forecast.plot(ax=ax, label="Chronos (zero-shot)", color="tab:purple", linewidth=1.3)
    if foundation_ci is not None:
        ax.fill_between(
            foundation_ci.index,
            foundation_ci["lower"],
            foundation_ci["upper"],
            color="tab:purple",
            alpha=0.15,
            label="10-90% interval",
        )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sarimax_variant_comparison(
    test: pd.Series,
    sarimax_univariate: pd.Series,
    sarimax_exog: pd.Series,
    days: int = 6,
    title: str = "SARIMAX: target-only vs. + exogenous weather",
):
    """
    Zoomed comparison of the two SARIMAX variants against actuals, to
    make visible (not just tabulated) that the exogenous variant does
    not track the actuals any more closely than the univariate one.
    """
    window_index = test.index[: days * 24]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    test.loc[window_index].plot(ax=ax, label="Actual", color="black", linewidth=2.2)
    sarimax_univariate.loc[window_index].plot(
        ax=ax, label="SARIMAX (target-only)", color="tab:blue", linewidth=1.4
    )
    sarimax_exog.loc[window_index].plot(
        ax=ax, label="SARIMAX (+ exogenous weather)", color="tab:red", linewidth=1.4, linestyle="--"
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    return fig
