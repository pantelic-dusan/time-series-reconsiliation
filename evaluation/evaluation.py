"""Forecast evaluation metrics.

Pure numeric metric functions + a per-model scorer (`evaluate_model`). The
per-level driver lives in `evaluate.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions (operate on numpy arrays)
# ---------------------------------------------------------------------------

def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """MAE = mean(|actual - forecast|)"""
    return float(np.mean(np.abs(actual - forecast)))


def root_mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """RMSE = sqrt(mean((actual - forecast)^2))"""
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mean_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """MAPE = mean(|actual - forecast| / |actual|) * 100.  Skips zeros in actual."""
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def symmetric_mean_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """sMAPE = mean(2 * |actual - forecast| / (|actual| + |forecast|)) * 100"""
    denominator = np.abs(actual) + np.abs(forecast)
    mask = denominator != 0
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100)


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """BIAS = mean(forecast - actual).  Positive = over-forecasting."""
    return float(np.mean(forecast - actual))


def tracking_signal(actual: np.ndarray, forecast: np.ndarray) -> float:
    """TS = cumulative_error / MAD.  Values far from 0 indicate systematic bias."""
    errors = forecast - actual
    mad = np.mean(np.abs(errors))
    if mad == 0:
        return np.nan
    return float(np.sum(errors) / mad)


def r_squared(actual: np.ndarray, forecast: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot.  1 = perfect, 0 = as good as mean, negative = worse."""
    ss_residual = np.sum((actual - forecast) ** 2)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    if ss_total == 0:
        return np.nan
    return float(1.0 - ss_residual / ss_total)


def weighted_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """WAPE = sum(|actual - forecast|) / sum(|actual|) * 100.  Robust to zeros."""
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.nan
    return float(np.sum(np.abs(actual - forecast)) / total_actual * 100)


# ---------------------------------------------------------------------------
# All metrics in one call
# ---------------------------------------------------------------------------

ALL_METRICS = {
    "MAE": mean_absolute_error,
    "RMSE": root_mean_squared_error,
    "MAPE": mean_absolute_percentage_error,
    "sMAPE": symmetric_mean_absolute_percentage_error,
    "BIAS": bias,
    "TS": tracking_signal,
    "R2": r_squared,
    "WAPE": weighted_absolute_percentage_error,
}


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """Compute all forecasting metrics for a single pair of arrays."""
    return {name: func(actual, forecast) for name, func in ALL_METRICS.items()}


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def evaluate_model(
    forecast_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    time_column: str,
    target_column: str,
) -> Dict[str, Any]:
    """Evaluate a single model's forecasts against actuals.

    Args:
        forecast_dataframe: Must have columns [time_column, "forecast", "ts_id"]
        test_dataframe:     Must have columns [time_column, target_column, "ts_id"]

    Returns:
        {"overall": {metric: value, ...}, "per_series": DataFrame}
    """
    forecast_dataframe = forecast_dataframe.copy()
    forecast_dataframe[time_column] = pd.to_datetime(forecast_dataframe[time_column])

    test_dataframe = test_dataframe.copy()
    test_dataframe[time_column] = pd.to_datetime(test_dataframe[time_column])

    merged = pd.merge(
        forecast_dataframe,
        test_dataframe[[time_column, target_column, "ts_id"]],
        on=["ts_id", time_column],
        how="inner",
    )

    if merged.empty:
        logger.warning("evaluate_model: no overlapping (ts_id, date) rows between forecast and test")
        return {
            "overall": {name: np.nan for name in ALL_METRICS},
            "per_series": pd.DataFrame(),
        }

    actual = merged[target_column].values
    forecast = merged["forecast"].values

    overall = compute_metrics(actual, forecast)
    overall["num_series"] = merged["ts_id"].nunique()
    overall["num_predictions"] = len(merged)

    per_series_rows = []
    for ts_id, group in merged.groupby("ts_id"):
        series_actual = group[target_column].values
        series_forecast = group["forecast"].values
        row = {"ts_id": ts_id}
        row.update(compute_metrics(series_actual, series_forecast))
        per_series_rows.append(row)

    return {
        "overall": overall,
        "per_series": pd.DataFrame(per_series_rows),
    }
