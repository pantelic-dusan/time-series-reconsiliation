from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

EPS = 1e-8

def normalized_root_mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """nRMSE = RMSE / mean(|actual|) * 100.  Scale-free, comparable across hierarchy levels.
    Returns NaN for all-zero series."""
    mean_abs_actual = float(np.mean(np.abs(actual)))
    if mean_abs_actual <= EPS:
        return float("nan")
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    return rmse / mean_abs_actual * 100


def mean_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """MAPE = mean(|actual - forecast| / |actual|) * 100, computed only over rows where |actual| > EPS.
    Returns NaN if no rows qualify (e.g., all-zero series)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.abs(actual) > EPS
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def symmetric_mean_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """sMAPE = mean(2 * |actual - forecast| / (|actual| + |forecast| + EPS)) * 100.  EPS guards against double-zero pairs."""
    denominator = np.maximum(np.abs(actual) + np.abs(forecast), EPS)
    return float(np.mean(2.0 * np.abs(actual - forecast) / denominator) * 100)


def bias_percentage(actual: np.ndarray, forecast: np.ndarray) -> float:
    """BIAS% = mean(forecast - actual) / mean(|actual|) * 100.  Sign preserved (positive = over-forecasting).
    Scale-free, comparable across hierarchy levels.  Returns NaN for all-zero series."""
    mean_abs_actual = float(np.mean(np.abs(actual)))
    if mean_abs_actual <= EPS:
        return float("nan")
    return float(np.mean(forecast - actual) / mean_abs_actual * 100)


def tracking_signal(actual: np.ndarray, forecast: np.ndarray) -> float:
    """TS = cumulative_error / MAD.  Values far from 0 indicate systematic bias.  Returns 0.0 for a perfect forecast."""
    errors = forecast - actual
    mad = max(float(np.mean(np.abs(errors))), EPS)
    return float(np.sum(errors) / mad)


def weighted_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """WMAPE = sum(|actual - forecast|) / sum(|actual|) * 100.  Returns NaN for all-zero series."""
    total_actual = float(np.sum(np.abs(actual)))
    if total_actual <= EPS:
        return float("nan")
    return float(np.sum(np.abs(actual - forecast)) / total_actual * 100)


def median_absolute_percentage_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """MdAPE = median(|actual - forecast| / |actual|) * 100, computed only over rows where |actual| > EPS.
    Returns NaN if no rows qualify."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = np.abs(actual) > EPS
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


ALL_METRICS = {
    "WMAPE": weighted_absolute_percentage_error,
    "nRMSE": normalized_root_mean_squared_error,
    "MAPE": mean_absolute_percentage_error,
    "MdAPE": median_absolute_percentage_error,
    "sMAPE": symmetric_mean_absolute_percentage_error,
    "BIAS_PCT": bias_percentage,
    "TS": tracking_signal,
}


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    """Compute all forecasting metrics for a single pair of arrays."""
    return {name: func(actual, forecast) for name, func in ALL_METRICS.items()}


def evaluate_model(
    forecast_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    time_column: str,
    target_column: str,
) -> Dict[str, Any]:
    """Evaluate a single model's forecasts against actuals."""
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

    per_series_rows = []
    for ts_id, group in merged.groupby("ts_id"):
        series_actual = group[target_column].values
        series_forecast = group["forecast"].values
        row = {"ts_id": ts_id}
        row.update(compute_metrics(series_actual, series_forecast))
        per_series_rows.append(row)

    per_series_df = pd.DataFrame(per_series_rows)

    # Level-aggregate metric = mean of per-series metrics (NaN-safe).
    overall = {
        name: float(np.nanmean(per_series_df[name].values)) if name in per_series_df else np.nan
        for name in ALL_METRICS
    }
    overall["num_series"] = merged["ts_id"].nunique()
    overall["num_predictions"] = len(merged)

    return {
        "overall": overall,
        "per_series": per_series_df,
    }
