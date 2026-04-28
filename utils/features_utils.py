from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def lag_feature_names(n_lags: int) -> List[str]:
    """Column names for the lag block: [lag_1, lag_2, ..., lag_n_lags]."""
    return [f"lag_{i + 1}" for i in range(n_lags)]


def calendar_feature_names() -> List[str]:
    """Column names for the calendar block."""
    return ["month", "quarter", "year"]


def all_feature_names(n_lags: int) -> List[str]:
    """Full ordered column list: lag features followed by calendar features."""
    return lag_feature_names(n_lags) + calendar_feature_names()


def _calendar_array(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 3) array of [month, quarter, year] for each date."""
    return np.column_stack([dates.month.values, dates.quarter.values, dates.year.values])


def build_feature_matrix(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    n_lags: int,
) -> np.ndarray:
    """Build a ``(N - n_lags + 1, n_lags + 3)`` lag + calendar feature matrix. """
    lags = np.lib.stride_tricks.sliding_window_view(series, n_lags)
    anchor_dates = dates[n_lags - 1 : n_lags - 1 + len(lags)]
    cal = _calendar_array(anchor_dates)
    return np.concatenate([lags, cal], axis=1)


def prediction_row(
    last_values: np.ndarray,
    anchor_date: pd.Timestamp,
    n_lags: int,
) -> pd.DataFrame:
    """Single-row DataFrame for ``model.predict()``."""
    cal = _calendar_array(pd.DatetimeIndex([anchor_date]))
    vec = np.concatenate([last_values.astype(float), cal[0]])
    return pd.DataFrame([vec], columns=all_feature_names(n_lags))
