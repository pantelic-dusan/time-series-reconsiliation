"""Hierarchical aggregation utilities.

Provides structural aggregation (grouping by subsets of id_cols)
and temporal aggregation (resampling to coarser frequencies)
for running forecasts at multiple hierarchy levels.
"""

import copy
from typing import Any, Dict, List, Tuple

import pandas as pd


# Maps temporal level names to pandas offset aliases and period aliases
TEMPORAL_FREQ_MAP = {
    "quarterly": {"offset": "QS", "period": "Q", "periods_per_year": 4},
    "half_yearly": {"offset": "6MS", "period": "6M", "periods_per_year": 2},
}


def aggregate_structural(
    dataframe: pd.DataFrame,
    level_columns: List[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Aggregate data by a subset of id_cols (structural aggregation).

    Groups by level_columns + time_col and sums the target column.
    Creates a new ts_id from the grouped columns.

    Args:
        dataframe: Raw data with all id_cols and ts_id already present.
        level_columns: Subset of id_cols to group by (e.g. ["material"]).
                       Empty list means total aggregation.
        config: Full experiment config.

    Returns:
        Aggregated DataFrame with ts_id, time_col, target_col columns.
    """
    target_column = config["data"]["target_col"]
    time_column = config["data"]["time_col"]

    if level_columns:
        grouped = (
            dataframe.groupby(level_columns + [time_column])[target_column]
            .sum()
            .reset_index()
        )
        grouped["ts_id"] = grouped[level_columns].astype(str).agg("_".join, axis=1)
    else:
        # Total aggregation — single series
        grouped = (
            dataframe.groupby(time_column)[target_column]
            .sum()
            .reset_index()
        )
        grouped["ts_id"] = "total"

    return grouped


def aggregate_temporal(
    dataframe: pd.DataFrame,
    temporal_level: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Resample data to a coarser temporal frequency.

    Operates on base structural level (material_customer_location).
    Groups by ts_id and resamples the target column by summing.

    Args:
        dataframe: Data at monthly frequency with ts_id column.
        temporal_level: Key in TEMPORAL_FREQ_MAP (e.g. "quarterly").
        config: Full experiment config.

    Returns:
        Resampled DataFrame with ts_id, time_col, target_col columns.
    """
    target_column = config["data"]["target_col"]
    time_column = config["data"]["time_col"]
    offset_freq = TEMPORAL_FREQ_MAP[temporal_level]["offset"]

    resampled_parts = []
    for ts_id, group in dataframe.groupby("ts_id"):
        group = group.set_index(time_column).sort_index()
        resampled = group[[target_column]].resample(offset_freq).sum().reset_index()
        resampled["ts_id"] = ts_id
        resampled_parts.append(resampled)

    return pd.concat(resampled_parts, ignore_index=True)


def get_level_config(
    level_type: str,
    level_name: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a modified copy of config adjusted for the aggregation level.

    Adjusts horizon, frequency, seasonal periods, lags, and context lengths
    so that each model receives appropriate parameters for the temporal grain.

    Args:
        level_type: "base", "structural", or "temporal".
        level_name: E.g. "material", "quarterly", "half_yearly".
        base_config: Original config dict.

    Returns:
        Deep-copied config with adjusted values.
    """
    config = copy.deepcopy(base_config)

    if level_type == "structural":
        # Structural levels stay at monthly frequency — no param changes needed
        return config

    if level_type == "temporal":
        freq_info = TEMPORAL_FREQ_MAP[level_name]
        ppy = freq_info["periods_per_year"]  # periods per year

        config["data"]["frequency"] = freq_info["offset"]

        # Horizon: forecast 1 year ahead in the new grain
        config["experiment"]["horizon"] = ppy

        # Adjust each model's params
        for model_config in config["models"]:
            params = model_config.get("params", {})

            # Seasonal period (ARIMA, Holt-Winters)
            if "seasonal_period" in params:
                params["seasonal_period"] = ppy
            if "seasonal_periods" in params:
                params["seasonal_periods"] = ppy

            # Lags (ML models)
            if "n_lags" in params:
                params["n_lags"] = max(ppy, 2)

            # Context length (DeepAR, Chronos, TimesFM)
            if "context_length" in params:
                params["context_length"] = max(ppy * 2, 4)

            # Input size (N-HiTS)
            if "input_size" in params:
                params["input_size"] = max(ppy, 2)

    return config


def get_all_levels(config: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
    """Return a list of (level_type, level_name, group_columns) for all hierarchy levels.

    The base level is NOT included — it is handled separately in main.py.
    """
    hierarchy = config.get("hierarchy", {})
    levels = []

    # Structural levels (monthly, different groupings)
    for level_name in hierarchy.get("structural_levels", []):
        if level_name == "total":
            group_columns = []
        else:
            group_columns = [level_name]
        levels.append(("structural", level_name, group_columns))

    # Temporal levels (base structural grouping, coarser frequency)
    for level_name in hierarchy.get("temporal_levels", []):
        levels.append(("temporal", level_name, []))

    return levels

