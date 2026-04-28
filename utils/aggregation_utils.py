import copy
import logging
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_structural(
    dataframe: pd.DataFrame,
    level_columns: List[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Aggregate data by a subset of id_cols (structural aggregation)."""
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


def get_level_config(
    level_type: str,
    level_name: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a (deep-copied) config for the aggregation level."""
    return copy.deepcopy(base_config)


def get_all_levels(config: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
    """Return [(level_type, level_name, group_columns), ...] for all hierarchy levels.

    The base level is NOT included — it is handled separately by callers.
    """
    hierarchy = config.get("hierarchy", {})
    levels: List[Tuple[str, str, List[str]]] = []

    for level_name in hierarchy.get("structural_levels", []):
        group_columns: List[str] = [] if level_name == "total" else [level_name]
        levels.append(("structural", level_name, group_columns))

    return levels


def iter_levels(
    config: Dict[str, Any],
    dataframe: pd.DataFrame,
) -> Iterator[Tuple[str, Dict[str, Any], pd.DataFrame]]:
    """Yield (level_label, level_config, level_frame) for base + structural levels."""
    yield "base", config, dataframe

    for level_type, level_name, group_columns in get_all_levels(config):
        level_label = f"{level_type}__{level_name}"
        level_config = get_level_config(level_type, level_name, config)

        if level_type == "structural":
            level_frame = aggregate_structural(dataframe, group_columns, config)
        else:
            logger.warning(f"Unknown level type '{level_type}', skipping.")
            continue

        yield level_label, level_config, level_frame

