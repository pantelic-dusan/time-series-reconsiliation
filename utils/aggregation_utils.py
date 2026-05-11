import copy
import logging
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_MONTHS_PER_TEMPORAL_FREQ: Dict[str, int] = {
    "MS": 1,
    "M": 1,
    "QS": 3,
    "Q": 3,
    "YS": 12,
    "AS": 12,
    "Y": 12,
    "A": 12,
}


def _months_per_temporal_period(freq: str) -> int:
    """Return the number of base monthly periods in one unit of ``freq``."""
    key = freq.split("-")[0].upper()  # strip anchoring suffix, e.g. "QS-JAN" → "QS"
    return _MONTHS_PER_TEMPORAL_FREQ.get(key, 1)


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


def aggregate_temporal(
    dataframe: pd.DataFrame,
    freq: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Resample each ts_id group to ``freq`` by summing the target column.:"""

    target_column = config["data"]["target_col"]
    time_column = config["data"]["time_col"]

    records: List[pd.DataFrame] = []
    for ts_id, group in dataframe.groupby("ts_id", sort=False):
        resampled = (
            group.set_index(time_column)[target_column]
            .resample(freq)
            .sum()
            .reset_index()
        )
        resampled["ts_id"] = ts_id
        records.append(resampled)

    if not records:
        return pd.DataFrame(columns=["ts_id", time_column, target_column])

    return pd.concat(records, ignore_index=True)


def _build_temporal_level_config(
    base_config: Dict[str, Any],
    temporal_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a deep-copied config adapted for a temporal aggregation level."""

    lc = copy.deepcopy(base_config)

    lc["data"]["frequency"] = temporal_cfg["freq"]
    lc["experiment"]["horizon"] = temporal_cfg["horizon"]

    num_val_temporal = temporal_cfg.get("num_val_periods", 1)
    months = num_val_temporal * _months_per_temporal_period(temporal_cfg["freq"])
    if "hpo" in lc:
        lc["hpo"]["num_val_periods"] = months

    param_overrides = temporal_cfg.get("param_overrides", {})
    search_space_overrides = temporal_cfg.get("search_space_overrides", {})
    if param_overrides or search_space_overrides:
        for model_entry in lc["models"]:
            overrides = param_overrides.get(model_entry["name"], {})
            if overrides:
                model_entry.setdefault("params", {})
                model_entry["params"] = {**model_entry["params"], **overrides}

            ss_overrides = search_space_overrides.get(model_entry["name"], {})
            if ss_overrides:
                hpo_block = model_entry.setdefault("hpo", {})
                search_space = hpo_block.setdefault("search_space", {})
                for param_name, spec in ss_overrides.items():
                    search_space[param_name] = copy.deepcopy(spec)

    return lc


def get_level_config(
    level_type: str,
    level_name: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a (deep-copied) config for the aggregation level."""
    return copy.deepcopy(base_config)


def get_all_levels(config: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
    """Return [(level_type, level_name, group_columns), ...] for all structural hierarchy levels.

    The base level is NOT included — it is handled separately by callers.    """
    hierarchy = config.get("hierarchy", {})
    levels: List[Tuple[str, str, List[str]]] = []

    for entry in hierarchy.get("structural_levels", []):
        level_name = entry if isinstance(entry, str) else entry["name"]
        group_columns: List[str] = [] if level_name == "total" else [level_name]
        levels.append(("structural", level_name, group_columns))

    return levels


def get_all_temporal_levels(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of temporal-level descriptors from config (may be empty)."""
    return config.get("hierarchy", {}).get("temporal_levels", [])


def get_level_sample_frac(level_label: str, config: Dict[str, Any]) -> float | None:
    """Resolve the sample_frac for a given level_label."""
    hierarchy = config.get("hierarchy", {})
    global_default = config.get("hpo", {}).get("sample_frac")

    # Strip "temporal__<name>__" prefix to find the underlying cross-section.
    cross_section_label = level_label
    if level_label.startswith("temporal__"):
        # "temporal__quarter__base"                  -> "base"
        # "temporal__quarter__structural__material"  -> "structural__material"
        parts = level_label.split("__", 2)
        cross_section_label = parts[2] if len(parts) > 2 else "base"

    if cross_section_label == "base":
        base_block = hierarchy.get("base") or {}
        resolved = base_block["sample_frac"] if "sample_frac" in base_block else global_default
    elif cross_section_label.startswith("structural__"):
        target_name = cross_section_label[len("structural__"):]
        resolved = global_default
        for entry in hierarchy.get("structural_levels", []):
            if isinstance(entry, dict) and entry.get("name") == target_name:
                if "sample_frac" in entry:
                    resolved = entry.get("sample_frac")
                break
            # Plain-string entry: no override → keep global default.
    else:
        resolved = global_default

    if resolved is None:
        return None
    try:
        f = float(resolved)
    except (TypeError, ValueError):
        return None
    if f <= 0 or f >= 1:
        return None
    return f


def iter_levels(
    config: Dict[str, Any],
    dataframe: pd.DataFrame,
) -> Iterator[Tuple[str, Dict[str, Any], pd.DataFrame]]:
    """Yield (level_label, level_config, level_frame) for all hierarchy levels."""
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

    for temporal_cfg in get_all_temporal_levels(config):
        temporal_name = temporal_cfg["name"]
        temporal_freq = temporal_cfg["freq"]

        # temporal × base
        temporal_base_frame = aggregate_temporal(dataframe, temporal_freq, config)
        temporal_base_config = _build_temporal_level_config(config, temporal_cfg)
        yield (
            f"temporal__{temporal_name}__base",
            temporal_base_config,
            temporal_base_frame,
        )

        # temporal × each structural level
        for level_type, level_name, group_columns in get_all_levels(config):
            if level_type != "structural":
                continue
            structural_frame = aggregate_structural(dataframe, group_columns, config)
            temporal_structural_frame = aggregate_temporal(
                structural_frame, temporal_freq, config
            )
            temporal_structural_config = _build_temporal_level_config(config, temporal_cfg)
            yield (
                f"temporal__{temporal_name}__structural__{level_name}",
                temporal_structural_config,
                temporal_structural_frame,
            )

