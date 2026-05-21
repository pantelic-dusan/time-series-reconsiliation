"""Shared helpers and constants for reconciliation mode runners."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.aggregation_utils import get_all_levels, get_all_temporal_levels


logger = logging.getLogger("reconcile")


# Filename tags to distinguish reconciled outputs across modes so different
# modes' reconciliations of the same (model, method) do not collide.
#   csm = cross_structural_monthly    — structural reconciliation at monthly frequency
#   csq = cross_structural_quarterly  — structural reconciliation at quarterly frequency
#   ct  = cross_temporal              — per-series monthly <-> quarterly
#   cts = cross_temporal_structural   — joint (level × frequency) reconciliation
MODE_TAG: Dict[str, str] = {
    "cross_structural_monthly":   "csm",
    "cross_structural_quarterly": "csq",
    "cross_temporal":             "ct",
    "cross_temporal_structural":  "cts",
}

SUPPORTED_MODES = set(MODE_TAG.keys())


def structural_level_labels(config: Dict[str, Any]) -> List[str]:
    return [f"structural__{name}" for _, name, _ in get_all_levels(config)]


def quarterly_temporal_name(config: Dict[str, Any]) -> Optional[str]:
    """Return the (single) temporal level name used by cross_structural_quarterly mode.

    Returns None if no temporal level is configured. If more than one temporal
    level is configured, the first one is used and a warning is logged.
    """
    temporal_levels = get_all_temporal_levels(config)
    if not temporal_levels:
        return None
    if len(temporal_levels) > 1:
        logger.warning(
            f"More than one temporal level configured; cross_structural_quarterly uses "
            f"the first ({temporal_levels[0]['name']!r})."
        )
    return temporal_levels[0]["name"]


def read_forecast_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def write_reconciled_csv(target_path: Path, df: pd.DataFrame) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["date", "forecast", "ts_id"]].copy()
    out.to_csv(target_path, index=False)
    logger.info(f"  wrote {target_path}")


def reconciled_filename(
    model_name: str,
    mode: str,
    method: str,
    suffix_sep: str,
) -> str:
    return f"{model_name}{suffix_sep}{MODE_TAG[mode]}_{method}_forecasts.csv"
