from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from aggregation import (
    aggregate_structural,
    aggregate_temporal,
    get_all_levels,
    get_level_config,
)
from evaluation.evaluation import evaluate_model
from logging_utils import setup_logging, timed


# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config.yaml"
LOG_FILE: str = "logs/evaluate.log"
FORECASTS_DIR: Optional[str] = None  # None -> take from config["storage"]["forecasts_dir"]
EVALUATIONS_DIR: Optional[str] = None  # None -> take from config["storage"]["evaluations_dir"]
ONLY_LEVEL: Optional[str] = None  # e.g. "base", "structural__material", "temporal__quarterly"


logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_and_split(config: Dict[str, Any]) -> pd.DataFrame:
    """Reload raw data, attach ts_id, return ONLY the post-train-end (test) rows."""
    data_config = config["data"]
    dataframe = pd.read_csv(data_config["data_path"], parse_dates=[data_config["time_col"]])
    dataframe["ts_id"] = dataframe[data_config["id_cols"]].astype(str).agg("_".join, axis=1)

    train_end_date = pd.Timestamp(config["experiment"]["train_end_date"])
    test_dataframe = dataframe[dataframe[data_config["time_col"]] > train_end_date]
    return test_dataframe


def _enumerate_levels(
    config: Dict[str, Any],
) -> List[Tuple[str, str, str, List[str]]]:
    """Return [(level_label, level_type, level_name, group_columns), ...] including base."""
    levels: List[Tuple[str, str, str, List[str]]] = [("base", "base", "base", [])]
    for level_type, level_name, group_columns in get_all_levels(config):
        levels.append((f"{level_type}__{level_name}", level_type, level_name, group_columns))
    return levels


def _build_level_test(
    level_type: str,
    level_name: str,
    group_columns: List[str],
    test_dataframe: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    if level_type == "base":
        return test_dataframe
    if level_type == "structural":
        return aggregate_structural(test_dataframe, group_columns, config)
    if level_type == "temporal":
        return aggregate_temporal(test_dataframe, level_name, config)
    raise ValueError(f"Unknown level type '{level_type}'")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def evaluate_level(
    level_label: str,
    level_config: Dict[str, Any],
    level_test: pd.DataFrame,
    level_forecasts_dir: Path,
    level_evaluations_dir: Path,
) -> List[Dict[str, Any]]:
    """Score every `*_forecasts.csv` in `level_forecasts_dir` against `level_test`."""
    if not level_forecasts_dir.is_dir():
        logger.warning(f"{level_label}: forecasts dir missing ({level_forecasts_dir}), skipping.")
        return []

    level_evaluations_dir.mkdir(parents=True, exist_ok=True)

    time_column = level_config["data"]["time_col"]
    target_column = level_config["data"]["target_col"]

    summary_rows: List[Dict[str, Any]] = []

    for model_config in level_config["models"]:
        model_name = model_config["name"]
        forecast_path = level_forecasts_dir / f"{model_name}_forecasts.csv"
        if not forecast_path.exists():
            logger.warning(f"{level_label}/{model_name}: no forecast file, skipping.")
            continue

        with timed(f"eval {level_label}/{model_name}", logger):
            forecast_dataframe = pd.read_csv(forecast_path)
            result = evaluate_model(forecast_dataframe, level_test, time_column, target_column)

            row = {"model": model_name}
            row.update(result["overall"])
            summary_rows.append(row)

            if not result["per_series"].empty:
                detail_path = level_evaluations_dir / f"{model_name}_evaluation_detail.csv"
                result["per_series"].to_csv(detail_path, index=False)

            overall = result["overall"]
            logger.info(
                f"  {model_name}: "
                f"MAE={overall.get('MAE', float('nan')):.2f}  "
                f"RMSE={overall.get('RMSE', float('nan')):.2f}  "
                f"WAPE={overall.get('WAPE', float('nan')):.1f}%  "
                f"BIAS={overall.get('BIAS', float('nan')):.2f}  "
                f"R2={overall.get('R2', float('nan')):.3f}"
            )

    if summary_rows:
        summary_path = level_evaluations_dir / "evaluation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"{level_label}: summary → {summary_path}")
    else:
        logger.warning(f"{level_label}: no models evaluated.")

    return summary_rows


def run_evaluation(config: Dict[str, Any]) -> None:
    test_dataframe = _load_and_split(config)
    storage_config = config["storage"]
    base_forecasts_dir = Path(
        FORECASTS_DIR or storage_config.get("forecasts_dir", storage_config["output_dir"])
    )
    base_evaluations_dir = Path(
        EVALUATIONS_DIR or storage_config.get("evaluations_dir", storage_config["output_dir"])
    )
    base_evaluations_dir.mkdir(parents=True, exist_ok=True)

    cross_level_rows: List[Dict[str, Any]] = []

    for level_label, level_type, level_name, group_columns in _enumerate_levels(config):
        if ONLY_LEVEL is not None and level_label != ONLY_LEVEL:
            continue

        with timed(f"level={level_label}", logger):
            if level_type == "base":
                level_config = config
            else:
                level_config = get_level_config(level_type, level_name, config)

            level_test = _build_level_test(
                level_type, level_name, group_columns, test_dataframe, config
            )
            logger.info(
                f"  {level_label}: {level_test['ts_id'].nunique()} series, "
                f"{len(level_test)} test rows"
            )
            level_rows = evaluate_level(
                level_label=level_label,
                level_config=level_config,
                level_test=level_test,
                level_forecasts_dir=base_forecasts_dir / level_label,
                level_evaluations_dir=base_evaluations_dir / level_label,
            )
            for row in level_rows:
                cross_level_rows.append({"level": level_label, **row})

    if cross_level_rows:
        cross_level_path = base_evaluations_dir / "kpi_by_level.csv"
        pd.DataFrame(cross_level_rows).to_csv(cross_level_path, index=False)
        logger.info(f"Cross-level KPI table → {cross_level_path}")
    else:
        logger.warning("No cross-level KPI rows were produced.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    setup_logging(LOG_FILE)
    config = load_config(CONFIG_PATH)
    try:
        with timed("full evaluation run", logger):
            run_evaluation(config)
    except Exception:
        logger.exception("Evaluation run failed at top level.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

