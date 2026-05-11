from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.aggregation_utils import iter_levels
from utils.utils import load_config, load_raw_data
from utils.evaluation_utils import evaluate_model
from utils.logging_utils import setup_logging, timed


# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config/config.yaml"
LOG_FILE: str = "logs/evaluate.log"
FORECASTS_DIR: Optional[str] = None  # None -> take from config["storage"]["forecasts_dir"]
EVALUATIONS_DIR: Optional[str] = None  # None -> take from config["storage"]["evaluations_dir"]
ONLY_LEVEL: Optional[str] = None  # e.g. "base", "structural__material"


logger = logging.getLogger("evaluate")


def _load_test_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    """Reload raw data, attach ts_id, return ONLY the post-train-end (test) rows."""
    dataframe = load_raw_data(config)
    train_end_date = pd.Timestamp(config["experiment"]["train_end_date"])
    test_dataframe = dataframe[dataframe[config["data"]["time_col"]] > train_end_date]
    return test_dataframe


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
                f"WMAPE={overall.get('WMAPE', float('nan')):.1f}%  "
                f"nRMSE={overall.get('nRMSE', float('nan')):.1f}%  "
                f"sMAPE={overall.get('sMAPE', float('nan')):.1f}%  "
                f"BIAS_PCT={overall.get('BIAS_PCT', float('nan')):.2f}%"
            )

    if summary_rows:
        summary_path = level_evaluations_dir / "evaluation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"{level_label}: summary → {summary_path}")
    else:
        logger.warning(f"{level_label}: no models evaluated.")

    return summary_rows


def run_evaluation(config: Dict[str, Any]) -> None:
    test_dataframe = _load_test_dataframe(config)
    storage_config = config["storage"]
    base_forecasts_dir = Path(
        FORECASTS_DIR or storage_config.get("forecasts_dir", storage_config["output_dir"])
    )
    base_evaluations_dir = Path(
        EVALUATIONS_DIR or storage_config.get("evaluations_dir", storage_config["output_dir"])
    )
    base_evaluations_dir.mkdir(parents=True, exist_ok=True)

    cross_level_rows: List[Dict[str, Any]] = []

    for level_label, level_config, level_test in iter_levels(config, test_dataframe):
        if ONLY_LEVEL is not None and level_label != ONLY_LEVEL:
            continue

        with timed(f"level={level_label}", logger):
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

