from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from models import MODEL_REGISTRY
from utils.aggregation_utils import iter_levels
from utils.logging_utils import setup_logging, timed
from utils.residuals_utils import load_or_compute_residuals
from utils.utils import (
    load_config,
    load_hpo_results,
    load_raw_data,
    resolve_model_params,
)


# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config/config.yaml"
LOG_FILE: str = "logs/residuals.log"
RESUME: bool = True              # When True, skip residuals whose cache already exists


logger = logging.getLogger("residuals")


def run_level(
    level_label: str,
    level_config: Dict[str, Any],
    level_train: pd.DataFrame,
    checkpoint_dir: Path,
    residuals_root: Path,
    param_overrides: Optional[Dict[str, Dict[str, Any]]],
    resume: bool,
) -> None:
    """Compute residuals for every configured model at a single hierarchy level."""
    for model_config in level_config["models"]:
        model_name = model_config["name"]
        model_type = model_config.get("type", model_name)

        if model_type not in MODEL_REGISTRY:
            logger.warning(
                f"  [{model_name}] unknown model type '{model_type}', skipping."
            )
            continue

        model_params = resolve_model_params(model_config, param_overrides)
        checkpoint_path = checkpoint_dir / f"{model_name}.pkl"

        if not checkpoint_path.exists():
            logger.warning(
                f"  [{model_name}] checkpoint missing ({checkpoint_path}), skipping."
            )
            continue

        task_label = f"residuals {level_label}/{model_name}"
        try:
            with timed(task_label, logger):
                load_or_compute_residuals(
                    level_train_df=level_train,
                    model_class=MODEL_REGISTRY[model_type],
                    model_params=model_params,
                    checkpoint_path=checkpoint_path,
                    residuals_root=residuals_root,
                    level_label=level_label,
                    model_name=model_name,
                    config=level_config,
                    force=not resume,
                )
        except NotImplementedError as exc:
            logger.warning(
                f"  [{model_name}] in_sample_fitted not implemented; skipping ({exc})"
            )
            continue
        except Exception as exc:
            logger.error(
                f"  [{model_name}] residual computation failed at {level_label}: {exc}"
            )
            continue


def run_residuals(config: Dict[str, Any], resume: bool = True) -> None:
    data_config = config["data"]

    dataframe = load_raw_data(config)
    train_end_date = pd.Timestamp(config["experiment"]["train_end_date"])
    train_dataframe = dataframe[dataframe[data_config["time_col"]] <= train_end_date]

    storage_config = config["storage"]
    base_checkpoint_dir = Path(storage_config["checkpoint_dir"])
    residuals_root = Path(storage_config["residuals_dir"])
    residuals_root.mkdir(parents=True, exist_ok=True)

    all_hpo_results: Dict[str, Dict[str, Any]] = load_hpo_results(config)

    for level_label, level_config, level_train in iter_levels(config, train_dataframe):

        with timed(f"level={level_label}", logger):
            logger.info(
                f"  {level_label}: {level_train['ts_id'].nunique()} series, "
                f"{len(level_train)} train rows"
            )
            run_level(
                level_label=level_label,
                level_config=level_config,
                level_train=level_train,
                checkpoint_dir=base_checkpoint_dir / level_label,
                residuals_root=residuals_root,
                param_overrides=all_hpo_results.get(level_label),
                resume=resume,
            )

    logger.info("Residual computation complete across all hierarchy levels.")


def main() -> int:
    setup_logging(LOG_FILE)
    config = load_config(CONFIG_PATH)
    try:
        with timed("full residuals run", logger):
            run_residuals(config, resume=RESUME)
    except Exception:
        logger.exception("Residuals run failed at top level.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
