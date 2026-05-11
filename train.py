from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.aggregation_utils import iter_levels
from utils.utils import load_config, load_hpo_results, load_raw_data
from utils.logging_utils import setup_logging, timed
from models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Script-level configuration (edit in-place instead of using CLI flags)
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config/config.yaml"
LOG_FILE: str = "logs/train.log"
RESUME: bool = True  # Skip models whose forecast CSV already exists


logger = logging.getLogger("train")


def _cleanup_artifacts(checkpoint_path: Path, results_path: Path) -> None:
    """Remove any partial checkpoint / forecast files. Safe to call unconditionally."""
    # Plain pickle checkpoint (ARIMA, HoltWinters, RF, LGBM, Chronos, TimesFM).
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"  cleanup: could not remove {checkpoint_path}: {exc}")

    # Prophet uses <name>.json.
    json_sibling = checkpoint_path.with_suffix(".json")
    try:
        json_sibling.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"  cleanup: could not remove {json_sibling}: {exc}")

    # DeepAR / NHITS serialize into a directory at <name> (no suffix).
    directory_sibling = checkpoint_path.with_suffix("")
    if directory_sibling.is_dir():
        shutil.rmtree(directory_sibling, ignore_errors=True)

    # Partial forecast CSV.
    try:
        results_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"  cleanup: could not remove {results_path}: {exc}")


def _check_coverage(
    train_dataframe: pd.DataFrame,
    forecast_dataframe: pd.DataFrame,
    horizon: int,
    model_name: str,
) -> None:
    """Hard-fail if any input ts_id is missing from the forecast or has wrong row count."""
    expected = set(train_dataframe["ts_id"].unique())
    got = set(forecast_dataframe["ts_id"].unique())

    missing = expected - got
    if missing:
        raise RuntimeError(
            f"{model_name}: missing forecasts for {len(missing)}/{len(expected)} "
            f"series. First 5: {sorted(list(missing))[:5]}"
        )

    row_counts = forecast_dataframe.groupby("ts_id").size()
    wrong = row_counts[row_counts != horizon]
    if len(wrong) > 0:
        raise RuntimeError(
            f"{model_name}: {len(wrong)} series have row count != horizon ({horizon}). "
            f"First 5: {wrong.head(5).to_dict()}"
        )

    extra = got - expected
    if extra:
        logger.warning(
            f"{model_name}: forecast contains {len(extra)} ts_ids not present in train "
            f"(ignored). First 5: {sorted(list(extra))[:5]}"
        )


def run_level(
    config: Dict[str, Any],
    train_dataframe: pd.DataFrame,
    output_dir: Path,
    checkpoint_dir: Path,
    level_label: str,
    resume: bool = False,
    param_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    """Fit + predict every configured model for one hierarchy level.

    Forecasts are clipped to zero. On coverage-check failure the checkpoint and
    forecast CSV for that model are deleted so nothing partial survives.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    horizon = config["experiment"]["horizon"]

    for model_config in config["models"]:
        model_name = model_config["name"]
        model_type = model_config.get("type", model_name)
        model_params = dict(model_config.get("params", {}))

        if param_overrides:
            inherit_key = model_config.get("hpo_inherit_from")
            override_source = inherit_key if inherit_key else model_name
            overrides = dict(param_overrides.get(override_source, {}))
            if inherit_key:
                overrides.pop("strategy", None)
            if overrides:
                model_params = {**model_params, **overrides}
                src_label = f" (inherited from '{inherit_key}')" if inherit_key else ""
                logger.info(f"  [{model_name}] applying HPO params{src_label}: {overrides}")

        if model_type not in MODEL_REGISTRY:
            logger.warning(f"Unknown model type '{model_type}' (entry '{model_name}'), skipping.")
            continue

        results_path = output_dir / f"{model_name}_forecasts.csv"
        checkpoint_path = checkpoint_dir / f"{model_name}.pkl"

        if resume and results_path.exists():
            logger.info(f"[SKIP] {level_label}/{model_name} — forecast exists at {results_path}")
            continue

        task_label = f"{level_label}/{model_name}"

        try:
            with timed(task_label, logger):
                model = MODEL_REGISTRY[model_type](params=model_params)

                used_checkpoint = False
                if resume and checkpoint_path.exists():
                    logger.info(f"  resuming from checkpoint: {checkpoint_path}")
                    try:
                        model.load(checkpoint_path)
                        forecast_dataframe = model.predict(horizon, config)
                        used_checkpoint = True
                    except Exception as exc:
                        logger.warning(f"  checkpoint load/predict failed, retraining: {exc}")

                if not used_checkpoint:
                    model.fit(train_dataframe, config)
                    forecast_dataframe = model.predict(horizon, config)

                forecast_dataframe["forecast"] = forecast_dataframe["forecast"].clip(lower=0)

                # Hard-fail if not every input ts_id got a full horizon of predictions.
                _check_coverage(train_dataframe, forecast_dataframe, horizon, model_name)

                if not used_checkpoint:
                    model.save(checkpoint_path)
                forecast_dataframe.to_csv(results_path, index=False)
                logger.info(f"  saved forecasts → {results_path}")

        except Exception as exc:
            _cleanup_artifacts(checkpoint_path, results_path)
            logger.error(f"  {task_label} aborted; partial artifacts removed ({exc})")
            # Continue with the next model — do not let one model block the rest.
            continue


def run_experiment(config: Dict[str, Any], resume: bool = False) -> None:
    data_config = config["data"]

    dataframe = load_raw_data(config)

    train_end_date = pd.Timestamp(config["experiment"]["train_end_date"])
    train_dataframe = dataframe[dataframe[data_config["time_col"]] <= train_end_date]

    storage_config = config["storage"]
    base_output_dir = Path(storage_config.get("forecasts_dir", storage_config["output_dir"]))
    base_checkpoint_dir = Path(config["storage"]["checkpoint_dir"])

    all_hpo_results: Dict[str, Dict[str, Any]] = load_hpo_results(config)

    for level_label, level_config, level_train in iter_levels(config, train_dataframe):
        with timed(f"level={level_label}", logger):
            logger.info(
                f"  {level_label}: {level_train['ts_id'].nunique()} series, "
                f"{len(level_train)} train rows"
            )
            run_level(
                config=level_config,
                train_dataframe=level_train,
                output_dir=base_output_dir / level_label,
                checkpoint_dir=base_checkpoint_dir / level_label,
                level_label=level_label,
                resume=resume,
                param_overrides=all_hpo_results.get(level_label),
            )

    logger.info("Experiment complete across all hierarchy levels.")


def main() -> int:
    setup_logging(LOG_FILE)
    config = load_config(CONFIG_PATH)
    try:
        with timed("full training run", logger):
            run_experiment(config, resume=RESUME)
    except Exception:
        logger.exception("Training run failed at top level.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

