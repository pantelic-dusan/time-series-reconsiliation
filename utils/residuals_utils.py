from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Type

import pandas as pd

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


def residuals_path(residuals_root: Path, level_label: str, model_name: str) -> Path:
    return Path(residuals_root) / level_label / f"{model_name}_residuals.csv"


def load_residuals(residuals_root: Path, level_label: str, model_name: str) -> pd.DataFrame:
    """Read a cached residuals CSV. Raises ``FileNotFoundError`` if absent."""
    path = residuals_path(residuals_root, level_label, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Residuals not found: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_residuals(
    level_train_df: pd.DataFrame,
    model_class: Type[ForecastModel],
    model_params: Dict[str, Any],
    checkpoint_path: Path,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Load the saved model and compute one-step-ahead in-sample residuals.

    Returns a DataFrame with columns ``ts_id, date, actual, fitted, residual``.
    Rows where the model could not produce a fitted value (NaN) are dropped;
    this naturally handles each model's own warm-up (AR lags, RNN context, etc.).
    Cross-model alignment to a common start date is handled separately by
    ``align_level_residuals``.
    """
    target_column = config["data"]["target_col"]
    time_column = config["data"]["time_col"]

    model = model_class(params=model_params).load(checkpoint_path)
    fitted_df = model.in_sample_fitted(level_train_df, config)

    actual_df = level_train_df[["ts_id", time_column, target_column]].rename(
        columns={time_column: "date", target_column: "actual"}
    )
    actual_df["date"] = pd.to_datetime(actual_df["date"])
    fitted_df["date"] = pd.to_datetime(fitted_df["date"])

    merged = actual_df.merge(fitted_df, on=["ts_id", "date"], how="inner")
    merged["residual"] = merged["actual"].astype(float) - merged["fitted"].astype(float)

    merged = (
        merged
        .sort_values(["ts_id", "date"])
        .dropna(subset=["fitted", "residual"])
        .reset_index(drop=True)
    )
    return merged[["ts_id", "date", "actual", "fitted", "residual"]]


def align_level_residuals(residuals_root: Path, level_label: str) -> None:
    """Trim all model residual files in a level to share the same start date.

    Different models have different warm-up lengths, so their residuals may
    start at different dates even within the same hierarchy level.  This
    function finds the *latest* start date (i.e. the tightest common cutoff)
    across all ``*_residuals.csv`` files for the level and trims every file
    that starts before that date, writing the result back in place.
    """
    level_dir = Path(residuals_root) / level_label
    if not level_dir.exists():
        return

    csv_files = sorted(level_dir.glob("*_residuals.csv"))
    if len(csv_files) <= 1:
        return

    file_starts: Dict[Path, "pd.Timestamp"] = {}
    for f in csv_files:
        try:
            tmp = pd.read_csv(f, usecols=["date"])
            tmp["date"] = pd.to_datetime(tmp["date"])
            file_starts[f] = tmp["date"].min()
        except Exception as exc:
            logger.warning(f"  [align] could not read {f.name}: {exc}")

    if not file_starts:
        return

    common_cutoff = max(file_starts.values())

    for f, start in file_starts.items():
        if start >= common_cutoff:
            continue
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        trimmed = df[df["date"] >= common_cutoff].reset_index(drop=True)
        trimmed.to_csv(f, index=False)
        logger.info(
            f"  [align] {level_label}/{f.name}: trimmed start "
            f"{start.date()} → {common_cutoff.date()} "
            f"({len(df) - len(trimmed)} rows removed)"
        )


def load_or_compute_residuals(
    level_train_df: pd.DataFrame,
    model_class: Type[ForecastModel],
    model_params: Dict[str, Any],
    checkpoint_path: Path,
    residuals_root: Path,
    level_label: str,
    model_name: str,
    config: Dict[str, Any],
    force: bool = False,
) -> pd.DataFrame:
    """Cached wrapper around ``compute_residuals``.

    Skips recomputation when the cache file exists and ``force`` is ``False``.
    """
    cache_path = residuals_path(residuals_root, level_label, model_name)
    if cache_path.exists() and not force:
        logger.info(f"  residuals cache hit: {cache_path}")
        return load_residuals(residuals_root, level_label, model_name)

    logger.info(f"  computing residuals: {level_label}/{model_name}")
    residuals_df = compute_residuals(
        level_train_df=level_train_df,
        model_class=model_class,
        model_params=model_params,
        checkpoint_path=checkpoint_path,
        config=config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    residuals_df.to_csv(cache_path, index=False)
    logger.info(f"  residuals saved: {cache_path} ({len(residuals_df)} rows)")
    return residuals_df
