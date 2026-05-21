from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.aggregation_utils import _months_per_temporal_period, get_all_temporal_levels
from utils.logging_utils import timed
from utils.reconciliation.core import (
    CROSS_TEMPORAL_STRUCTURAL_METHODS,
    build_hierarchy_mapping,
    build_joint_cts_S_and_index,
    build_joint_cts_residual_matrix,
)
from utils.reconciliation.library import reconcile_one
from utils.reconciliation.modes._common import (
    read_forecast_csv,
    reconciled_filename,
    structural_level_labels,
    write_reconciled_csv,
)
from utils.residuals_utils import load_residuals
from utils.utils import load_raw_data


logger = logging.getLogger("reconcile")


def _cts_on_disk_label(level: str, freq: str, temporal_name: str) -> str:
    if freq == "monthly":
        return level
    return f"temporal__{temporal_name}__{level}"


def _cross_temporal_structural_for_model(
    model_name: str,
    methods: List[str],
    config: Dict[str, Any],
    mapping: pd.DataFrame,
    S_joint: np.ndarray,
    row_index: List[Dict[str, Any]],
    monthly_horizon: int,
    agg_factor: int,
    temporal_name: str,
    forecasts_dir: Path,
    residuals_root: Path,
    nonnegative: bool,
    suffix_sep: str,
    resume: bool,
) -> None:
    """Joint cross-temporal-structural reconciliation for one model."""
    n_q = monthly_horizon // agg_factor
    structural_labels = structural_level_labels(config)
    all_keys: List[Tuple[str, str]] = (
        [("base", "monthly"), ("base", "quarterly")]
        + [(lvl, "monthly") for lvl in structural_labels]
        + [(lvl, "quarterly") for lvl in structural_labels]
    )

    methods_to_run: List[str] = []
    targets_by_method: Dict[str, Dict[Tuple[str, str], Path]] = {}
    for method in methods:
        per_key_targets: Dict[Tuple[str, str], Path] = {}
        all_exist = True
        for (lvl, freq) in all_keys:
            on_disk = _cts_on_disk_label(lvl, freq, temporal_name)
            tgt = forecasts_dir / on_disk / reconciled_filename(
                model_name, "cross_temporal_structural", method, suffix_sep
            )
            per_key_targets[(lvl, freq)] = tgt
            if not tgt.exists():
                all_exist = False
        targets_by_method[method] = per_key_targets
        if resume and all_exist:
            logger.info(
                f"  [{model_name}/cross_temporal_structural/{method}] all targets exist, skipping"
            )
            continue
        methods_to_run.append(method)

    if not methods_to_run:
        return

    base_ids = sorted(mapping["base_ts_id"].unique().tolist())
    date_lookup: Dict[Tuple[str, str], List[Any]] = {}
    forecast_arrays: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for (lvl, freq) in all_keys:
        on_disk = _cts_on_disk_label(lvl, freq, temporal_name)
        df = read_forecast_csv(forecasts_dir / on_disk / f"{model_name}_forecasts.csv")
        if df is None:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural] missing forecasts for "
                f"({lvl}, {freq}); skipping entire model."
            )
            return
        expected_steps = monthly_horizon if freq == "monthly" else n_q
        if lvl == "base":
            expected_ids = set(base_ids)
        else:
            expected_ids = set(mapping[lvl].astype(str).unique().tolist())
        per_ts_dates: Dict[str, List[Any]] = {}
        per_ts_values: Dict[str, np.ndarray] = {}
        for ts_id, sub in df.groupby("ts_id"):
            sub_sorted = sub.sort_values("date")
            per_ts_dates[str(ts_id)] = sub_sorted["date"].tolist()
            per_ts_values[str(ts_id)] = sub_sorted["forecast"].to_numpy(dtype=float)
        actual_ids = set(per_ts_values.keys())
        missing = expected_ids - actual_ids
        if missing:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural] ({lvl}, {freq}) "
                f"missing forecasts for {len(missing)} ts_ids (first 3: "
                f"{list(missing)[:3]}); skipping entire model."
            )
            return
        canonical = per_ts_dates[next(iter(expected_ids))]
        if len(canonical) != expected_steps:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural] ({lvl}, {freq}) "
                f"horizon {len(canonical)} != expected {expected_steps}; "
                f"skipping entire model."
            )
            return
        bad_dates = [tid for tid in expected_ids if per_ts_dates[tid] != canonical]
        if bad_dates:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural] ({lvl}, {freq}) "
                f"inconsistent dates across ts_ids (first 3: {bad_dates[:3]}); "
                f"skipping entire model."
            )
            return
        date_lookup[(lvl, freq)] = canonical
        forecast_arrays[(lvl, freq)] = per_ts_values

    y_hat = np.empty(len(row_index), dtype=float)
    for i, info in enumerate(row_index):
        key = (info["level"], info["freq"])
        ts_id = str(info["level_value"])
        arr = forecast_arrays[key].get(ts_id)
        if arr is None:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural] missing forecast for "
                f"({info['level']}, {info['freq']}, {ts_id}); skipping entire model."
            )
            return
        y_hat[i] = arr[info["step"] - 1]

    residual_methods = [m for m in methods_to_run if CROSS_TEMPORAL_STRUCTURAL_METHODS[m][1]]
    nonresidual_methods = [m for m in methods_to_run if not CROSS_TEMPORAL_STRUCTURAL_METHODS[m][1]]
    residual_matrix: Optional[np.ndarray] = None
    if residual_methods:
        monthly_res_by_key: Dict[Tuple[str, str], np.ndarray] = {}
        quarterly_res_by_key: Dict[Tuple[str, str], np.ndarray] = {}
        skip_residual = False
        for (lvl, freq) in all_keys:
            on_disk = _cts_on_disk_label(lvl, freq, temporal_name)
            try:
                res_df = load_residuals(residuals_root, on_disk, model_name)
            except FileNotFoundError as exc:
                logger.warning(
                    f"  [{model_name}/cross_temporal_structural] residuals missing for "
                    f"({lvl}, {freq}): {exc}; dropping residual methods {residual_methods}."
                )
                skip_residual = True
                break
            target = monthly_res_by_key if freq == "monthly" else quarterly_res_by_key
            for ts_id, sub in res_df.groupby("ts_id"):
                vals = (
                    sub.sort_values("date")["residual"].dropna().to_numpy(dtype=float)
                )
                if vals.size >= 2:
                    target[(lvl, str(ts_id))] = vals
        if not skip_residual:
            try:
                residual_matrix = build_joint_cts_residual_matrix(
                    row_index=row_index,
                    monthly_residuals_by_key=monthly_res_by_key,
                    quarterly_residuals_by_key=quarterly_res_by_key,
                    agg_factor=agg_factor,
                )
            except (KeyError, ValueError) as exc:
                logger.warning(
                    f"  [{model_name}/cross_temporal_structural] residual matrix "
                    f"build failed: {exc}; dropping residual methods {residual_methods}."
                )
                skip_residual = True
        if skip_residual:
            methods_to_run = nonresidual_methods
            residual_methods = []
            residual_matrix = None
            if not methods_to_run:
                return

    for method in methods_to_run:
        mint_method, needs_res = CROSS_TEMPORAL_STRUCTURAL_METHODS[method]
        try:
            y_tilde = reconcile_one(
                S=S_joint,
                y_hat=y_hat,
                method_str=mint_method,
                needs_residuals=needs_res,
                residuals=residual_matrix if needs_res else None,
                nonnegative=nonnegative,
                diag_label=f"model={model_name} mode=cts method={method}",
            )
        except Exception as exc:
            logger.warning(
                f"  [{model_name}/cross_temporal_structural/{method}] reconcile_one "
                f"failed: {exc}; skipping method."
            )
            continue

        per_key_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for i, info in enumerate(row_index):
            key = (info["level"], info["freq"])
            per_key_rows.setdefault(key, []).append(
                {
                    "ts_id": str(info["level_value"]),
                    "date": date_lookup[key][info["step"] - 1],
                    "forecast": float(y_tilde[i]),
                }
            )
        for key, rows in per_key_rows.items():
            tgt = targets_by_method[method][key]
            if resume and tgt.exists():
                continue
            out_df = pd.DataFrame(rows).sort_values(["ts_id", "date"])
            write_reconciled_csv(tgt, out_df)


def run_cross_temporal_structural(
    config: Dict[str, Any],
    forecasts_dir: Path,
    residuals_root: Path,
    methods: List[str],
    nonnegative: bool,
    suffix_sep: str,
    resume: bool
) -> None:
    """Mode D — joint cross-temporal-structural reconciliation."""
    unsupported = [m for m in methods if m not in CROSS_TEMPORAL_STRUCTURAL_METHODS]
    if unsupported:
        logger.warning(
            f"cross_temporal_structural: methods {unsupported} are not recognised "
            f"(supported: {list(CROSS_TEMPORAL_STRUCTURAL_METHODS.keys())})."
        )
    methods = [m for m in methods if m in CROSS_TEMPORAL_STRUCTURAL_METHODS]
    if not methods:
        logger.warning("cross_temporal_structural: no supported methods enabled; skipping mode.")
        return

    temporal_levels = get_all_temporal_levels(config)
    if not temporal_levels:
        logger.info("cross_temporal_structural: no temporal_levels configured, skipping.")
        return
    if len(temporal_levels) > 1:
        logger.warning(
            f"cross_temporal_structural: multiple temporal_levels configured "
            f"({[t['name'] for t in temporal_levels]}); using only the first."
        )
    temporal_cfg = temporal_levels[0]
    temporal_name = temporal_cfg["name"]
    agg_factor = _months_per_temporal_period(temporal_cfg["freq"])

    monthly_horizon = int(config.get("data", {}).get("horizon", 6))
    if monthly_horizon % agg_factor != 0:
        logger.warning(
            f"cross_temporal_structural: monthly_horizon={monthly_horizon} is not "
            f"divisible by agg_factor={agg_factor}; skipping mode."
        )
        return

    raw_df = load_raw_data(config)
    mapping = build_hierarchy_mapping(raw_df, config)
    with timed("cross_temporal_structural/build_joint_S", logger):
        S_joint, row_index, _bottom_index = build_joint_cts_S_and_index(
            mapping=mapping,
            monthly_horizon=monthly_horizon,
            agg_factor=agg_factor,
        )
    logger.info(
        f"cross_temporal_structural: joint S shape = {S_joint.shape} "
        f"(rows={S_joint.shape[0]}, bottom_size={S_joint.shape[1]})"
    )

    for model_config in config["models"]:
        model_name = model_config["name"]
        with timed(f"cross_temporal_structural/{model_name}", logger):
            _cross_temporal_structural_for_model(
                model_name=model_name,
                methods=methods,
                config=config,
                mapping=mapping,
                S_joint=S_joint,
                row_index=row_index,
                monthly_horizon=monthly_horizon,
                agg_factor=agg_factor,
                temporal_name=temporal_name,
                forecasts_dir=forecasts_dir,
                residuals_root=residuals_root,
                nonnegative=nonnegative,
                suffix_sep=suffix_sep,
                resume=resume,
            )
