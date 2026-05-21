"""Cross-temporal reconciliation mode (per-series monthly <-> quarterly)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.aggregation_utils import _months_per_temporal_period, get_all_temporal_levels
from utils.logging_utils import timed
from utils.reconciliation.core import (
    CROSS_TEMPORAL_METHODS,
    build_per_series_temporal_residual_matrix,
    per_series_temporal_S,
)
from utils.reconciliation.library import reconcile_one
from utils.reconciliation.modes._common import (
    read_forecast_csv,
    reconciled_filename,
    structural_level_labels,
    write_reconciled_csv,
)
from utils.residuals_utils import load_residuals


logger = logging.getLogger("reconcile")


def _cross_temporal_for_xs(
    model_name: str,
    xs_label: str,
    temporal_name: str,
    agg_factor: int,
    methods: List[str],
    forecasts_dir: Path,
    residuals_root: Path,
    nonnegative: bool,
    suffix_sep: str,
    resume: bool,
) -> None:
    """Per-series temporal reconciliation for one (xs_label, model) pair."""
    monthly_dir = forecasts_dir / xs_label
    quarterly_dir = forecasts_dir / f"temporal__{temporal_name}__{xs_label}"

    methods_to_run: List[str] = []
    targets_by_method: Dict[str, Tuple[Path, Path]] = {}
    for method in methods:
        m_path = monthly_dir / reconciled_filename(model_name, "cross_temporal", method, suffix_sep)
        q_path = quarterly_dir / reconciled_filename(model_name, "cross_temporal", method, suffix_sep)
        targets_by_method[method] = (m_path, q_path)
        if resume and m_path.exists() and q_path.exists():
            logger.info(
                f"  [{model_name}/cross_temporal/{xs_label}/{method}] targets exist, skipping"
            )
            continue
        methods_to_run.append(method)

    if not methods_to_run:
        return

    monthly_df = read_forecast_csv(monthly_dir / f"{model_name}_forecasts.csv")
    quarterly_df = read_forecast_csv(quarterly_dir / f"{model_name}_forecasts.csv")
    if monthly_df is None or quarterly_df is None:
        logger.warning(
            f"  [{model_name}/{xs_label}] missing monthly or quarterly forecast; "
            f"skipping cross_temporal"
        )
        return

    needs_residuals = any(CROSS_TEMPORAL_METHODS[m][1] for m in methods_to_run)
    monthly_res_df: Optional[pd.DataFrame] = None
    quarterly_res_df: Optional[pd.DataFrame] = None
    if needs_residuals:
        try:
            monthly_res_df = load_residuals(residuals_root, xs_label, model_name)
            quarterly_res_df = load_residuals(
                residuals_root, f"temporal__{temporal_name}__{xs_label}", model_name
            )
        except FileNotFoundError as exc:
            logger.warning(
                f"  [{model_name}/{xs_label}] {exc}; residual-based methods disabled"
            )
            methods_to_run = [m for m in methods_to_run if not CROSS_TEMPORAL_METHODS[m][1]]
            if not methods_to_run:
                return

    monthly_rows: Dict[str, List[pd.DataFrame]] = {m: [] for m in methods_to_run}
    quarterly_rows: Dict[str, List[pd.DataFrame]] = {m: [] for m in methods_to_run}

    shared_ts_ids = sorted(set(monthly_df["ts_id"]).intersection(set(quarterly_df["ts_id"])))
    if not shared_ts_ids:
        logger.warning(
            f"  [{model_name}/{xs_label}] no shared ts_ids between monthly and quarterly; skipping"
        )
        return

    residual_methods = [m for m in methods_to_run if CROSS_TEMPORAL_METHODS[m][1]]
    nonresidual_methods = [m for m in methods_to_run if not CROSS_TEMPORAL_METHODS[m][1]]
    ts_ids_ok: List[str] = []
    horizon_failures: List[str] = []
    for ts_id in shared_ts_ids:
        m_count = int((monthly_df["ts_id"] == ts_id).sum())
        q_count = int((quarterly_df["ts_id"] == ts_id).sum())
        if m_count == 0 or q_count == 0 or m_count % agg_factor != 0 or q_count != m_count // agg_factor:
            horizon_failures.append(ts_id)
        else:
            ts_ids_ok.append(ts_id)
    if horizon_failures:
        logger.warning(
            f"  [{model_name}/{xs_label}] horizon mismatch for "
            f"{len(horizon_failures)}/{len(shared_ts_ids)} series "
            f"(first 3: {horizon_failures[:3]}); skipping entire (level, model)."
        )
        return

    residual_lookup_m: Dict[str, np.ndarray] = {}
    residual_lookup_q: Dict[str, np.ndarray] = {}
    if residual_methods:
        missing_res: List[str] = []
        for ts_id in ts_ids_ok:
            m_res = (
                monthly_res_df[monthly_res_df["ts_id"] == ts_id]
                .sort_values("date")["residual"]
                .dropna()
                .to_numpy(dtype=float)
            )
            q_res = (
                quarterly_res_df[quarterly_res_df["ts_id"] == ts_id]
                .sort_values("date")["residual"]
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(m_res) < 2 or len(q_res) < 2:
                missing_res.append(ts_id)
                continue
            residual_lookup_m[ts_id] = m_res
            residual_lookup_q[ts_id] = q_res
        if missing_res:
            logger.warning(
                f"  [{model_name}/{xs_label}] {len(missing_res)}/{len(ts_ids_ok)} "
                f"series have <2 monthly or quarterly residuals (first 3: "
                f"{missing_res[:3]}); skipping residual-based methods "
                f"{residual_methods} for this (level, model)."
            )
            methods_to_run = nonresidual_methods
            residual_methods = []
            monthly_rows = {m: [] for m in methods_to_run}
            quarterly_rows = {m: [] for m in methods_to_run}
            if not methods_to_run:
                return

    first_ts = ts_ids_ok[0]
    first_m = monthly_df[monthly_df["ts_id"] == first_ts]
    monthly_horizon = int(len(first_m))
    n_q = monthly_horizon // agg_factor
    S_ts = per_series_temporal_S(monthly_horizon, agg_factor)

    n_failed = 0
    for ts_id in ts_ids_ok:
        m_rows = monthly_df[monthly_df["ts_id"] == ts_id].sort_values("date")
        q_rows = quarterly_df[quarterly_df["ts_id"] == ts_id].sort_values("date")
        monthly_dates = m_rows["date"].tolist()
        quarterly_dates = q_rows["date"].tolist()
        monthly_forecasts = m_rows["forecast"].to_numpy(dtype=float)
        quarterly_forecasts = q_rows["forecast"].to_numpy(dtype=float)
        y_hat = np.concatenate([quarterly_forecasts, monthly_forecasts])

        residual_matrix: Optional[np.ndarray] = None
        if ts_id in residual_lookup_m and ts_id in residual_lookup_q:
            try:
                residual_matrix = build_per_series_temporal_residual_matrix(
                    monthly_residuals=residual_lookup_m[ts_id],
                    quarterly_residuals=residual_lookup_q[ts_id],
                    monthly_horizon=monthly_horizon,
                    agg_factor=agg_factor,
                )
            except ValueError as exc:
                logger.warning(
                    f"  [{model_name}/{xs_label}] ts_id={ts_id!r} residual matrix "
                    f"build failed: {exc}"
                )
                residual_matrix = None

        for method in methods_to_run:
            mint_method, needs_res = CROSS_TEMPORAL_METHODS[method]
            if needs_res and residual_matrix is None:
                n_failed += 1
                continue
            try:
                y_tilde = reconcile_one(
                    S=S_ts,
                    y_hat=y_hat,
                    method_str=mint_method,
                    needs_residuals=needs_res,
                    residuals=residual_matrix if needs_res else None,
                    nonnegative=nonnegative,
                    diag_label=f"model={model_name} mode=ct xs={xs_label} ts_id={ts_id}",
                )
            except Exception as exc:
                logger.warning(
                    f"  [{model_name}/{xs_label}] ts_id={ts_id!r} method={method} "
                    f"temporal reconcile failed: {exc}"
                )
                n_failed += 1
                continue
            rec_q = y_tilde[:n_q]
            rec_m = y_tilde[n_q:]
            monthly_rows[method].append(
                pd.DataFrame({"date": monthly_dates, "forecast": rec_m, "ts_id": ts_id})
            )
            quarterly_rows[method].append(
                pd.DataFrame({"date": quarterly_dates, "forecast": rec_q, "ts_id": ts_id})
            )

    if n_failed:
        logger.warning(
            f"  [{model_name}/{xs_label}] {n_failed} series/method combos failed or were skipped"
        )

    for method in methods_to_run:
        if not monthly_rows[method] or not quarterly_rows[method]:
            logger.warning(
                f"  [{model_name}/{xs_label}/{method}] no reconciled rows produced, skipping write"
            )
            continue
        m_out = pd.concat(monthly_rows[method], ignore_index=True).sort_values(["ts_id", "date"])
        q_out = pd.concat(quarterly_rows[method], ignore_index=True).sort_values(["ts_id", "date"])
        m_path, q_path = targets_by_method[method]
        if not (resume and m_path.exists()):
            write_reconciled_csv(m_path, m_out)
        if not (resume and q_path.exists()):
            write_reconciled_csv(q_path, q_out)


def run_cross_temporal(
    config: Dict[str, Any],
    forecasts_dir: Path,
    residuals_root: Path,
    methods: List[str],
    nonnegative: bool,
    suffix_sep: str,
    resume: bool
) -> None:
    """Mode C — per-base-series temporal reconciliation (manual MinT, all 4 methods)."""
    methods = [m for m in methods if m in CROSS_TEMPORAL_METHODS]
    if not methods:
        logger.warning(
            "cross_temporal: no enabled methods are recognised "
            f"(supported: {list(CROSS_TEMPORAL_METHODS.keys())})"
        )
        return

    temporal_levels = get_all_temporal_levels(config)
    if not temporal_levels:
        logger.info("cross_temporal: no temporal_levels configured, skipping.")
        return

    xs_labels = ["base"] + structural_level_labels(config)

    for temporal_cfg in temporal_levels:
        temporal_name = temporal_cfg["name"]
        agg_factor = _months_per_temporal_period(temporal_cfg["freq"])
        for xs_label in xs_labels:
            for model_config in config["models"]:
                model_name = model_config["name"]
                with timed(f"cross_temporal/{temporal_name}/{xs_label}/{model_name}", logger):
                    _cross_temporal_for_xs(
                        model_name=model_name,
                        xs_label=xs_label,
                        temporal_name=temporal_name,
                        agg_factor=agg_factor,
                        methods=methods,
                        forecasts_dir=forecasts_dir,
                        residuals_root=residuals_root,
                        nonnegative=nonnegative,
                        suffix_sep=suffix_sep,
                        resume=resume,
                    )
