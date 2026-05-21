"""Cross-structural reconciliation modes (monthly + quarterly)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace

from utils.logging_utils import timed
from utils.reconciliation.core import (
    CROSS_STRUCTURAL_METHODS,
    _find_reconciled_column,
    _hf_reconciled_column,
    assemble_long_frame,
    build_S_and_tags,
    build_hierarchy_mapping,
    disassemble_to_levels,
)
from utils.reconciliation.modes._common import (
    read_forecast_csv,
    reconciled_filename,
    structural_level_labels,
    quarterly_temporal_name,
    write_reconciled_csv,
)
from utils.residuals_utils import load_residuals
from utils.utils import load_raw_data


logger = logging.getLogger("reconcile")


def _cross_structural_for_model(
    model_name: str,
    mode_name: str,
    level_labels_in_S: List[str],
    input_level_prefix: str,
    output_level_prefix: str,
    methods: List[str],
    forecasts_dir: Path,
    residuals_root: Path,
    S_df: pd.DataFrame,
    tags: Dict[str, Any],
    unique_id_to_level: Dict[str, str],
    nonnegative: bool,
    suffix_sep: str,
    resume: bool,
) -> None:
    """Reconcile one model across one cross-section in a single pass."""
    on_disk_levels = [f"{input_level_prefix}{lvl}" for lvl in level_labels_in_S]
    out_levels = [f"{output_level_prefix}{lvl}" for lvl in level_labels_in_S]

    methods_to_run: List[str] = []
    target_paths_by_method: Dict[str, List[Path]] = {}
    for method in methods:
        targets = [
            forecasts_dir / out_lvl / reconciled_filename(model_name, mode_name, method, suffix_sep)
            for out_lvl in out_levels
        ]
        target_paths_by_method[method] = targets
        if resume and all(p.exists() for p in targets):
            logger.info(f"  [{model_name}/{mode_name}/{method}] all targets exist, skipping")
            continue
        methods_to_run.append(method)

    if not methods_to_run:
        return

    level_to_forecast: Dict[str, pd.DataFrame] = {}
    for in_lvl, key in zip(on_disk_levels, level_labels_in_S):
        df = read_forecast_csv(forecasts_dir / in_lvl / f"{model_name}_forecasts.csv")
        if df is None:
            logger.warning(
                f"  [{model_name}] missing forecast at level '{in_lvl}'; "
                f"skipping model for {mode_name}"
            )
            return
        level_to_forecast[key] = df

    Y_hat_df = assemble_long_frame(level_to_forecast, "forecast", model_name, "date")

    needs_residuals = any(CROSS_STRUCTURAL_METHODS[m][1] for m in methods_to_run)
    Y_df: Optional[pd.DataFrame] = None
    if needs_residuals:
        level_to_residual: Dict[str, pd.DataFrame] = {}
        residuals_ok = True
        for in_lvl, key in zip(on_disk_levels, level_labels_in_S):
            try:
                level_to_residual[key] = load_residuals(residuals_root, in_lvl, model_name)
            except FileNotFoundError as exc:
                logger.warning(f"  [{model_name}] {exc}; residual-based methods disabled")
                residuals_ok = False
                break
        if residuals_ok:
            Y_df = assemble_long_frame(level_to_residual, "residual", model_name, "date")
        else:
            methods_to_run = [m for m in methods_to_run if not CROSS_STRUCTURAL_METHODS[m][1]]
            if not methods_to_run:
                logger.warning(
                    f"  [{model_name}] no runnable methods for {mode_name} after residual check"
                )
                return

    reconcilers = [
        MinTrace(method=CROSS_STRUCTURAL_METHODS[m][0], nonnegative=nonnegative)
        for m in methods_to_run
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    reconcile_kwargs: Dict[str, Any] = {
        "Y_hat_df": Y_hat_df,
        "S": S_df,
        "tags": tags,
    }
    if Y_df is not None:
        reconcile_kwargs["Y_df"] = Y_df

    reconciled = hrec.reconcile(**reconcile_kwargs)
    if "unique_id" not in reconciled.columns:
        reconciled = reconciled.reset_index()

    for method in methods_to_run:
        mint_method = CROSS_STRUCTURAL_METHODS[method][0]
        col = _find_reconciled_column(reconciled, _hf_reconciled_column(model_name, mint_method))
        per_level = disassemble_to_levels(
            reconciled_df=reconciled,
            method_column=col,
            unique_id_to_level=unique_id_to_level,
            time_col="date",
            level_prefix=output_level_prefix,
        )
        for out_lvl, lvl_df in per_level.items():
            target = forecasts_dir / out_lvl / reconciled_filename(
                model_name, mode_name, method, suffix_sep
            )
            if resume and target.exists():
                continue
            write_reconciled_csv(target, lvl_df)


def run_cross_structural_monthly(
    config: Dict[str, Any],
    forecasts_dir: Path,
    residuals_root: Path,
    methods: List[str],
    nonnegative: bool,
    suffix_sep: str,
    resume: bool
) -> None:
    """Mode A — cross-structural reconciliation at monthly frequency only."""
    raw_df = load_raw_data(config)
    mapping = build_hierarchy_mapping(raw_df, config)
    S_df, tags, unique_id_to_level = build_S_and_tags(mapping, config)

    methods = [m for m in methods if m in CROSS_STRUCTURAL_METHODS]
    if not methods:
        logger.warning("cross_structural_monthly: no enabled methods are recognised; skipping.")
        return

    level_labels = ["base"] + structural_level_labels(config)
    for model_config in config["models"]:
        model_name = model_config["name"]
        with timed(f"cross_structural_monthly/{model_name}", logger):
            _cross_structural_for_model(
                model_name=model_name,
                mode_name="cross_structural_monthly",
                level_labels_in_S=level_labels,
                input_level_prefix="",
                output_level_prefix="",
                methods=methods,
                forecasts_dir=forecasts_dir,
                residuals_root=residuals_root,
                S_df=S_df,
                tags=tags,
                unique_id_to_level=unique_id_to_level,
                nonnegative=nonnegative,
                suffix_sep=suffix_sep,
                resume=resume,
            )


def run_cross_structural_quarterly(
    config: Dict[str, Any],
    forecasts_dir: Path,
    residuals_root: Path,
    methods: List[str],
    nonnegative: bool,
    suffix_sep: str,
    resume: bool
) -> None:
    """Mode B — cross-structural reconciliation at quarterly frequency only."""
    temporal_name = quarterly_temporal_name(config)
    if temporal_name is None:
        logger.info("cross_structural_quarterly: no temporal_levels configured, skipping.")
        return
    prefix = f"temporal__{temporal_name}__"

    raw_df = load_raw_data(config)
    mapping = build_hierarchy_mapping(raw_df, config)
    S_df, tags, unique_id_to_level = build_S_and_tags(mapping, config)

    methods = [m for m in methods if m in CROSS_STRUCTURAL_METHODS]
    if not methods:
        logger.warning("cross_structural_quarterly: no enabled methods are recognised; skipping.")
        return

    level_labels = ["base"] + structural_level_labels(config)
    for model_config in config["models"]:
        model_name = model_config["name"]
        with timed(f"cross_structural_quarterly/{model_name}", logger):
            _cross_structural_for_model(
                model_name=model_name,
                mode_name="cross_structural_quarterly",
                level_labels_in_S=level_labels,
                input_level_prefix=prefix,
                output_level_prefix=prefix,
                methods=methods,
                forecasts_dir=forecasts_dir,
                residuals_root=residuals_root,
                S_df=S_df,
                tags=tags,
                unique_id_to_level=unique_id_to_level,
                nonnegative=nonnegative,
                suffix_sep=suffix_sep,
                resume=resume,
            )
