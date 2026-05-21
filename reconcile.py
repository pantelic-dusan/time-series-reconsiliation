from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hierarchicalforecast
from utils.logging_utils import setup_logging, timed
from utils.reconciliation.modes import (
    SUPPORTED_MODES,
    run_cross_structural_monthly,
    run_cross_structural_quarterly,
    run_cross_temporal,
    run_cross_temporal_structural,
)
from utils.utils import load_config


# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config/config.yaml"
LOG_FILE: str = "logs/reconcile.log"
RESUME: bool = True


logger = logging.getLogger("reconcile")


def run_reconciliation(config: Dict[str, Any], resume: bool) -> None:
    storage_config = config["storage"]
    forecasts_dir = Path(storage_config.get("forecasts_dir", storage_config["output_dir"]))
    residuals_root = Path(storage_config["residuals_dir"])

    recon_cfg = config.get("reconciliation", {}) or {}
    methods: List[str] = list(recon_cfg.get("enabled_methods", []))
    enabled_modes: List[str] = list(recon_cfg.get("enabled_modes", []))
    nonnegative: bool = bool(recon_cfg.get("nonnegative", True))
    suffix_sep: str = recon_cfg.get("filename_suffix_separator", "__")

    requested_modes: List[str] = []
    for mode in enabled_modes:
        if mode not in SUPPORTED_MODES:
            logger.warning(f"reconcile: mode '{mode}' not implemented by this script, skipping.")
            continue
        requested_modes.append(mode)

    if not requested_modes:
        logger.warning("reconcile: nothing to do (no supported modes enabled).")
        return

    common_kwargs: Dict[str, Any] = {
        "config": config,
        "forecasts_dir": forecasts_dir,
        "residuals_root": residuals_root,
        "methods": methods,
        "nonnegative": nonnegative,
        "suffix_sep": suffix_sep,
        "resume": resume,
    }

    mode_runners = {
        "cross_structural_monthly":   run_cross_structural_monthly,
        "cross_structural_quarterly": run_cross_structural_quarterly,
        "cross_temporal":             run_cross_temporal,
        "cross_temporal_structural":  run_cross_temporal_structural,
    }

    for mode in requested_modes:
        with timed(f"mode={mode}", logger):
            mode_runners[mode](**common_kwargs)


def main() -> int:
    setup_logging(LOG_FILE)
    config = load_config(CONFIG_PATH)
    try:
        with timed("full reconciliation run", logger):
            run_reconciliation(config, resume=RESUME)
    except Exception:
        logger.exception("Reconciliation run failed at top level.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
