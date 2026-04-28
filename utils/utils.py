from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """Parse a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the CSV referenced by the config and attach a joined `ts_id`."""
    data_config = config["data"]
    dataframe = pd.read_csv(
        data_config["data_path"], parse_dates=[data_config["time_col"]]
    )
    dataframe["ts_id"] = (
        dataframe[data_config["id_cols"]].astype(str).agg("_".join, axis=1)
    )
    return dataframe


_DEFAULT_TUNED_CONFIG = "config_tuned.yaml"


def tuned_config_path(config: Dict[str, Any]) -> Path:
    """Return the Path to the tuned-config file declared by `config`."""
    return Path(config.get("hpo", {}).get("output_config", _DEFAULT_TUNED_CONFIG))


def load_hpo_results(config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load `hpo_results` from the tuned-config file produced by tune.py."""
    path = tuned_config_path(config)
    if not path.exists():
        logger.info(f"No tuned-config file at {path}, using base params.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            tuned = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(f"Could not read {path} ({exc!r}); using base params.")
        return {}
    results = tuned.get("hpo_results") or {}
    if not isinstance(results, dict):
        logger.warning(f"Ignoring malformed hpo_results in {path}.")
        return {}
    n = sum(len(m) for m in results.values())
    logger.info(f"Loaded {n} tuned (level, model) overrides from {path}.")
    return results


def write_hpo_results(
    config: Dict[str, Any],
    hpo_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Path:
    """Write the minimal tuned-config file (only `hpo_results`)."""
    path = tuned_config_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"hpo_results": hpo_results}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path
