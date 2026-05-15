from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# All level labels that participate in the cross-structural S matrix, in the
# row order used inside S (bottom rows last).
def cross_structural_level_labels(config: Dict[str, Any]) -> List[str]:
    """Return ['structural__<name>', ..., 'base'] in S row order (aggregates first)."""
    structural = [s if isinstance(s, str) else s["name"]
                  for s in config["hierarchy"].get("structural_levels", [])]
    return [f"structural__{name}" for name in structural] + ["base"]


def build_hierarchy_mapping(raw_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Return one row per base series with its ts_id at every aggregation level.

    Columns: base_ts_id, structural__<name>, ... for every level in
    config.hierarchy.structural_levels. Each value is the ts_id used inside
    that level's forecast files (matches `aggregate_structural` output).
    """
    id_cols = config["data"]["id_cols"]
    structural_names = [
        s if isinstance(s, str) else s["name"]
        for s in config["hierarchy"].get("structural_levels", [])
    ]

    # Non-total levels that need a raw data column (not synthesised as "total").
    column_levels = [n for n in structural_names if n != "total"]
    # Levels that are not already in id_cols must be present as extra columns.
    extra_cols = [n for n in column_levels if n not in id_cols]

    needed_cols = id_cols + extra_cols
    missing = [c for c in needed_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"build_hierarchy_mapping: missing columns in raw data: {missing}"
        )

    mapping = raw_df[needed_cols].drop_duplicates().reset_index(drop=True)

    # Base ts_id matches load_raw_data() in utils/utils.py: id_cols joined by '_'.
    mapping["base_ts_id"] = mapping[id_cols].astype(str).agg("_".join, axis=1)

    # Single-column structural levels expose the column value directly as ts_id
    # (matches `aggregate_structural` with a 1-element level_columns list).
    for col in column_levels:
        mapping[f"structural__{col}"] = mapping[col].astype(str)

    if "total" in structural_names:
        mapping["structural__total"] = "total"

    keep = ["base_ts_id"] + [f"structural__{n}" for n in structural_names]
    return mapping[keep]


def build_S_and_tags(
    mapping: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
    """Build the S_df + tags structure for HierarchicalReconciliation.

    Returns
    -------
    S_df : DataFrame
        Long format with columns ['unique_id', <bottom_id_1>, ...]; each row is a
        series (aggregate or bottom) with 0/1 indicators for which bottom series
        it sums.
    tags : dict[str, np.ndarray]
        Maps each level label (e.g. 'structural__customer', 'base') to the array
        of unique_ids belonging to that level, in the same order as in S_df.
    unique_id_to_level : dict[str, str]
        Reverse lookup unique_id -> level label, used during disassembly.
    """
    bottom_ids = sorted(mapping["base_ts_id"].unique().tolist())
    bottom_index = {uid: i for i, uid in enumerate(bottom_ids)}

    aggregate_columns = [c for c in mapping.columns if c.startswith("structural__")]

    rows: List[Dict[str, Any]] = []
    tags: Dict[str, List[str]] = {}
    unique_id_to_level: Dict[str, str] = {}

    # --- Aggregate levels first (matches hierarchicalforecast convention) ---
    for level_label in [f"structural__{name}"
                        for name in (s if isinstance(s, str) else s["name"]
                                     for s in config["hierarchy"].get("structural_levels", []))]:
        if level_label not in aggregate_columns:
            continue
        level_tags: List[str] = []
        groups = mapping.groupby(level_label, sort=True)
        for level_value, group in groups:
            row = np.zeros(len(bottom_ids), dtype=float)
            for base_id in group["base_ts_id"]:
                row[bottom_index[base_id]] = 1.0
            record = {"unique_id": level_value}
            for j, col in enumerate(bottom_ids):
                record[col] = row[j]
            rows.append(record)
            level_tags.append(level_value)
            unique_id_to_level[level_value] = level_label
        tags[level_label] = np.array(level_tags, dtype=object)

    # --- Bottom level last ---
    base_tags: List[str] = []
    for i, base_id in enumerate(bottom_ids):
        row = np.zeros(len(bottom_ids), dtype=float)
        row[i] = 1.0
        record = {"unique_id": base_id}
        for j, col in enumerate(bottom_ids):
            record[col] = row[j]
        rows.append(record)
        base_tags.append(base_id)
        unique_id_to_level[base_id] = "base"
    tags["base"] = np.array(base_tags, dtype=object)

    S_df = pd.DataFrame(rows, columns=["unique_id"] + bottom_ids)
    logger.info(
        f"build_S_and_tags: S shape={S_df.shape[0]}x{len(bottom_ids)} "
        f"(rows = aggregates + bottom; cols = bottom)"
    )
    return S_df, tags, unique_id_to_level


# ---------------------------------------------------------------------------
# Long-format assembly / disassembly for the Nixtla API.
# ---------------------------------------------------------------------------

def assemble_long_frame(
    level_to_df: Dict[str, pd.DataFrame],
    value_col_in: str,
    value_col_out: str,
    time_col: str,
) -> pd.DataFrame:
    """Concatenate per-level DataFrames into the (unique_id, ds, <value>) format.

    Each input DataFrame must already have columns ['ts_id', time_col, value_col_in].
    Output columns: ['unique_id', 'ds', value_col_out].
    """
    pieces: List[pd.DataFrame] = []
    for level_label, df in level_to_df.items():
        if df is None or df.empty:
            continue
        sub = df[["ts_id", time_col, value_col_in]].rename(
            columns={"ts_id": "unique_id", time_col: "ds", value_col_in: value_col_out}
        )
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame(columns=["unique_id", "ds", value_col_out])
    out = pd.concat(pieces, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"])
    return out


def disassemble_to_levels(
    reconciled_df: pd.DataFrame,
    method_column: str,
    unique_id_to_level: Dict[str, str],
    time_col: str,
    level_prefix: str = "",
) -> Dict[str, pd.DataFrame]:
    """Split a Nixtla-format reconciled frame back into per-level forecast DataFrames.

    The output schema matches the existing `<level>/<model>_forecasts.csv` files:
    `date, forecast, ts_id`. ``level_prefix`` is prepended to each level label
    (e.g. ``"temporal__quarter__"``) so the output keys can be used directly as
    on-disk directory names for temporal cross-sections.
    """
    df = reconciled_df.copy()
    df["__level"] = df["unique_id"].map(unique_id_to_level)
    if df["__level"].isna().any():
        unmapped = df.loc[df["__level"].isna(), "unique_id"].unique()
        raise ValueError(
            f"disassemble_to_levels: {len(unmapped)} unique_ids have no level mapping. "
            f"First 5: {list(unmapped)[:5]}"
        )

    out: Dict[str, pd.DataFrame] = {}
    for level_label, group in df.groupby("__level", sort=False):
        per_level = group[["unique_id", "ds", method_column]].rename(
            columns={"unique_id": "ts_id", "ds": time_col, method_column: "forecast"}
        )
        per_level[time_col] = pd.to_datetime(per_level[time_col])
        per_level = per_level.sort_values(["ts_id", time_col]).reset_index(drop=True)
        out[f"{level_prefix}{level_label}"] = per_level
    return out


# ---------------------------------------------------------------------------
# Reconciliation method registry.
# ---------------------------------------------------------------------------

# Friendly name (config key) -> (MinTrace method string, needs in-sample residuals)
CROSS_STRUCTURAL_METHODS: Dict[str, Tuple[str, bool]] = {
    "OLS":         ("ols",         False),
    "WLS_struct":  ("wls_struct",  False),
    "WLS_var":     ("wls_var",     True),
    "MinT_shrink": ("mint_shrink", True),
}

# Methods supported by hierarchicalforecast 1.5.1 with temporal=True.
# WLS_var / MinT_shrink require insample residuals, rejected when temporal=True.
CROSS_TEMPORAL_METHODS: Dict[str, str] = {
    "OLS":        "ols",
    "WLS_struct": "wls_struct",
}


def _hf_reconciled_column(model_name: str, mint_method: str) -> str:
    """Output column prefix produced by hierarchicalforecast for MinTrace.

    When nonnegative=True the library appends '_nonnegative-True'; use
    ``_find_reconciled_column`` for robust lookups against actual output columns.
    """
    return f"{model_name}/MinTrace_method-{mint_method}"


def _find_reconciled_column(reconciled_df: pd.DataFrame, prefix: str) -> str:
    """Find the single reconciled column that starts with ``prefix``.

    Handles the optional '_nonnegative-True' suffix added by hierarchicalforecast
    when nonnegative=True.
    """
    matches = [c for c in reconciled_df.columns if c == prefix or c.startswith(prefix + "_")]
    if not matches:
        raise RuntimeError(
            f"Expected output column with prefix '{prefix}' missing. "
            f"Got: {list(reconciled_df.columns)}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous output columns for prefix '{prefix}': {matches}."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Temporal S-matrix helpers.
# ---------------------------------------------------------------------------

def _build_temporal_S_and_tags(
    agg_factor: int,
    monthly_horizon: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Build the temporal S matrix for a *single* series repeated across all ts_ids.

    For ``agg_factor=3`` (quarterly) and ``monthly_horizon=6`` the bottom columns
    are the 6 monthly step indices ('step_1'…'step_6').

    S_df columns: ['temporal_id', 'step_1', …, 'step_<monthly_horizon>']
    Rows:
      - one aggregate row per quarter  (temporal_id = 'quarter_1', 'quarter_2', …)
      - one bottom row per monthly step (temporal_id = 'monthly_1', …)

    tags:
      - 'quarterly': array of quarter temporal_id strings
      - 'monthly': array of monthly temporal_id strings
    """
    n_agg = monthly_horizon // agg_factor  # e.g. 6 // 3 = 2
    if monthly_horizon % agg_factor != 0:
        raise ValueError(
            f"_build_temporal_S_and_tags: monthly_horizon={monthly_horizon} is not "
            f"divisible by agg_factor={agg_factor}."
        )

    bottom_cols = [f"step_{i+1}" for i in range(monthly_horizon)]
    rows: List[Dict[str, Any]] = []

    # Aggregate rows (quarters)
    agg_ids: List[str] = []
    for q in range(n_agg):
        row: Dict[str, Any] = {"temporal_id": f"quarter_{q+1}"}
        for i, col in enumerate(bottom_cols):
            row[col] = 1.0 if q * agg_factor <= i < (q + 1) * agg_factor else 0.0
        rows.append(row)
        agg_ids.append(f"quarter_{q+1}")

    # Bottom rows (monthly steps)
    monthly_ids: List[str] = []
    for i in range(monthly_horizon):
        row = {"temporal_id": f"monthly_{i+1}"}
        for j, col in enumerate(bottom_cols):
            row[col] = 1.0 if j == i else 0.0
        rows.append(row)
        monthly_ids.append(f"monthly_{i+1}")

    S_df = pd.DataFrame(rows, columns=["temporal_id"] + bottom_cols)
    tags: Dict[str, np.ndarray] = {
        "quarterly": np.array(agg_ids, dtype=object),
        "monthly": np.array(monthly_ids, dtype=object),
    }
    return S_df, tags


def assemble_temporal_long_frame(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    model_name: str,
    monthly_time_col: str,
    quarterly_time_col: str,
    ts_id: str,
    agg_factor: int,
) -> Tuple[pd.DataFrame, List[Any], List[Any]] | None:
    """Build a Nixtla-style temporal Y_hat_df for one series.

    Returns (Y_hat_df, sorted_monthly_dates, sorted_quarterly_dates) or None
    if the series cannot be reconciled (mismatched horizons, etc.).

    Y_hat_df columns: ['unique_id', 'ds', 'temporal_id', model_name]
    where unique_id is always ts_id, ds is the step index (1-based integer),
    and temporal_id is 'monthly_1'…'monthly_N' or 'quarter_1'…'quarter_M'.
    """
    m_rows = monthly_df[monthly_df["ts_id"] == ts_id].sort_values(monthly_time_col)
    q_rows = quarterly_df[quarterly_df["ts_id"] == ts_id].sort_values(quarterly_time_col)

    monthly_horizon = len(m_rows)
    quarterly_horizon = len(q_rows)

    if monthly_horizon == 0 or quarterly_horizon == 0:
        logger.warning(
            f"assemble_temporal_long_frame: ts_id={ts_id!r} has no rows in one "
            f"of the frequency frames (monthly={monthly_horizon}, "
            f"quarterly={quarterly_horizon}) — skipping."
        )
        return None

    expected_q = monthly_horizon // agg_factor
    if monthly_horizon % agg_factor != 0 or quarterly_horizon != expected_q:
        logger.warning(
            f"assemble_temporal_long_frame: ts_id={ts_id!r} horizon mismatch — "
            f"monthly={monthly_horizon}, quarterly={quarterly_horizon}, "
            f"agg_factor={agg_factor} (expected {expected_q} quarters) — skipping."
        )
        return None

    records: List[Dict[str, Any]] = []
    monthly_dates = m_rows[monthly_time_col].tolist()
    quarterly_dates = q_rows[quarterly_time_col].tolist()

    for step_idx, (_, row) in enumerate(m_rows.iterrows(), start=1):
        records.append({
            "unique_id": ts_id,
            "ds": step_idx,
            "temporal_id": f"monthly_{step_idx}",
            model_name: row["forecast"],
        })
    for q_idx, (_, row) in enumerate(q_rows.iterrows(), start=1):
        records.append({
            "unique_id": ts_id,
            "ds": q_idx * agg_factor,  # map quarter index to last monthly step in that quarter
            "temporal_id": f"quarter_{q_idx}",
            model_name: row["forecast"],
        })

    Y_hat_df = pd.DataFrame(records)
    return Y_hat_df, monthly_dates, quarterly_dates
