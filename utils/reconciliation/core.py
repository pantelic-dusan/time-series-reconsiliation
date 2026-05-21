from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method registries (shared across all modes — library MinTrace supports 6).
#
# Friendly name (config key) -> (MinTrace method string, needs in-sample residuals)
# ---------------------------------------------------------------------------

_MINT_METHODS_6: Dict[str, Tuple[str, bool]] = {
    "OLS":         ("ols",         False),
    "WLS_struct":  ("wls_struct",  False),
    "WLS_var":     ("wls_var",     True),
    "MinT_shrink": ("mint_shrink", True),
    "MinT_cov":    ("mint_cov",    True),
    "EMinT":       ("emint",       True),
}

# All four modes (csm / csq / ct / cts) now share the same six-method registry.
CROSS_STRUCTURAL_METHODS:            Dict[str, Tuple[str, bool]] = dict(_MINT_METHODS_6)
CROSS_TEMPORAL_METHODS:              Dict[str, Tuple[str, bool]] = dict(_MINT_METHODS_6)
CROSS_TEMPORAL_STRUCTURAL_METHODS:   Dict[str, Tuple[str, bool]] = dict(_MINT_METHODS_6)


# ---------------------------------------------------------------------------
# Cross-structural S builder
# ---------------------------------------------------------------------------

def cross_structural_level_labels(config: Dict[str, Any]) -> List[str]:
    """Return ['structural__<name>', ..., 'base'] in S row order (aggregates first)."""
    structural = [s if isinstance(s, str) else s["name"]
                  for s in config["hierarchy"].get("structural_levels", [])]
    return [f"structural__{name}" for name in structural] + ["base"]


def build_hierarchy_mapping(raw_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Return one row per base series with its ts_id at every aggregation level."""
    id_cols = config["data"]["id_cols"]
    structural_names = [
        s if isinstance(s, str) else s["name"]
        for s in config["hierarchy"].get("structural_levels", [])
    ]

    column_levels = [n for n in structural_names if n != "total"]
    extra_cols = [n for n in column_levels if n not in id_cols]

    needed_cols = id_cols + extra_cols
    missing = [c for c in needed_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"build_hierarchy_mapping: missing columns in raw data: {missing}"
        )

    mapping = raw_df[needed_cols].drop_duplicates().reset_index(drop=True)
    mapping["base_ts_id"] = mapping[id_cols].astype(str).agg("_".join, axis=1)

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
    """Build the S_df + tags structure for HierarchicalReconciliation."""
    bottom_ids = sorted(mapping["base_ts_id"].unique().tolist())
    bottom_index = {uid: i for i, uid in enumerate(bottom_ids)}

    aggregate_columns = [c for c in mapping.columns if c.startswith("structural__")]

    rows: List[Dict[str, Any]] = []
    tags: Dict[str, List[str]] = {}
    unique_id_to_level: Dict[str, str] = {}

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
# Long-frame assembly / disassembly for the Nixtla API.
# ---------------------------------------------------------------------------

def assemble_long_frame(
    level_to_df: Dict[str, pd.DataFrame],
    value_col_in: str,
    value_col_out: str,
    time_col: str,
) -> pd.DataFrame:
    """Concatenate per-level DataFrames into the (unique_id, ds, <value>) format."""
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
    """Split a Nixtla-format reconciled frame back into per-level forecast DataFrames."""
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
# hierarchicalforecast output column naming (used only by csm/csq, which call
# HRec → MinTrace and therefore observe the library's column-naming scheme).
# ---------------------------------------------------------------------------

def _hf_reconciled_column(model_name: str, mint_method: str) -> str:
    """Output column prefix produced by hierarchicalforecast for MinTrace."""
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
# Per-series temporal S-matrix + residual-matrix builders (ct mode).
# Joint vector ordering: y = [q_1, ..., q_{n_q},  m_1, ..., m_{H_m}]
# ---------------------------------------------------------------------------

def per_series_temporal_S(monthly_horizon: int, agg_factor: int) -> np.ndarray:
    """Aggregation matrix S for one base series — quarters first, then monthly identity."""
    if monthly_horizon % agg_factor != 0:
        raise ValueError(
            f"per_series_temporal_S: monthly_horizon={monthly_horizon} is not "
            f"divisible by agg_factor={agg_factor}."
        )
    n_q = monthly_horizon // agg_factor
    S_top = np.zeros((n_q, monthly_horizon), dtype=float)
    for q in range(n_q):
        S_top[q, q * agg_factor:(q + 1) * agg_factor] = 1.0
    return np.vstack([S_top, np.eye(monthly_horizon, dtype=float)])


def build_per_series_temporal_residual_matrix(
    monthly_residuals: np.ndarray,
    quarterly_residuals: np.ndarray,
    monthly_horizon: int,
    agg_factor: int,
) -> np.ndarray:
    """Build an (K, T_q) residual matrix for one series, in the joint vector order.

    Row layout (matches ``per_series_temporal_S`` row order):
      - rows 0..n_q-1   : quarter-aggregate rows → all use the same quarterly
                          residual series of length T_q.
      - rows n_q..n_q+H_m-1 : the monthly step at intra-quarter position
                          ``p = (step-1) mod agg_factor`` uses the monthly
                          residual series at intra-quarter position ``p``,
                          i.e. ``m_grid[:, p]`` of length T_q where
                          ``m_grid = monthly_residuals[-T_q*agg:].reshape(T_q, agg)``.

    Caller is responsible for ensuring at least 2 quarters of aligned residuals;
    we still raise here as a defense-in-depth check.
    """
    n_q = monthly_horizon // agg_factor
    if quarterly_residuals.size < 2:
        raise ValueError(
            f"build_per_series_temporal_residual_matrix: <2 quarterly residuals "
            f"(got {quarterly_residuals.size})."
        )
    T_q = quarterly_residuals.size
    needed_m = T_q * agg_factor
    if monthly_residuals.size < needed_m:
        # Trim T_q to fit available monthly history.
        T_q = monthly_residuals.size // agg_factor
        if T_q < 2:
            raise ValueError(
                f"build_per_series_temporal_residual_matrix: insufficient aligned "
                f"residuals (T_q={T_q}); need ≥2."
            )
        quarterly_residuals = quarterly_residuals[-T_q:]

    m = np.asarray(monthly_residuals[-T_q * agg_factor:], dtype=float)
    q = np.asarray(quarterly_residuals[-T_q:], dtype=float)
    m_grid = m.reshape(T_q, agg_factor)  # row = historical quarter t, col = intra-quarter position

    K = n_q + monthly_horizon
    R = np.empty((K, T_q), dtype=float)
    # Quarter aggregate rows.
    for j in range(n_q):
        R[j, :] = q
    # Monthly step rows.
    for i in range(monthly_horizon):
        p = i % agg_factor
        R[n_q + i, :] = m_grid[:, p]
    return R


# ---------------------------------------------------------------------------
# Temporal long-frame for one series (still used by callers that pre-1.6 went
# through HRec; kept for backwards compatibility).
# ---------------------------------------------------------------------------

def assemble_temporal_long_frame(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    model_name: str,
    monthly_time_col: str,
    quarterly_time_col: str,
    ts_id: str,
    agg_factor: int,
) -> Tuple[pd.DataFrame, List[Any], List[Any]] | None:
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
            "ds": q_idx * agg_factor,
            "temporal_id": f"quarter_{q_idx}",
            model_name: row["forecast"],
        })

    Y_hat_df = pd.DataFrame(records)
    return Y_hat_df, monthly_dates, quarterly_dates


# ---------------------------------------------------------------------------
# Joint cross-temporal-structural reconciliation (mode D).
#
# Bottom (S columns): (base_ts_id, monthly_step) for monthly_step ∈ 1..H_m,
#     ordered by base_ts_id (sorted) then monthly_step.
#
# Rows (S rows):
#   1. (structural_level, level_value, monthly_step)
#   2. (structural_level, level_value, quarterly_step)
#   3. (base, base_ts_id, quarterly_step)
#   4. (base, base_ts_id, monthly_step) — bottom identity rows.
# ---------------------------------------------------------------------------

def build_joint_cts_S_and_index(
    mapping: pd.DataFrame,
    monthly_horizon: int,
    agg_factor: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Build the joint S matrix for cross_temporal_structural reconciliation."""
    if monthly_horizon % agg_factor != 0:
        raise ValueError(
            f"build_joint_cts_S_and_index: monthly_horizon={monthly_horizon} is not "
            f"divisible by agg_factor={agg_factor}."
        )
    n_q = monthly_horizon // agg_factor
    base_ids = sorted(mapping["base_ts_id"].unique().tolist())
    bottom_index: List[Tuple[str, int]] = [
        (bid, t) for bid in base_ids for t in range(1, monthly_horizon + 1)
    ]
    bottom_pos = {pair: j for j, pair in enumerate(bottom_index)}
    n_bottom = len(bottom_index)

    structural_cols = [c for c in mapping.columns if c.startswith("structural__")]

    rows: List[np.ndarray] = []
    row_index: List[Dict[str, Any]] = []

    def _append_row(level: str, freq: str, level_value: str, step: int, member_pairs: List[Tuple[str, int]]) -> None:
        row = np.zeros(n_bottom, dtype=float)
        for pair in member_pairs:
            row[bottom_pos[pair]] = 1.0
        rows.append(row)
        row_index.append(
            {"level": level, "freq": freq, "level_value": level_value, "step": step}
        )

    for col in structural_cols:
        for level_value, grp in mapping.groupby(col, sort=True):
            members = sorted(grp["base_ts_id"].tolist())
            for t in range(1, monthly_horizon + 1):
                _append_row(col, "monthly", str(level_value), t, [(bid, t) for bid in members])

    for col in structural_cols:
        for level_value, grp in mapping.groupby(col, sort=True):
            members = sorted(grp["base_ts_id"].tolist())
            for q in range(1, n_q + 1):
                steps = [(q - 1) * agg_factor + offset + 1 for offset in range(agg_factor)]
                pairs = [(bid, t) for bid in members for t in steps]
                _append_row(col, "quarterly", str(level_value), q, pairs)

    for bid in base_ids:
        for q in range(1, n_q + 1):
            steps = [(q - 1) * agg_factor + offset + 1 for offset in range(agg_factor)]
            _append_row("base", "quarterly", bid, q, [(bid, t) for t in steps])

    for bid in base_ids:
        for t in range(1, monthly_horizon + 1):
            _append_row("base", "monthly", bid, t, [(bid, t)])

    S = np.vstack(rows)
    return S, row_index, bottom_index


def build_joint_cts_residual_matrix(
    row_index: List[Dict[str, Any]],
    monthly_residuals_by_key: Dict[Tuple[str, str], np.ndarray],
    quarterly_residuals_by_key: Dict[Tuple[str, str], np.ndarray],
    agg_factor: int,
) -> np.ndarray:
    """Build an (K, T_q) residual matrix for the joint cts vector.

    Parameters
    ----------
    row_index : list of dicts with keys ``level``, ``freq``, ``level_value``, ``step``.
    monthly_residuals_by_key : maps ``(level, level_value)`` → 1-D monthly
        residual series (length must be a multiple of ``agg_factor``;
        truncated from the right so that ``T_q*agg_factor`` are used).
    quarterly_residuals_by_key : maps ``(level, level_value)`` → 1-D quarterly
        residual series. T_q is the minimum length across all required keys
        (so the matrix columns align across rows).
    agg_factor : monthly steps per quarter.

    The function is strict: every key referenced by ``row_index`` must be
    present in both lookups with at least ``T_q`` quarterly + ``T_q*agg`` monthly
    aligned residuals. Missing keys raise ``KeyError``.
    """
    # Determine common T_q across all keys.
    required_keys = sorted({(info["level"], str(info["level_value"])) for info in row_index})
    T_qs: List[int] = []
    for key in required_keys:
        if key not in quarterly_residuals_by_key:
            raise KeyError(f"build_joint_cts_residual_matrix: missing quarterly residuals for {key!r}")
        if key not in monthly_residuals_by_key:
            raise KeyError(f"build_joint_cts_residual_matrix: missing monthly residuals for {key!r}")
        q_len = quarterly_residuals_by_key[key].size
        m_len = monthly_residuals_by_key[key].size // agg_factor
        T_qs.append(min(q_len, m_len))
    T_q = min(T_qs)
    if T_q < 2:
        raise ValueError(
            f"build_joint_cts_residual_matrix: insufficient aligned residuals "
            f"(min T_q={T_q} across {len(required_keys)} keys); need ≥2."
        )

    K = len(row_index)
    R = np.empty((K, T_q), dtype=float)

    for i, info in enumerate(row_index):
        key = (info["level"], str(info["level_value"]))
        if info["freq"] == "quarterly":
            q = quarterly_residuals_by_key[key][-T_q:]
            R[i, :] = q
        else:
            m = monthly_residuals_by_key[key][-T_q * agg_factor:]
            m_grid = m.reshape(T_q, agg_factor)
            p = (info["step"] - 1) % agg_factor
            R[i, :] = m_grid[:, p]
    return R
