"""Reconciliation utilities package.

Public API is re-exported here so callers can keep using
``from utils.reconciliation import ...``. The package is split into:

- ``core``     : S-matrix builders, registries, long-frame helpers, residual
                 alignment.
- ``library``  : Thin wrappers around ``hierarchicalforecast.MinTrace`` plus
                 residual-validation guards.
"""

from utils.reconciliation.core import (
    CROSS_STRUCTURAL_METHODS,
    CROSS_TEMPORAL_METHODS,
    CROSS_TEMPORAL_STRUCTURAL_METHODS,
    assemble_long_frame,
    assemble_temporal_long_frame,
    build_S_and_tags,
    build_hierarchy_mapping,
    build_joint_cts_S_and_index,
    build_joint_cts_residual_matrix,
    build_per_series_temporal_residual_matrix,
    cross_structural_level_labels,
    disassemble_to_levels,
    per_series_temporal_S,
    _find_reconciled_column,
    _hf_reconciled_column,
)
from utils.reconciliation.library import (
    reconcile_one,
    validate_residuals_or_raise,
)

__all__ = [
    # registries
    "CROSS_STRUCTURAL_METHODS",
    "CROSS_TEMPORAL_METHODS",
    "CROSS_TEMPORAL_STRUCTURAL_METHODS",
    # builders + helpers
    "assemble_long_frame",
    "assemble_temporal_long_frame",
    "build_S_and_tags",
    "build_hierarchy_mapping",
    "build_joint_cts_S_and_index",
    "build_joint_cts_residual_matrix",
    "build_per_series_temporal_residual_matrix",
    "cross_structural_level_labels",
    "disassemble_to_levels",
    "per_series_temporal_S",
    "_find_reconciled_column",
    "_hf_reconciled_column",
    # library wrapper
    "reconcile_one",
    "validate_residuals_or_raise",
]
