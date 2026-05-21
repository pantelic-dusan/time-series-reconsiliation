from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hierarchicalforecast.methods import MinTrace

logger = logging.getLogger(__name__)


def validate_residuals_or_raise(
    residuals: Optional[np.ndarray],
    t_min: int,
    label: str,
) -> None:
    """Raise ``ValueError`` if ``residuals`` is None, empty, or has fewer than
    ``t_min`` observations along the time axis.

    For a 2-D residual matrix ``(K, T)`` the time axis is ``T = shape[1]``;
    for a 1-D vector it is ``len(residuals)``.
    """
    if residuals is None:
        raise ValueError(f"{label}: residuals are None.")
    arr = np.asarray(residuals)
    if arr.ndim == 1:
        T = arr.size
    elif arr.ndim == 2:
        T = arr.shape[1]
    else:
        raise ValueError(f"{label}: residuals must be 1-D or 2-D, got ndim={arr.ndim}.")
    if T < t_min:
        raise ValueError(f"{label}: residuals have T={T} < t_min={t_min}.")


def reconcile_one(
    S: np.ndarray,
    y_hat: np.ndarray,
    method_str: str,
    needs_residuals: bool,
    residuals: Optional[np.ndarray] = None,
    nonnegative: bool = True,
    mint_shr_ridge: float = 2e-08,
    num_threads: int = 1,
    diag_label: str = "",
) -> np.ndarray:
    """Reconcile a single forecast horizon using ``MinTrace``.

    Parameters
    ----------
    S : ``(K, n_bottom)`` summing matrix.
    y_hat : ``(K,)`` base forecast vector in row order of ``S``.
    method_str : library method name — one of ``ols``, ``wls_struct``,
        ``wls_var``, ``mint_shrink``, ``mint_cov``, ``emint``.
    needs_residuals : if True, ``residuals`` must be a ``(K, T)`` matrix.
    residuals : pre-computed residual matrix; passed as ``y_insample`` with
        ``y_hat_insample = zeros_like(residuals)`` so the library's internal
        ``y - y_hat`` recovers the residuals.
    nonnegative : enable library nonnegative QP.
    mint_shr_ridge : library ridge regularization for shrinkage covariance.
    num_threads : passed to MinTrace (OpenMP backend for covariance / QP).
    diag_label : optional tag included in the INFO log line for grep-ability,
        e.g. ``"model=arima mode=ct ts_id=cust_42"``.

    Returns
    -------
    y_tilde : ``(K,)`` reconciled forecast vector, in the same row order as
        ``y_hat``.
    """
    if needs_residuals:
        validate_residuals_or_raise(residuals, t_min=2, label=f"reconcile_one[{method_str}]")
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim != 2 or residuals.shape[0] != S.shape[0]:
            raise ValueError(
                f"reconcile_one[{method_str}]: residuals shape={residuals.shape} does not "
                f"match S rows={S.shape[0]}."
            )
        y_insample = residuals
        y_hat_insample = np.zeros_like(residuals)
        T = residuals.shape[1]
    else:
        y_insample = None
        y_hat_insample = None
        T = 0

    K = S.shape[0]
    logger.info(
        f"  reconcile_one[{method_str}] {diag_label} K={K} T={T} "
        f"ridge={mint_shr_ridge:g} nonneg={nonnegative}"
    )

    reconciler = MinTrace(
        method=method_str,
        nonnegative=nonnegative,
        mint_shr_ridge=mint_shr_ridge,
        num_threads=num_threads,
    )
    # MinTrace.fit_predict expects 2-D y_hat of shape (K, horizon). For a single
    # horizon column, wrap and unwrap.
    y_hat_2d = np.asarray(y_hat, dtype=float).reshape(-1, 1)
    out = reconciler.fit_predict(
        S=np.asarray(S, dtype=float),
        y_hat=y_hat_2d,
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
    )
    # MinTrace returns a dict with key "mean" (and CI keys if level is set).
    if isinstance(out, dict):
        mean = out.get("mean", out)
    else:
        mean = out
    return np.asarray(mean, dtype=float).reshape(-1)
