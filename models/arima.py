import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima import auto_arima

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


def _fit_single_series(ts_id, series, seasonal_period):
    """Fit auto_arima on a single series (for parallel execution)."""
    kwargs = dict(
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        seasonal_test="ocsb",
        max_p=2, max_q=2,
        max_P=1, max_Q=1,
        max_d=1, max_D=1,
        trace=False,
    )
    try:
        fitted = auto_arima(series, **kwargs)
    except ValueError as exc:
        if "singular matrices" in str(exc):
            logger.warning(
                f"ARIMA[{ts_id}]: OCSB failed at m={seasonal_period} "
                f"({exc}); retrying with D=0."
            )
            kwargs["D"] = 0
            fitted = auto_arima(series, **kwargs)
        else:
            raise
    return ts_id, fitted


class ARIMAModel(ForecastModel):
    """Auto ARIMA wrapper using pmdarima. Automatically selects the best (p,d,q)(P,D,Q,s) order per series via AIC minimization."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="arima", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Any] = {}
        self._last_dates: Dict[str, pd.Timestamp] = {}
        self._last_values: Dict[str, float] = {}

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ARIMAModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        seasonal_period = self.params.get("seasonal_period", 12)
        n_jobs = self.params.get("n_jobs", -1)

        series_list = []
        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column)
            series = (
                group_dataframe.set_index(time_column)[target_column]
                .asfreq(self._freq)
                .ffill()
                .bfill()
            )
            self._last_dates[ts_id] = series.index[-1]
            self._last_values[ts_id] = float(series.iloc[-1])
            series_list.append((ts_id, series))

        logger.info(f"ARIMA: fitting {len(series_list)} series in parallel (n_jobs={n_jobs})...")

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_fit_single_series)(ts_id, series, seasonal_period)
            for ts_id, series in series_list
        )

        for ts_id, fitted in results:
            self._fitted_models[ts_id] = fitted

        logger.info(f"ARIMA: fitted {len(self._fitted_models)} series")

        self._model = {
            "fitted_models": self._fitted_models,
            "last_dates": self._last_dates,
            "last_values": self._last_values,
            "freq": self._freq,
        }
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []
        fallback_ts_ids: list[str] = []

        for ts_id, last_date in self._last_dates.items():
            future_dates = pd.date_range(
                start=last_date, periods=horizon + 1, freq=self._freq
            )[1:]
            forecast: np.ndarray
            try:
                forecast = np.asarray(
                    self._fitted_models[ts_id].predict(n_periods=horizon)
                )
                if not np.isfinite(forecast).all():
                    raise ValueError("non-finite forecast")
            except Exception as exc:
                # Per-series naive fallback: repeat the last observed training
                # value. Required because pmdarima's internal conf-int checker
                # raises on NaN forecasts for some degenerate series.
                fallback_ts_ids.append(str(ts_id))
                last_value = self._last_values.get(ts_id, 0.0)
                if not np.isfinite(last_value):
                    last_value = 0.0
                forecast = np.full(horizon, last_value, dtype=float)
                logger.debug(f"ARIMA[{ts_id}]: naive fallback ({exc!r})")

            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast,
                "ts_id": ts_id,
            }))

        if fallback_ts_ids:
            n = len(fallback_ts_ids)
            sample = ", ".join(fallback_ts_ids[:5])
            logger.warning(
                f"ARIMA: naive last-value fallback used for "
                f"{n}/{len(self._last_dates)} series at predict time. "
                f"First {min(n, 5)}: [{sample}]"
            )

        return pd.concat(all_forecasts, ignore_index=True)
