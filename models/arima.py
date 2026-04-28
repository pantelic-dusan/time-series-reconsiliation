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
            "freq": self._freq,
        }
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        for ts_id, last_date in self._last_dates.items():
            future_dates = pd.date_range(
                start=last_date, periods=horizon + 1, freq=self._freq
            )[1:]
            forecast = np.asarray(self._fitted_models[ts_id].predict(n_periods=horizon))
            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)
