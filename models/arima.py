import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima import ARIMA as PmdARIMA
from pmdarima import auto_arima

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


def _fit_single_series(ts_id, series, seasonal_period):
    """Fit auto_arima on a single series (for parallel execution).

    Fallback chain:
      1. auto_arima with small search space
      2. simple ARIMA(0,1,0) random walk
      3. None (caller will use naive last-value forecast)
    """
    for attempt in ("auto", "rw"):
        try:
            if attempt == "auto":
                fitted = auto_arima(
                    series,
                    seasonal=True,
                    m=seasonal_period,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=2, max_q=2,
                    max_P=1, max_Q=1,
                    max_d=1, max_D=1,
                    trace=False,
                )
            else:
                fitted = PmdARIMA(order=(0, 1, 0)).fit(series)
            return ts_id, fitted, None
        except Exception as e:
            last_error = str(e)
    return ts_id, None, last_error


class ARIMAModel(ForecastModel):
    """Auto ARIMA wrapper using pmdarima. Automatically selects the best
    (p,d,q)(P,D,Q,s) order per series via AIC minimization.

    Series that fail to fit fall back to naive forecast (last observed value).
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="arima", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Any] = {}
        self._last_dates: Dict[str, pd.Timestamp] = {}
        self._last_values: Dict[str, float] = {}  # Naive fallback values

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ARIMAModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        seasonal_period = self.params.get("seasonal_period", 12)
        n_jobs = self.params.get("n_jobs", -1)

        # Prepare all series (align to frequency, fill gaps to avoid NaN-induced fit failures)
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
            self._last_values[ts_id] = float(series.iloc[-1]) if len(series) else 0.0
            series_list.append((ts_id, series))

        logger.info(f"ARIMA: fitting {len(series_list)} series in parallel (n_jobs={n_jobs})...")

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_fit_single_series)(ts_id, series, seasonal_period)
            for ts_id, series in series_list
        )

        failed = []
        for ts_id, fitted, error in results:
            if fitted is not None:
                self._fitted_models[ts_id] = fitted
            else:
                failed.append(ts_id)

        total = len(series_list)
        if failed:
            logger.warning(
                f"ARIMA: {len(failed)}/{total} series failed to fit — using naive fallback. "
                f"First 5: {failed[:5]}"
            )
        logger.info(f"ARIMA: successfully fitted {total - len(failed)}/{total} series")

        # Persist everything needed to forecast (including naive fallback state)
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

        for ts_id, last_date in self._last_dates.items():
            future_dates = pd.date_range(
                start=last_date, periods=horizon + 1, freq=self._freq
            )[1:]

            fitted_model = self._fitted_models.get(ts_id)
            if fitted_model is not None:
                forecast = np.asarray(fitted_model.predict(n_periods=horizon))
            else:
                forecast = np.full(horizon, self._last_values[ts_id])

            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)
