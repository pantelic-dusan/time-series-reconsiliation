import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


def _try_fit(series, *, trend, seasonal, seasonal_periods):
    """Attempt a single ExponentialSmoothing fit; return fitted model or None."""
    try:
        kwargs = {"trend": trend, "seasonal": seasonal}
        if seasonal is not None:
            kwargs["seasonal_periods"] = seasonal_periods
        return ExponentialSmoothing(series, **kwargs).fit(optimized=True)
    except Exception:
        return None


class HoltWintersModel(ForecastModel):
    """Statsmodels ExponentialSmoothing wrapper. Trains independently per series.

    Series with insufficient data for seasonal fitting fall back to naive forecast.
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="holt_winters", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Any] = {}
        self._last_dates: Dict[str, pd.Timestamp] = {}
        self._last_values: Dict[str, float] = {}

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "HoltWintersModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]

        seasonal_periods = self.params.get("seasonal_periods", 12)
        trend = self.params.get("trend", "add")
        seasonal = self.params.get("seasonal", "add")

        failed: list = []
        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column)
            series = group_dataframe.set_index(time_column)[target_column].asfreq(self._freq)
            clean = series.dropna()

            self._last_dates[ts_id] = series.index[-1]
            self._last_values[ts_id] = float(clean.iloc[-1]) if len(clean) > 0 else 0.0

            # Fallback chain: seasonal -> trend-only -> simple ES -> naive
            has_enough_for_seasonal = len(clean) >= 2 * seasonal_periods
            attempts = []
            if has_enough_for_seasonal:
                attempts.append({"trend": trend, "seasonal": seasonal,
                                 "seasonal_periods": seasonal_periods})
            attempts.append({"trend": trend, "seasonal": None, "seasonal_periods": seasonal_periods})
            attempts.append({"trend": None, "seasonal": None, "seasonal_periods": seasonal_periods})

            fitted = None
            for kwargs in attempts:
                fitted = _try_fit(series, **kwargs)
                if fitted is not None:
                    break

            if fitted is not None:
                self._fitted_models[ts_id] = fitted
            else:
                failed.append(ts_id)

        total = len(self._fitted_models) + len(failed)
        if failed:
            logger.warning(
                f"Holt-Winters: {len(failed)} series failed — using naive fallback. "
                f"First 5: {failed[:5]}"
            )
        logger.info(f"Holt-Winters: fitted {len(self._fitted_models)}/{total} series")

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
                forecast = np.asarray(fitted_model.forecast(horizon))
            else:
                forecast = np.full(horizon, self._last_values[ts_id])

            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)
