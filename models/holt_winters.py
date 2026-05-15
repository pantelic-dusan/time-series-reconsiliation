import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


class HoltWintersModel(ForecastModel):
    """Statsmodels ExponentialSmoothing wrapper. Trains independently per series."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="holt_winters", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Any] = {}
        self._last_dates: Dict[str, pd.Timestamp] = {}

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "HoltWintersModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]

        seasonal_periods = self.params.get("seasonal_periods", 12)
        trend = self.params.get("trend", "add")
        seasonal = self.params.get("seasonal", "add")

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column)
            series = group_dataframe.set_index(time_column)[target_column].asfreq(self._freq)

            self._last_dates[ts_id] = series.index[-1]

            kwargs = {"trend": trend, "seasonal": seasonal}
            if seasonal is not None:
                kwargs["seasonal_periods"] = seasonal_periods
            self._fitted_models[ts_id] = ExponentialSmoothing(series, **kwargs).fit(optimized=True)

        logger.info(f"Holt-Winters: fitted {len(self._fitted_models)} series")

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
            forecast = np.asarray(self._fitted_models[ts_id].forecast(horizon))
            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)

    def in_sample_fitted(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """One-step in-sample fitted values from ``results.fittedvalues``."""
        time_column = config["data"]["time_col"]
        target_column = config["data"]["target_col"]

        records: list[pd.DataFrame] = []
        for ts_id, group in dataframe.groupby("ts_id"):
            group = group.sort_values(time_column)
            actual = group[target_column].values.astype(float)
            dates = pd.to_datetime(group[time_column].values)

            fitted_model = self._fitted_models.get(ts_id)
            if fitted_model is None:
                fitted = actual.copy()
            else:
                fitted = np.asarray(fitted_model.fittedvalues, dtype=float)
                if fitted.shape[0] != actual.shape[0] or not np.isfinite(fitted).all():
                    fitted = np.where(np.isfinite(fitted), fitted, actual)

            records.append(pd.DataFrame({
                "ts_id": ts_id,
                "date": dates,
                "fitted": fitted,
            }))

        return pd.concat(records, ignore_index=True)
