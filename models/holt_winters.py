from typing import Any, Dict

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from models.model_interface import ForecastModel


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
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            series = group_dataframe.set_index(time_column)[target_column].asfreq(self._freq)

            if len(series.dropna()) < 2 * seasonal_periods:
                continue

            model = ExponentialSmoothing(
                series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
            )
            self._fitted_models[ts_id] = model.fit(optimized=True)
            self._last_dates[ts_id] = series.index[-1]

        self._model = self._fitted_models
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]

        all_forecasts = []

        for ts_id, fitted_model in self._fitted_models.items():
            forecast = fitted_model.forecast(horizon)
            last_date = self._last_dates[ts_id]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]

            forecast_dataframe = pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast.values,
                "ts_id": ts_id,
            })
            all_forecasts.append(forecast_dataframe)

        return pd.concat(all_forecasts, ignore_index=True)
