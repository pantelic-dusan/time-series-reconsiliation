import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from models.model_interface import ForecastModel


class ProphetModel(ForecastModel):
    """Meta Prophet wrapper. Trains independently per series."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="prophet", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Prophet] = {}

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ProphetModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]

        prophet_params = {k: v for k, v in self.params.items()}

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            prophet_dataframe = group_dataframe[[time_column, target_column]].rename(
                columns={time_column: "ds", target_column: "y"}
            )
            prophet_dataframe["ds"] = pd.to_datetime(prophet_dataframe["ds"])

            model = Prophet(**prophet_params)
            model.fit(prophet_dataframe)
            self._fitted_models[ts_id] = model

        self._model = self._fitted_models
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]

        all_forecasts = []

        for ts_id, fitted_model in self._fitted_models.items():
            future = fitted_model.make_future_dataframe(periods=horizon, freq=self._freq)
            forecast = fitted_model.predict(future)
            result = forecast[["ds", "yhat"]].tail(horizon).reset_index(drop=True)
            result.columns = [time_column, "forecast"]
            result["ts_id"] = ts_id
            all_forecasts.append(result)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        path = Path(path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {ts_id: model_to_json(m) for ts_id, m in self._fitted_models.items()}
        with open(path, "w") as f:
            json.dump(serialized, f)

    def load(self, path: Path) -> "ProphetModel":
        path = Path(path).with_suffix(".json")
        with open(path, "r") as f:
            serialized = json.load(f)
        self._fitted_models = {ts_id: model_from_json(s) for ts_id, s in serialized.items()}
        self._model = self._fitted_models
        return self

