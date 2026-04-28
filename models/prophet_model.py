import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


def _fit_single_series(ts_id, prophet_dataframe, prophet_kwargs):
    """Fit one Prophet model on a single series (for parallel execution)."""
    model = Prophet(**prophet_kwargs)
    model.fit(prophet_dataframe)
    return ts_id, model


class ProphetModel(ForecastModel):
    """Meta Prophet wrapper. Trains independently per series, in parallel."""

    # Keys consumed by the wrapper itself; everything else is forwarded to Prophet.
    _WRAPPER_KEYS = {"n_jobs"}

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="prophet", params=params)
        self._freq: str | None = None
        self._fitted_models: Dict[str, Prophet] = {}

    def _prophet_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in self.params.items() if k not in self._WRAPPER_KEYS}

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ProphetModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        n_jobs = self.params.get("n_jobs", -1)

        prophet_kwargs = self._prophet_kwargs()

        series_list = []
        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            prophet_dataframe = group_dataframe[[time_column, target_column]].rename(
                columns={time_column: "ds", target_column: "y"}
            )
            prophet_dataframe["ds"] = pd.to_datetime(prophet_dataframe["ds"])
            series_list.append((ts_id, prophet_dataframe))

        logger.info(
            f"Prophet: fitting {len(series_list)} series in parallel (n_jobs={n_jobs})..."
        )

        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_fit_single_series)(ts_id, prophet_dataframe, prophet_kwargs)
            for ts_id, prophet_dataframe in series_list
        )

        for ts_id, model in results:
            self._fitted_models[ts_id] = model

        logger.info(f"Prophet: fitted {len(self._fitted_models)} series")

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

