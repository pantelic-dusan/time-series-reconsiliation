import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from models.model_interface import ForecastModel

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class TimesFMModel(ForecastModel):
    """Google TimesFM zero-shot forecasting model. Fully local inference."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="timesfm", params=params)
        self._freq: str | None = None
        self._series_contexts: Dict[str, tuple] = {}
        self._model_path: str = self.params.get("model_path", "google/timesfm-1.0-200m")

    def _load_model(self, horizon: int):
        """Load TimesFM from local HuggingFace cache."""
        import timesfm
        self._tfm = timesfm.TimesFm(
            context_len=self.params.get("context_length", 128),
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        self._tfm.load_from_checkpoint(repo_id=self._model_path)

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "TimesFMModel":
        """TimesFM is zero-shot; fit stores context windows per series."""
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        context_length = self.params.get("context_length", 128)

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            context = group_dataframe[target_column].values[-context_length:].astype(float)
            last_date = pd.to_datetime(group_dataframe[time_column].iloc[-1])
            self._series_contexts[ts_id] = (context, last_date)

        horizon = config["experiment"]["horizon"]
        self._load_model(horizon)
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        # Batch all contexts for efficient inference
        ts_ids = list(self._series_contexts.keys())
        contexts = [self._series_contexts[tid][0] for tid in ts_ids]

        frequency_input = [0] * len(contexts)  # 0 = monthly in TimesFM convention
        forecast_output = self._tfm.forecast(contexts, freq=frequency_input)

        for index, ts_id in enumerate(ts_ids):
            _, last_date = self._series_contexts[ts_id]
            predictions = forecast_output[index, :horizon]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]

            forecast = pd.DataFrame({
                time_column: future_dates,
                "forecast": predictions,
                "ts_id": ts_id,
            })
            all_forecasts.append(forecast)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"series_contexts": self._series_contexts, "freq": self._freq}, path)

    def load(self, path: Path) -> "TimesFMModel":
        import joblib
        path = Path(path)
        data = joblib.load(path)
        self._series_contexts = data["series_contexts"]
        self._freq = data["freq"]
        return self
