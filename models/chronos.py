import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import torch
from chronos import ChronosPipeline, ChronosBoltPipeline

from models.model_interface import ForecastModel

# Enforce offline mode so no data leaves the machine after initial download
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ChronosModel(ForecastModel):
    """Amazon Chronos zero-shot forecasting model using ChronosPipeline.
    Pre-download the model once (with internet):
        huggingface-cli download amazon/chronos-t5-small
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="chronos", params=params)
        self._freq: str | None = None
        self._series_contexts: Dict[str, tuple] = {}
        self._device: str = self.params.get("device", "cpu")
        self._model_path: str = self.params.get("model_path", "amazon/chronos-t5-small")
        self._pipeline: ChronosPipeline | None = None

    def _load_pipeline(self):
        """Load the Chronos pipeline from local HuggingFace cache. """
        pipeline_class = (
            ChronosBoltPipeline if "bolt" in self._model_path.lower()
            else ChronosPipeline
        )
        self._pipeline = pipeline_class.from_pretrained(
            self._model_path,
            device_map=self._device,
            dtype=torch.float32,
            local_files_only=True,
        )
        self._is_bolt = isinstance(self._pipeline, ChronosBoltPipeline)

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ChronosModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        context_length = self.params.get("context_length", 64)

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            values = group_dataframe[target_column].values.astype(float)
            effective_length = min(context_length, len(values))
            context = values[-effective_length:]
            last_date = pd.to_datetime(group_dataframe[time_column].iloc[-1])
            self._series_contexts[ts_id] = (context, last_date)

        self._load_pipeline()
        self._model = self._pipeline
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        ts_ids = list(self._series_contexts.keys())
        context_tensors = [
            torch.tensor(self._series_contexts[ts_id][0], dtype=torch.float32)
            for ts_id in ts_ids
        ]

        with torch.no_grad():
            forecast_samples = self._pipeline.predict(
                inputs=context_tensors,
                prediction_length=horizon,
            )


        if self._is_bolt:
            quantiles = getattr(self._pipeline, "quantiles", None)
            if quantiles is not None and 0.5 in list(quantiles):
                median_index = list(quantiles).index(0.5)
            else:
                median_index = forecast_samples.shape[1] // 2
            point_forecasts = forecast_samples[:, median_index, :].cpu().numpy()
        else:
            point_forecasts = forecast_samples.median(dim=1).values.cpu().numpy()

        for i, ts_id in enumerate(ts_ids):
            _, last_date = self._series_contexts[ts_id]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]

            forecast_dataframe = pd.DataFrame({
                time_column: future_dates,
                "forecast": point_forecasts[i],
                "ts_id": ts_id,
            })
            all_forecasts.append(forecast_dataframe)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "series_contexts": self._series_contexts,
            "freq": self._freq,
        }, path)

    def load(self, path: Path) -> "ChronosModel":
        path = Path(path)
        data = joblib.load(path)
        self._series_contexts = data["series_contexts"]
        self._freq = data["freq"]
        self._load_pipeline()
        self._model = self._pipeline
        return self
