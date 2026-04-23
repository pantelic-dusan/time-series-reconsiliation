"""Amazon Chronos zero-shot forecasting model wrapper."""

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

    Chronos is a pretrained foundation model — it requires no training.
    fit() stores context windows per series, predict() generates forecasts
    using the pretrained weights from the local HuggingFace cache.

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
        """Load the Chronos pipeline from local HuggingFace cache.

        Automatically selects ChronosBoltPipeline for Bolt models
        and ChronosPipeline for original T5 models.
        """
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
        """Chronos is zero-shot; fit() stores context windows per series."""
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        context_length = self.params.get("context_length", 64)

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            context = group_dataframe[target_column].values[-context_length:].astype(float)
            last_date = pd.to_datetime(group_dataframe[time_column].iloc[-1])
            self._series_contexts[ts_id] = (context, last_date)

        self._load_pipeline()
        self._model = self._pipeline
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        # Batch all contexts into a list of tensors for efficient inference
        ts_ids = list(self._series_contexts.keys())
        context_tensors = [
            torch.tensor(self._series_contexts[ts_id][0], dtype=torch.float32)
            for ts_id in ts_ids
        ]

        # ChronosPipeline.predict returns shape (num_series, num_samples, horizon)
        with torch.no_grad():
            forecast_samples = self._pipeline.predict(
                inputs=context_tensors,
                prediction_length=horizon,
            )

        # Bolt returns (num_series, num_quantiles, horizon) — take median (index 4 = 0.5 quantile)
        # Original returns (num_series, num_samples, horizon) — take median across samples
        if self._is_bolt:
            # Quantile index 4 = 0.5 (median) based on [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            point_forecasts = forecast_samples[:, 4, :].cpu().numpy()
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
