import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

from models.model_interface import ForecastModel

# Enforce offline mode so no data leaves the machine after initial download
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _detect_kind(model_path: str) -> str:
    """Classify which Chronos pipeline class to use based on model path."""
    path_lower = model_path.lower()
    if "chronos-2" in path_lower or path_lower.endswith("/chronos-2"):
        return "chronos2"
    if "bolt" in path_lower:
        return "bolt"
    return "legacy"


class ChronosModel(ForecastModel):
    """Amazon Chronos zero-shot forecasting model.

    Supports three pipeline families, dispatched by `model_path`:
      - Chronos-2  (e.g. ``amazon/chronos-2``) — newest, uses ``Chronos2Pipeline``
      - Chronos-Bolt (``amazon/chronos-bolt-*``) — uses ``ChronosBoltPipeline``
      - Legacy Chronos (``amazon/chronos-t5-*``) — uses ``ChronosPipeline``

    Pre-download once (with internet), e.g.:
        huggingface-cli download amazon/chronos-2
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="chronos", params=params)
        self._freq: str | None = None
        self._series_contexts: Dict[str, tuple] = {}
        self._device: str = self.params.get("device", "cpu")
        self._model_path: str = self.params.get("model_path", "amazon/chronos-2")
        self._kind: str = _detect_kind(self._model_path)
        self._pipeline = None

    def _load_pipeline(self):
        """Load the Chronos pipeline from local HuggingFace cache."""
        if self._kind == "chronos2":
            from chronos import Chronos2Pipeline

            # Note: Chronos2Pipeline.from_pretrained does not accept
            # `local_files_only`; offline behavior is enforced via the
            # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE env vars set above.
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self._model_path,
                device_map=self._device,
                dtype=torch.float32,
            )
        elif self._kind == "bolt":
            from chronos import ChronosBoltPipeline

            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self._model_path,
                device_map=self._device,
                dtype=torch.float32,
                local_files_only=True,
            )
        else:
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self._model_path,
                device_map=self._device,
                dtype=torch.float32,
                local_files_only=True,
            )

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ChronosModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        context_length = self.params.get("context_length", 128)

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
            if self._kind == "chronos2":
                quantile_levels = [0.1, 0.5, 0.9]
                # Returns (quantile_forecasts, mean_forecasts):
                #   quantile_forecasts: list[Tensor(1, H, Q)] — one per series
                #   mean_forecasts:     list[Tensor(1, H)]
                quantile_forecasts, _ = self._pipeline.predict_quantiles(
                    inputs=context_tensors,
                    prediction_length=horizon,
                    quantile_levels=quantile_levels,
                )
                median_index = quantile_levels.index(0.5)
                # Stack to (B, H) by selecting the median quantile and squeezing.
                point_forecasts = np.stack(
                    [
                        q.squeeze(0)[:, median_index].cpu().numpy()
                        for q in quantile_forecasts
                    ],
                    axis=0,
                )
            elif self._kind == "bolt":
                forecast_samples = self._pipeline.predict(
                    inputs=context_tensors,
                    prediction_length=horizon,
                )
                quantiles = getattr(self._pipeline, "quantiles", None)
                if quantiles is not None and 0.5 in list(quantiles):
                    median_index = list(quantiles).index(0.5)
                else:
                    median_index = forecast_samples.shape[1] // 2
                point_forecasts = forecast_samples[:, median_index, :].cpu().numpy()
            else:
                forecast_samples = self._pipeline.predict(
                    inputs=context_tensors,
                    prediction_length=horizon,
                )
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
            "model_path": self._model_path,
        }, path)

    def load(self, path: Path) -> "ChronosModel":
        path = Path(path)
        data = joblib.load(path)
        self._series_contexts = data["series_contexts"]
        self._freq = data["freq"]
        # Restore model_path if saved (newer checkpoints) so kind stays consistent.
        if "model_path" in data:
            self._model_path = data["model_path"]
            self._kind = _detect_kind(self._model_path)
        self._load_pipeline()
        self._model = self._pipeline
        return self
