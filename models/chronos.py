import os
from pathlib import Path
from typing import Any, Dict


import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models.model_interface import ForecastModel

# Enforce offline mode so no data leaves the machine
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ChronosModel(ForecastModel):
    """Amazon Chronos-T5 zero-shot forecasting model. Fully local inference."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="chronos", params=params)
        self._freq: str | None = None
        self._series_contexts: Dict[str, tuple] = {}
        self._device: str = self.params.get("device", "cpu")
        self._model_path: str = self.params.get("model_path", "amazon/chronos-t5-small")

    def _load_pipeline(self):
        """Load model and tokenizer from local cache."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, local_files_only=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path, local_files_only=True)
        self._model.to(self._device)
        self._model.eval()
    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "ChronosModel":
        """Chronos is zero-shot; fit stores context windows per series."""
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
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        for ts_id, (context, last_date) in self._series_contexts.items():
            with torch.no_grad():
                input_ids = self._tokenizer(
                    context.tolist(), return_tensors="pt", padding=True
                ).input_ids.to(self._device)
                outputs = self._model.generate(input_ids, max_new_tokens=horizon)
                predictions = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

            forecast_values = [float(v) for v in predictions[0].split()[:horizon]]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]
            forecast = pd.DataFrame({
                time_column: future_dates,
                "forecast": forecast_values[:horizon],
                "ts_id": ts_id,
            })
            all_forecasts.append(forecast)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"series_contexts": self._series_contexts, "freq": self._freq}, path)

    def load(self, path: Path) -> "ChronosModel":
        import joblib
        path = Path(path)
        data = joblib.load(path)
        self._series_contexts = data["series_contexts"]
        self._freq = data["freq"]
        self._load_pipeline()
        return self
