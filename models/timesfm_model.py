import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from models.model_interface import ForecastModel

# Enforce offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class TimesFMModel(ForecastModel):
    """Google TimesFM zero-shot forecasting model. Fully local inference.

    Pre-download the model once (with internet):
        python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/timesfm-1.0-200m'))"
    """

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="timesfm", params=params)
        self._freq: str | None = None
        self._series_contexts: Dict[str, tuple] = {}
        self._model_path: str = self.params.get("model_path", "google/timesfm-1.0-200m-pytorch")

    def _load_model(self):
        """Load TimesFM from local HuggingFace cache using the updated API.

        TimesFM 2.0 (``google/timesfm-2.0-500m-pytorch``) requires a different
        architecture (50 layers, 1280-dim, no positional embeddings, longer
        output patch). We detect the version from ``model_path`` and pass the
        correct hparams; defaults below match the 1.0-200m checkpoint.
        """
        import timesfm

        is_v2 = "2.0" in self._model_path
        per_core_batch_size = min(
            32, len(self._series_contexts) if self._series_contexts else 32
        )

        if is_v2:
            hparams = timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=per_core_batch_size,
                horizon_len=128,
                num_layers=50,
                model_dims=1280,
                input_patch_len=32,
                output_patch_len=128,
                use_positional_embedding=False,
            )
        else:
            hparams = timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=per_core_batch_size,
                horizon_len=128,
            )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self._model_path,
        )
        self._tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "TimesFMModel":
        """TimesFM is zero-shot; fit stores context windows per series."""
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        context_length = self.params.get("context_length", 128)

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            values = group_dataframe[target_column].values.astype(float)
            # Clamp context_length to available history.
            effective_length = min(context_length, len(values))
            context = values[-effective_length:]
            last_date = pd.to_datetime(group_dataframe[time_column].iloc[-1])
            self._series_contexts[ts_id] = (context, last_date)

        self._load_model()
        self._model = self._tfm
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        # Batch all contexts for efficient inference
        ts_ids = list(self._series_contexts.keys())
        contexts = [self._series_contexts[tid][0].tolist() for tid in ts_ids]

        # freq: 0 = monthly in TimesFM convention
        frequency_input = [0] * len(contexts)
        point_forecasts, _ = self._tfm.forecast(contexts, freq=frequency_input)

        for index, ts_id in enumerate(ts_ids):
            _, last_date = self._series_contexts[ts_id]
            predictions = point_forecasts[index, :horizon]
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]

            forecast = pd.DataFrame({
                time_column: future_dates,
                "forecast": predictions,
                "ts_id": ts_id,
            })
            all_forecasts.append(forecast)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"series_contexts": self._series_contexts, "freq": self._freq}, path)

    def load(self, path: Path) -> "TimesFMModel":
        path = Path(path)
        data = joblib.load(path)
        self._series_contexts = data["series_contexts"]
        self._freq = data["freq"]
        # Re-instantiate the model — weights come from the local HF cache.
        self._load_model()
        self._model = self._tfm
        return self
