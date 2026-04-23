from pathlib import Path
from typing import Any, Dict

import json

import pandas as pd

import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.predictor import PyTorchPredictor

from models.model_interface import ForecastModel

# PyTorch 2.6+ defaults to weights_only=True which blocks GluonTS internal
# classes during checkpoint loading. Lightning passes weights_only=True explicitly,
# so we force it to False since all checkpoints are local and trusted.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class DeepARModel(ForecastModel):
    """GluonTS DeepAR wrapper. Trains globally on all series, produces probabilistic forecasts."""

    # GluonTS has a split-brain issue with frequencies:
    #   - get_lags_for_frequency() / DeepAREstimator accept "MS" (offset alias)
    #   - PandasDataset.to_period() needs "M" (period alias)
    # We use PERIOD_FREQ_MAP for the dataset and the original freq for the estimator.
    PERIOD_FREQ_MAP = {"MS": "M", "QS": "Q", "YS": "Y", "AS": "A"}

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="deepar", params=params)
        self._freq: str | None = None           # Original pandas freq (e.g. "MS") — for estimator + date_range
        self._period_freq: str | None = None     # Period-compatible freq (e.g. "M") — for PandasDataset
        self._predictor: PyTorchPredictor | None = None
        self._train_dataset: PandasDataset | None = None

    def _build_dataset(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> PandasDataset:
        """Convert dataframe to GluonTS PandasDataset using pre-computed ts_id."""
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]

        dataframe = dataframe.copy()
        dataframe[time_column] = pd.to_datetime(dataframe[time_column])
        dataframe = dataframe.set_index(time_column)

        return PandasDataset.from_long_dataframe(
            dataframe,
            item_id="ts_id",
            target=target_column,
            freq=self._period_freq,
        )

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "DeepARModel":
        self._freq = config["data"]["frequency"]
        self._period_freq = self.PERIOD_FREQ_MAP.get(self._freq, self._freq)
        horizon = config["experiment"]["horizon"]

        self._train_dataset = self._build_dataset(dataframe, config)
        self._validation_dataset = self._train_dataset

        context_length = self.params.get("context_length", 12)

        estimator = DeepAREstimator(
            prediction_length=horizon,
            context_length=context_length,
            freq=self._freq,
            hidden_size=self.params.get("hidden_size", 32),
            num_layers=self.params.get("rnn_layers", 2),
            batch_size=self.params.get("batch_size", 32),
            trainer_kwargs={
                "max_epochs": self.params.get("max_epochs", 50),
                "accelerator": "auto",
                "enable_progress_bar": True,
            },
        )

        self._predictor = estimator.train(
            training_data=self._train_dataset,
            validation_data=self._validation_dataset,
        )
        self._model = self._predictor
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        forecasts = list(self._predictor.predict(self._train_dataset))

        all_forecasts = []
        for forecast in forecasts:
            # Use mean of probabilistic samples as point forecast
            mean_forecast = forecast.mean
            future_dates = pd.date_range(
                start=forecast.start_date.to_timestamp(),
                periods=horizon,
                freq=self._freq,
            )
            forecast_dataframe = pd.DataFrame({
                time_column: future_dates,
                "forecast": mean_forecast[:horizon],
                "ts_id": forecast.item_id,
            })
            all_forecasts.append(forecast_dataframe)

        return pd.concat(all_forecasts, ignore_index=True)

    def save(self, path: Path) -> None:
        # GluonTS serialize() expects a directory, not a file.
        # Strip .pkl extension and use path as a directory.
        path = Path(path).with_suffix("")
        path.mkdir(parents=True, exist_ok=True)
        self._predictor.serialize(path)
        # Persist freq info so predict() works after load().
        (path / "_freq.json").write_text(
            json.dumps({"freq": self._freq, "period_freq": self._period_freq})
        )

    def load(self, path: Path) -> "DeepARModel":
        path = Path(path).with_suffix("")
        self._predictor = PyTorchPredictor.deserialize(path)
        meta_path = path / "_freq.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._freq = meta.get("freq")
            self._period_freq = meta.get("period_freq")
        self._model = self._predictor
        return self

