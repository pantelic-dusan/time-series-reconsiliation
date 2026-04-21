from pathlib import Path
from typing import Any, Dict

import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.predictor import PyTorchPredictor

from models.model_interface import ForecastModel


class DeepARModel(ForecastModel):
    """GluonTS DeepAR wrapper. Trains globally on all series, produces probabilistic forecasts."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="deepar", params=params)
        self._freq: str | None = None
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
            freq=self._freq,
        )

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "DeepARModel":
        self._freq = config["data"]["frequency"]
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._predictor.serialize(path)

    def load(self, path: Path) -> "DeepARModel":
        path = Path(path)
        self._predictor = PyTorchPredictor.deserialize(path)
        self._model = self._predictor
        return self

