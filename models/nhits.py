from pathlib import Path
from typing import Any, Dict

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from utils.logging_utils import make_dl_training_logger
from models.model_interface import ForecastModel


class NHITSModel(ForecastModel):
    """Nixtla NeuralForecast N-HiTS wrapper. Trains globally on all series."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="nhits", params=params)
        self._freq: str | None = None
        self._nf: NeuralForecast | None = None

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "NHITSModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        horizon = config["experiment"]["horizon"]

        # NeuralForecast expects columns: unique_id, ds, y
        nf_dataframe = dataframe[["ts_id", time_column, target_column]].copy()
        nf_dataframe.columns = ["unique_id", "ds", "y"]
        nf_dataframe["ds"] = pd.to_datetime(nf_dataframe["ds"])

        input_size = self.params.get("input_size", 12)
        max_steps = self.params.get("max_steps", 1000)

        model = NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            scaler_type="standard",
            val_check_steps=50,
            early_stop_patience_steps=5,
            logger=make_dl_training_logger("nhits"),
        )
        self._nf = NeuralForecast(models=[model], freq=self._freq)
        self._nf.fit(df=nf_dataframe, val_size=horizon)
        self._model = self._nf
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        forecast_dataframe = self._nf.predict()
        result = forecast_dataframe.reset_index()[["unique_id", "ds", "NHITS"]]
        result.columns = ["ts_id", time_column, "forecast"]
        return result

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._nf.save(path=str(path), model_index=None, overwrite=True)

    def load(self, path: Path) -> "NHITSModel":
        path = Path(path)
        self._nf = NeuralForecast.load(path=str(path))
        self._model = self._nf
        return self

    def in_sample_fitted(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """One-step in-sample fitted values via ``NeuralForecast.predict_insample``.

        ``predict_insample(step_size=1)`` produces rolling-window predictions for
        the full horizon at every cutoff. We keep only the one-step-ahead row
        (smallest ``ds - cutoff``) per ``(unique_id, ds)``.
        """
        time_column = config["data"]["time_col"]
        if self._nf is None:
            raise RuntimeError("NHITS.in_sample_fitted called before fit/load")

        in_sample = self._nf.predict_insample(step_size=1).reset_index(drop=False)
        in_sample["ds"] = pd.to_datetime(in_sample["ds"])
        in_sample["cutoff"] = pd.to_datetime(in_sample["cutoff"])
        in_sample["__step"] = (in_sample["ds"] - in_sample["cutoff"]).dt.days
        # Keep the most recent forecast (smallest step) per (unique_id, ds).
        idx = in_sample.groupby(["unique_id", "ds"], sort=False)["__step"].idxmin()
        one_step = in_sample.loc[idx, ["unique_id", "ds", "NHITS"]]

        return pd.DataFrame({
            "ts_id": one_step["unique_id"].values,
            "date": one_step["ds"].values,
            "fitted": one_step["NHITS"].astype(float).values,
        })
