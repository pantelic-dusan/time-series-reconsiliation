import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from utils.features_utils import all_feature_names, build_feature_matrix, prediction_row
from models.model_interface import ForecastModel

logger = logging.getLogger(__name__)


class LightGBMModel(ForecastModel):
    """LightGBM regressor trained globally across all series."""

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="lightgbm", params=params)
        self._n_lags: int = self.params.get("n_lags", 12)
        self._strategy: str = self.params.get("strategy", "recursive")
        self._freq: str | None = None
        self._series_state: Dict[str, Tuple[np.ndarray, pd.Timestamp]] = {}
        self._models: Dict[int, LGBMRegressor] = {}

    def _non_model_params(self) -> Dict[str, Any]:
        params = {
            k: v for k, v in self.params.items() if k not in ("n_lags", "strategy")
        }
        params.setdefault("verbosity", -1)
        return params

    def _fit_recursive(self, X_train: pd.DataFrame, all_y: List[np.ndarray], seed: int) -> None:
        """Train a single one-step-ahead model."""
        y_train = np.concatenate(all_y, axis=0)[:len(X_train)]
        model = LGBMRegressor(**self._non_model_params(), random_state=seed)
        model.fit(X_train, y_train)
        self._models[0] = model

    def _fit_direct(
        self,
        X_train: pd.DataFrame,
        all_y_per_step: Dict[int, List[np.ndarray]],
        horizon: int,
        seed: int,
    ) -> None:
        """Train one model per horizon step."""
        for h in range(horizon):
            y_train = np.concatenate(all_y_per_step[h], axis=0)[:len(X_train)]
            model = LGBMRegressor(**self._non_model_params(), random_state=seed)
            model.fit(X_train, y_train)
            self._models[h] = model

    def _predict_recursive(
        self, last_values: np.ndarray, last_date: pd.Timestamp, horizon: int
    ) -> List[float]:
        """Predict step-by-step, feeding each prediction back as a lag."""
        predictions = []
        current_lags = last_values.astype(float).copy()
        current_date = last_date
        offset = pd.tseries.frequencies.to_offset(self._freq)
        for _ in range(horizon):
            x = prediction_row(current_lags, current_date, self._n_lags)
            yhat = float(self._models[0].predict(x)[0])
            predictions.append(yhat)
            current_lags = np.append(current_lags[1:], yhat)
            current_date = current_date + offset
        return predictions

    def _predict_direct(
        self, last_values: np.ndarray, last_date: pd.Timestamp, horizon: int
    ) -> List[float]:
        """Each model independently predicts its horizon step from original lags."""
        x = prediction_row(last_values.astype(float), last_date, self._n_lags)
        return [float(self._models[h].predict(x)[0]) for h in range(horizon)]

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "LightGBMModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        horizon = config["experiment"]["horizon"]
        seed = config["experiment"].get("seed", 42)

        all_X: List[np.ndarray] = []
        all_y_per_step: Dict[int, List[np.ndarray]] = {h: [] for h in range(horizon)}
        self._series_state = {}

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            values = group_dataframe[target_column].values.astype(float)
            dates = pd.DatetimeIndex(pd.to_datetime(group_dataframe[time_column]))
            last_date = dates[-1]

            X = build_feature_matrix(values, dates, self._n_lags)

            if self._strategy == "direct":
                min_length = len(values) - self._n_lags - (horizon - 1)
                all_X.append(X[:min_length])
                for h in range(horizon):
                    start = self._n_lags + h
                    all_y_per_step[h].append(values[start : start + min_length])
            else:
                X_aligned = X[:-1]
                y_aligned = values[self._n_lags : self._n_lags + len(X_aligned)]
                all_X.append(X_aligned)
                all_y_per_step[0].append(y_aligned)

            self._series_state[ts_id] = (values[-self._n_lags:], last_date)

        X_train = pd.DataFrame(
            np.concatenate(all_X, axis=0),
            columns=all_feature_names(self._n_lags),
        )
        if self._strategy == "direct":
            self._fit_direct(X_train, all_y_per_step, horizon, seed)
        else:
            self._fit_recursive(X_train, all_y_per_step[0], seed)

        self._model = self._models
        return self

    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        time_column = config["data"]["time_col"]
        all_forecasts = []

        for ts_id, (last_values, last_date) in self._series_state.items():
            if self._strategy == "direct":
                predictions = self._predict_direct(last_values, last_date, horizon)
            else:
                predictions = self._predict_recursive(last_values, last_date, horizon)

            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]
            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": predictions,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)
