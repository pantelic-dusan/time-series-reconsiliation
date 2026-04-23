from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from models.model_interface import ForecastModel


"""LightGBM regressor trained globally across all series."""
class LightGBMModel(ForecastModel):

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(model_name="lightgbm", params=params)
        self._n_lags: int = self.params.get("n_lags", 12)
        self._strategy: str = self.params.get("strategy", "recursive")
        self._freq: str | None = None
        self._series_state: Dict[str, Tuple[np.ndarray, pd.Timestamp]] = {}
        self._models: Dict[int, LGBMRegressor] = {}

    @staticmethod
    def _create_lag_features(series: np.ndarray, n_lags: int) -> pd.DataFrame:
        """Create lag feature matrix with named columns."""
        X = np.lib.stride_tricks.sliding_window_view(series, n_lags)
        return pd.DataFrame(X, columns=[f"lag_{i+1}" for i in range(n_lags)])

    def _fit_recursive(self, X_train: np.ndarray, all_y: List[np.ndarray], seed: int) -> None:
        """Train a single one-step-ahead model."""
        y_train = np.concatenate(all_y, axis=0)[:len(X_train)]
        params = {k: v for k, v in self.params.items() if k not in ("n_lags", "strategy")}
        params.setdefault("verbosity", -1)
        model = LGBMRegressor(**params, random_state=seed)
        model.fit(X_train, y_train)
        self._models[0] = model

    def _fit_direct(self, X_train: np.ndarray, all_y_per_step: Dict[int, List[np.ndarray]], horizon: int, seed: int) -> None:
        """Train one model per horizon step."""
        params = {k: v for k, v in self.params.items() if k not in ("n_lags", "strategy")}
        params.setdefault("verbosity", -1)
        for h in range(horizon):
            y_train = np.concatenate(all_y_per_step[h], axis=0)[:len(X_train)]
            model = LGBMRegressor(**params, random_state=seed)
            model.fit(X_train, y_train)
            self._models[h] = model

    def _predict_recursive(self, last_values: np.ndarray, horizon: int) -> List[float]:
        """Predict step-by-step, feeding each prediction back as input."""
        predictions = []
        current = last_values.copy()
        columns = [f"lag_{i+1}" for i in range(self._n_lags)]
        for _ in range(horizon):
            x = pd.DataFrame([current], columns=columns)
            yhat = self._models[0].predict(x)[0]
            predictions.append(yhat)
            current = np.append(current[1:], yhat)
        return predictions

    def _predict_direct(self, last_values: np.ndarray, horizon: int) -> List[float]:
        """Each model independently predicts its horizon step from original lags."""
        columns = [f"lag_{i+1}" for i in range(self._n_lags)]
        x = pd.DataFrame([last_values], columns=columns)
        return [self._models[h].predict(x)[0] for h in range(horizon)]

    def fit(self, dataframe: pd.DataFrame, config: Dict[str, Any]) -> "LightGBMModel":
        target_column = config["data"]["target_col"]
        time_column = config["data"]["time_col"]
        self._freq = config["data"]["frequency"]
        horizon = config["experiment"]["horizon"]
        seed = config["experiment"].get("seed", 42)

        all_X: List[np.ndarray] = []
        all_y_per_step: Dict[int, List[np.ndarray]] = {h: [] for h in range(horizon)}

        for ts_id, group_dataframe in dataframe.groupby("ts_id"):
            group_dataframe = group_dataframe.sort_values(time_column).reset_index(drop=True)
            values = group_dataframe[target_column].values.astype(float)
            last_date = pd.to_datetime(group_dataframe[time_column].iloc[-1])

            X = self._create_lag_features(values, self._n_lags)

            if self._strategy == "direct":
                # Each row i of X predicts targets at positions n_lags+h for h in [0, horizon).
                # Truncate X and every y[h] to the SAME per-series length so rows stay
                # aligned after cross-series concatenation.
                min_length = len(values) - self._n_lags - (horizon - 1)
                all_X.append(X.iloc[:min_length])
                for h in range(horizon):
                    start = self._n_lags + h
                    all_y_per_step[h].append(values[start : start + min_length])
            else:
                # One-step-ahead: row i of X[:-1] predicts values[i + n_lags].
                X_aligned = X.iloc[:-1]
                y_aligned = values[self._n_lags : self._n_lags + len(X_aligned)]
                all_X.append(X_aligned)
                all_y_per_step[0].append(y_aligned)

            self._series_state[ts_id] = (values[-self._n_lags:], last_date)

        X_train = np.concatenate(all_X, axis=0)
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
                predictions = self._predict_direct(last_values, horizon)
            else:
                predictions = self._predict_recursive(last_values, horizon)

            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=self._freq)[1:]
            all_forecasts.append(pd.DataFrame({
                time_column: future_dates,
                "forecast": predictions,
                "ts_id": ts_id,
            }))

        return pd.concat(all_forecasts, ignore_index=True)
