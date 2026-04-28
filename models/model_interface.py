from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


class ForecastModel(ABC):
    """Base interface for time-series forecasting models."""

    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.params = params or {}
        self._model = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> "ForecastModel":
        """Train the model on a single time-series DataFrame."""
        ...

    @abstractmethod
    def predict(self, horizon: int, config: Dict[str, Any]) -> pd.DataFrame:
        """ Generate forecasts for the given horizon. """
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted model to disk. """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: Path) -> "ForecastModel":
        """Load a previously saved model from disk."""
        path = Path(path)
        self._model = joblib.load(path)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

