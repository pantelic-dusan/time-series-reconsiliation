"""Model registry — maps config keys to concrete ForecastModel subclasses."""

# Statistical models
from models.arima import ARIMAModel
from models.holt_winters import HoltWintersModel
from models.prophet_model import ProphetModel

# Classic ML models
from models.random_forest import RandomForestModel
from models.lightgbm_model import LightGBMModel

# Deep learning models
from models.deepar import DeepARModel
from models.nhits import NHITSModel

# Foundation models (pretrained, zero-shot)
from models.chronos import ChronosModel
from models.timesfm_model import TimesFMModel

MODEL_REGISTRY = {
    # Statistical
    "arima": ARIMAModel,
    "holt_winters": HoltWintersModel,
    "prophet": ProphetModel,
    # Classic ML
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    # Deep learning
    "deepar": DeepARModel,
    "nhits": NHITSModel,
    # Foundation
    "chronos": ChronosModel,
    "timesfm": TimesFMModel,
}
