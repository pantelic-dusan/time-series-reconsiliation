from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from models import MODEL_REGISTRY


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML experiment configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config: Dict[str, Any]) -> None:
    """Run the full forecasting experiment as defined in config."""
    data_config = config["data"]
    experiment_config = config["experiment"]

    # Load data
    dataframe = pd.read_csv(data_config["data_path"], parse_dates=[data_config["time_col"]])

    # Create a unified time-series ID column
    id_columns = data_config["id_cols"]
    dataframe["ts_id"] = dataframe[id_columns].astype(str).agg("_".join, axis=1)

    # Prepare output directories
    output_dir = Path(config["storage"]["output_dir"])
    checkpoint_dir = Path(config["storage"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    horizon = experiment_config["horizon"]
    train_end_date = pd.Timestamp(experiment_config["train_end_date"])

    # Train/test split by date
    train_dataframe = dataframe[dataframe[data_config["time_col"]] <= train_end_date]
    test_dataframe = dataframe[dataframe[data_config["time_col"]] > train_end_date]

    for model_config in config["models"]:
        model_name = model_config["name"]
        model_params = model_config.get("params", {})

        if model_name not in MODEL_REGISTRY:
            print(f"[WARN] Unknown model '{model_name}', skipping.")
            continue

        ModelClass = MODEL_REGISTRY[model_name]
        print(f"\n{'='*60}")
        print(f"Running model: {model_name}")
        print(f"{'='*60}")

        # Instantiate, fit, predict
        model = ModelClass(params=model_params)
        try:
            model.fit(train_dataframe, config)
            forecast_dataframe = model.predict(horizon, config)

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"{model_name}.pkl"
            model.save(checkpoint_path)

            # Save forecasts
            results_path = output_dir / f"{model_name}_forecasts.csv"
            forecast_dataframe.to_csv(results_path, index=False)
            print(f"  Saved forecasts → {results_path}")

        except Exception as e:
            print(f"  [ERROR] {model_name} failed: {e}")
            continue


if __name__ == "__main__":
    config = load_config("config.yaml")
    run_experiment(config)
