# Time Series Reconciliation

Hierarchical time-series forecasting benchmark with multiple model families.
Each model runs at each hierarchy level, forecasts are stored as CSV files,
and evaluation is a separate step.

## 1. Quick Start

### 1.1 Requirements

- Python 3.10 or 3.11
- Poetry (recommended) or pip + venv
- Optional GPU for DeepAR, N-HiTS, Chronos, TimesFM

### 1.2 Install

Option A (Poetry):

```powershell
poetry install
poetry shell
```

Option B (pip + venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn lightgbm statsmodels pmdarima prophet `
            neuralforecast gluonts torch lightning transformers `
            chronos-forecasting joblib pyyaml timesfm orjson
```

### 1.3 One-time foundation model download (if used)

```powershell
huggingface-cli download amazon/chronos-bolt-small
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/timesfm-1.0-200m-pytorch')"
```

### 1.4 Run workflow

1. Optional HPO:

```powershell
python tune.py
```

2. Train and forecast:

```powershell
python train.py
```

3. Evaluate:

```powershell
python evaluate.py
```

All scripts use top-of-file constants (for example `CONFIG_PATH`, `LOG_FILE`,
`RESUME`) instead of CLI flags.

## 2. Minimal Configuration

Main settings live in [`config.yaml`](config.yaml):

```yaml
data:
  data_path: "data/data.csv"
  target_col: "quantity"
  time_col: "date"
  id_cols: ["material", "customer", "location"]
  frequency: "MS"

experiment:
  horizon: 6
  train_end_date: "2024-12-01"
  seed: 42

hierarchy:
  structural_levels: ["material", "customer", "location", "total"]
```

Input CSV must include at least date, id columns, and target. A `ts_id` is built
from `id_cols`.

## 3. What You Get

- Forecasts per level/model: `outputs/<level>/<model>_forecasts.csv`
- Evaluation detail per level/model: `outputs/<level>/<model>_evaluation_detail.csv`
- Evaluation summary per level: `outputs/<level>/evaluation_summary.csv`
- Cross-level KPI table: `outputs/kpi_by_level.csv`
- Checkpoints: `outputs/checkpoints/<level>/...`
- Logs: `logs/train_<timestamp>.log`, `logs/evaluate_<timestamp>.log`, `logs/tune_<timestamp>.log`

## 4. Model Set

| Name | Family | Notes |
| --- | --- | --- |
| `arima` | Statistical | `pmdarima.auto_arima`, per-series parallelism. |
| `holt_winters` | Statistical | Seasonal to trend to simple to naive fallback chain. |
| `prophet` | Statistical | Per-series Prophet with joblib parallelism. |
| `random_forest` | Classic ML | Global recursive strategy, lag + calendar features. |
| `random_forest_direct` | Classic ML | Same model class, direct strategy, inherits HPO from `random_forest`. |
| `lightgbm` | Classic ML | Global recursive strategy, same feature design as RF. |
| `lightgbm_direct` | Classic ML | Same model class, direct strategy, inherits HPO from `lightgbm`. |
| `deepar` | Deep learning | GluonTS probabilistic RNN. |
| `nhits` | Deep learning | NeuralForecast N-HiTS. |
| `chronos` | Foundation | Zero-shot Chronos or Chronos-Bolt. |
| `timesfm` | Foundation | Zero-shot TimesFM. |

## 5. HPO Notes

- `tune.py` writes only `hpo_results` to `config_tuned.yaml`.
- `train.py` loads and merges those overrides automatically.
- HPO resumes automatically: completed `(level, model)` pairs are skipped.
- Sampler is automatic: Grid when fully enumerable and small, otherwise TPE.

For RF and LightGBM:

- `strategy` is intentionally not in search spaces.
- Recursive entries (`random_forest`, `lightgbm`) are tuned.
- Direct entries (`random_forest_direct`, `lightgbm_direct`) set `hpo.enabled: false` and use `hpo_inherit_from` to reuse recursive tuned params.
- `type` selects the registry model class, while `name` remains the unique key for filenames, logs, and HPO mappings.

## 6. Architecture

| File / Dir | Purpose |
| --- | --- |
| `train.py` | Fit all configured models by level, save forecasts and checkpoints. |
| `evaluate.py` | Score saved forecasts vs test split. |
| `tune.py` | Optuna HPO across enabled models and levels. |
| `utils/aggregation_utils.py` | Hierarchy aggregation and `iter_levels()`. |
| `utils/utils.py` | Config and data I/O, tuned config read/write. |
| `utils/evaluation_utils.py` | Metric functions and scoring logic. |
| `utils/logging_utils.py` | Logger setup and execution timing helpers. |
| `models/` | Model implementations, all via `ForecastModel` interface. |

## 7. Metrics

MAE, RMSE, MAPE, sMAPE, BIAS, Tracking Signal, R2, WAPE,
`num_series`, and `num_predictions`.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Foundation model files not found | Pre-download checkpoints (see Quick Start). |
| Coverage check failure in training | Model skipped some `ts_id`; partial artifacts are auto-removed. |
| `ModuleNotFoundError: pmdarima` | Ensure environment install succeeded with compatible numpy. |
| Prophet build fails on Windows | Install Visual C++ Build Tools or use WSL. |
| GPU not used | Install CUDA-enabled torch build; models use `accelerator="auto"`. |

## 9. Adding a New Model

1. Add `models/<your_model>.py` implementing `ForecastModel`.
2. Register it in `models/__init__.py`.
3. Add a model block in `config.yaml`.
4. Optionally add an HPO block with `enabled: true` and `search_space`.

Contract: predictions must include all input `ts_id` values and exactly
`horizon` rows per series.

