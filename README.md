# Time Series Reconciliation

Multi-model, multi-level hierarchical time-series forecasting benchmark.
Every configured model is fit at every hierarchy level (base + structural +
temporal aggregations), forecasts are saved as CSVs, and a separate
evaluation step scores them with standard KPIs.

---

## 1. What's in the box

| File / dir             | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `train.py`             | Entry point: fit every model at every level, save forecasts + checkpoints. |
| `evaluate.py`          | Entry point: score saved forecasts against the test split.            |
| `main.py`              | Deprecation stub — prints a pointer to the two scripts above.         |
| `config.yaml`          | Single source of truth for data, horizon, hierarchy, and model params.|
| `aggregation.py`       | Structural & temporal aggregation + per-level config derivation.      |
| `evaluation/evaluation.py` | Pure metric functions + `evaluate_model(...)` scorer.             |
| `logging_utils.py`     | Shared logger setup + `timed()` context manager.                      |
| `models/`              | One file per model; all implement `ForecastModel` from `model_interface.py`. |
| `data/data.csv`        | Input dataset (edit `config.yaml → data.data_path` to change).        |
| `outputs/`             | Forecast CSVs, checkpoints, and evaluation summaries (auto-created).  |
| `logs/`                | Per-run timestamped log files (auto-created).                         |

### Supported models (configurable in `config.yaml → models`)

| Name            | Family        | Notes                                                    |
| --------------- | ------------- | -------------------------------------------------------- |
| `arima`         | Statistical   | `pmdarima.auto_arima`, parallel per series, naive fallback on fit failure. |
| `holt_winters`  | Statistical   | `statsmodels` ExponentialSmoothing, seasonal → trend → simple → naive fallback chain. |
| `prophet`       | Statistical   | Meta Prophet, one model per series.                      |
| `random_forest` | Classic ML    | Global model, recursive or direct multi-step strategy.   |
| `lightgbm`      | Classic ML    | Same design as `random_forest`.                          |
| `deepar`        | Deep learning | GluonTS probabilistic RNN.                               |
| `nhits`         | Deep learning | Nixtla NeuralForecast N-HiTS.                            |
| `chronos`       | Foundation    | Amazon Chronos / Chronos-Bolt, zero-shot.                |
| `timesfm`       | Foundation    | Google TimesFM, zero-shot.                               |

---

## 2. Requirements

- **Python 3.10 or 3.11** (see `pyproject.toml`; PyTorch / TimesFM constraints prevent 3.12).
- **Poetry** (recommended) or plain `pip`.
- ~4 GB RAM minimum; foundation models (`chronos`, `timesfm`) benefit from more.
- Optional GPU for `deepar` / `nhits` / foundation models — CPU works, just slower.

### Install (Poetry)

```powershell
poetry install
poetry shell
```

### Install (pip)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn lightgbm statsmodels pmdarima prophet `
            neuralforecast gluonts torch lightning transformers `
            chronos-forecasting joblib pyyaml timesfm orjson
```

### Pre-download foundation models (one-time, online)

Chronos and TimesFM are run in fully offline mode (`HF_HUB_OFFLINE=1`).
Download them once while you have network access:

```powershell
huggingface-cli download amazon/chronos-bolt-small
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/timesfm-1.0-200m-pytorch')"
```

---

## 3. Input data format

CSV file pointed to by `config.yaml → data.data_path`, containing at minimum:

| Column                 | Meaning                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `date`                 | Time stamp (daily/monthly; must match `data.frequency`).          |
| `material`, `customer`, `location` | Grouping keys (listed in `data.id_cols`). Any subset works. |
| `quantity`             | Numeric target (name from `data.target_col`).                      |

A unique `ts_id` is built at load time by joining `id_cols` with `_`.
Each series is assumed to have at least ~4 years of history at the base
frequency (required by the lag-based ML models without naive fallback).

---

## 4. Configuration

All behavior is driven by [`config.yaml`](config.yaml):

```yaml
data:
  data_path: "data/data.csv"
  target_col: "quantity"
  time_col: "date"
  id_cols: ["material", "customer", "location"]
  frequency: "MS"              # pandas offset alias (MS = month-start)

experiment:
  horizon: 6                   # steps to forecast at base frequency
  train_end_date: "2024-12-01" # rows with date > this go to the test split
  seed: 42

hierarchy:
  structural_levels: ["material", "customer", "location", "total"]
  temporal_levels:   ["quarterly"]   # currently "quarterly" or "half_yearly"

models:
  - name: "arima"
    params: { seasonal_period: 12, n_jobs: -1 }
  - name: "holt_winters"
    params: { seasonal_periods: 12, trend: "add", seasonal: "add" }
  # ... see config.yaml for the full list

storage:
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
```

`aggregation.get_level_config(...)` automatically adjusts `horizon`,
`frequency`, seasonal periods, lags, and context lengths for each
temporal level, so model params only need to be written once.

---

## 5. Running

Both scripts use **UPPERCASE top-of-file constants** instead of CLI flags —
edit the script, then run it.

### 5.1 Train + predict

Edit the constants at the top of [`train.py`](train.py):

```python
CONFIG_PATH: str = "config.yaml"
LOG_FILE:    str = "logs/train.log"
RESUME:      bool = False    # True = skip models whose forecast CSV already exists
```

Then:

```powershell
python train.py
```

What it does, for every `(level, model)` combination:

1. Aggregates the training data to the level's grain.
2. Fits the model, saves a checkpoint under `outputs/checkpoints/<level>/`.
3. Generates `horizon`-step forecasts, clips to zero.
4. **Hard-fail coverage check:** every input `ts_id` must appear in the
   output with exactly `horizon` rows. On failure, the partial checkpoint
   (pickle + `.json` sibling + directory sibling) and forecast CSV are
   deleted and training continues with the next model.
5. Writes `outputs/<level>/<model>_forecasts.csv`.

### 5.2 Evaluate

Edit the constants at the top of [`evaluate.py`](evaluate.py):

```python
CONFIG_PATH: str           = "config.yaml"
LOG_FILE:    str           = "logs/evaluate.log"
OUTPUT_DIR:  Optional[str] = None              # None → from config
ONLY_LEVEL:  Optional[str] = None              # e.g. "base", "structural__material"
```

Then:

```powershell
python evaluate.py
```

For each level it writes:

- `outputs/<level>/<model>_evaluation_detail.csv` — per-series metrics.
- `outputs/<level>/evaluation_summary.csv`       — one row per model.

No cross-level aggregation is performed (by design — keep levels isolated).

### 5.3 Metrics produced

MAE, RMSE, MAPE, sMAPE, BIAS, Tracking Signal, R², WAPE, plus
`num_series` and `num_predictions`.

---

## 6. Outputs layout

```
outputs/
├── base/
│   ├── arima_forecasts.csv
│   ├── arima_evaluation_detail.csv
│   ├── …
│   └── evaluation_summary.csv
├── structural__material/
│   └── …
├── structural__customer/
├── structural__location/
├── structural__total/
├── temporal__quarterly/
└── checkpoints/
    ├── base/
    │   ├── arima.pkl
    │   ├── prophet.json          # Prophet uses JSON
    │   ├── deepar/               # GluonTS uses a directory
    │   └── …
    └── <level>/…
```

---

## 7. Logs

Per-run, timestamped, accumulating files:

```
logs/
├── train_20260422_153012.log
├── evaluate_20260422_153047.log
└── …
```

The `timed(label)` context manager (from `logging_utils.py`) wraps every
level and every `(level, model)` pair and emits:

```
[START] level=base/arima
[DONE]  level=base/arima in 42.7s
```

Unhandled exceptions are logged with the full traceback via `exc_info=True`.

To turn off the timestamp suffix, call
`setup_logging(LOG_FILE, timestamped=False)`.

---

## 8. Troubleshooting

| Symptom                                              | Fix                                                                 |
| ---------------------------------------------------- | ------------------------------------------------------------------- |
| `OSError: Can't find model files locally …`          | Foundation model wasn't pre-downloaded. See §2.                     |
| Coverage-check failure logged for a model            | That model skipped series during fit. Partial artifacts were auto-cleaned; fix the model or raise its minimum-history requirement. |
| `ModuleNotFoundError: pmdarima`                      | `pmdarima` requires `numpy < 2.0`. Rerun `poetry install`.          |
| PyTorch complains about `weights_only=True`          | Already patched in `models/deepar.py`.                              |
| Prophet install fails on Windows                     | Install Visual C++ Build Tools, or use WSL.                         |
| GPU not used                                         | `deepar` / `nhits` use `accelerator="auto"`; install CUDA build of torch. |

---

## 9. Adding a new model

1. Create `models/<your_model>.py` subclassing `ForecastModel` with
   `fit`, `predict`, and optionally `save`/`load`.
2. Register it in `models/__init__.py → MODEL_REGISTRY`.
3. Add a `- name: "<your_model>"` block with params under `models:` in
   `config.yaml`.
4. If its params include any of `seasonal_period[s]`, `n_lags`,
   `context_length`, `input_size`, the per-temporal-level overrides in
   `aggregation.get_level_config` will adjust them automatically.

`fit(df, config)` must include **every** `ts_id` present in the input in
its predictions. `predict(horizon, config)` must return a DataFrame with
columns `[time_col, "forecast", "ts_id"]` and exactly `horizon` rows per
series — otherwise the coverage check in `train.py` will fail.

