# AGENT.md — guidance for AI coding agents

This file is a briefing for automated coding agents (Copilot, Claude,
Cursor, etc.) working in this repository. Humans should read
[`README.md`](README.md) instead.

---

## 1. What this project is

A hierarchical time-series forecasting benchmark. Nine models × multiple
hierarchy levels × a standard KPI pack. Two entry points:

- **`train.py`** — fits models, saves forecasts as CSV + checkpoints.
- **`evaluate.py`** — scores saved forecasts against the held-out split.

Both use **top-of-file `UPPERCASE` constants** instead of CLI flags.
Don't add `argparse` — the user explicitly rejected it.

Control flow at the top level:

```
config.yaml
    │
    ▼
train.run_experiment
    │   for each (level, model):
    │       aggregate → fit → predict → clip≥0 → coverage-check → save
    ▼
outputs/<level>/<model>_forecasts.csv
    │
    ▼
evaluate.run_evaluation
    │   for each (level, model): merge(forecast, test) → metrics
    ▼
outputs/<level>/evaluation_summary.csv
```

---

## 2. House rules for edits

1. **Preserve the coverage contract.** Every model's `predict()` must
   return a DataFrame with one row per `(ts_id, future_timestamp)` pair,
   covering every `ts_id` present in `fit()`'s input, with exactly
   `horizon` rows per series. `train._check_coverage` hard-fails
   otherwise. If you introduce a model that could drop series, add a
   deterministic fallback inside the model — **don't** weaken the check.

2. **The user has confirmed ≥4 years of history per series.** Do not add
   "insufficient data" naive fallbacks to `random_forest` / `lightgbm` —
   the user explicitly rejected them. Naive fallbacks in `arima` /
   `holt_winters` exist only for numerical/convergence fit failures and
   should stay.

3. **No CLI args in `train.py` / `evaluate.py`.** Configure via the
   `UPPERCASE` module-level constants at the top of each file. Anything
   else needs to come from `config.yaml`.

4. **Logging goes through `logging_utils`.** Use `setup_logging(...)`
   once per entry point and `with timed("label"):` around meaningful
   units of work (per-level, per-model). Don't call `logging.basicConfig`.

5. **Log files are per-run and timestamped.** `setup_logging(log_file)`
   auto-appends `_YYYYmmdd_HHMMSS` unless `timestamped=False` is passed.
   Logs are accumulating — there is no rotation, no cleanup.

6. **On train-time failure, clean partial artifacts.**
   `train._cleanup_artifacts` deletes pickle, `.json` sibling, and
   directory sibling of the checkpoint, plus the forecast CSV. Any new
   model with a novel serialization format needs matching cleanup logic.

7. **`evaluation/` is a folder**, not a single file. Use
   `from evaluation.evaluation import evaluate_model`. Don't flatten it
   — the user chose this layout.

8. **Don't touch `evaluation.ALL_METRICS`** order without checking
   `evaluate_level`'s log-line formatting. `ME` was removed as a
   duplicate of `BIAS`; don't re-add it.

9. **`pmdarima` needs `numpy < 2`.** Don't bump numpy. If you must,
   verify `auto_arima` still imports.

10. **Foundation models are offline-only** (`HF_HUB_OFFLINE=1`,
    `TRANSFORMERS_OFFLINE=1` set at module import). Don't remove those
    environment flags.

---

## 3. Code conventions

- **Python 3.10 / 3.11**. Use `from __future__ import annotations` and
  `str | None` unions freely.
- **`snake_case` everywhere.** Variable names are long and descriptive
  (`forecast_dataframe`, not `fdf`).
- **Models subclass `ForecastModel`** from `models/model_interface.py`
  and implement `fit`, `predict`, and optionally override `save`/`load`.
- **`save()` must persist *everything*** needed to `load()` + `predict()`
  later — including `_freq`, last dates, naive fallback state. Past bugs
  (DeepAR, TimesFM) were caused by losing this state across load.
- **Pandas typing warnings** about `Hashable` vs `str` from `groupby`
  keys are ignored project-wide — they're noise.

---

## 4. Known design choices (don't "fix" these)

| Choice                                        | Why                                                   |
| --------------------------------------------- | ----------------------------------------------------- |
| Separate `train.py` / `evaluate.py`           | User explicitly asked for the split.                  |
| No CLI arguments                              | User prefers top-of-file macros.                      |
| Accumulating logs with per-run timestamped files | User asked for this specifically.                   |
| Hard-fail coverage check, cleans artifacts    | User asked for this specifically.                     |
| No cross-level reconciliation / coherence yet | Deferred by user — per-level metrics only for now.    |
| Checkpoint directory inside `outputs/`        | Matches `config.yaml` default; leave alone.           |
| `main.py` is a deprecation stub               | Keeps stale invocations failing loudly.               |

---

## 5. Where to add things

- **New model** → `models/<name>.py` + register in
  `models/__init__.py → MODEL_REGISTRY` + config block in
  `config.yaml`. See `models/arima.py` for the cleanest reference
  implementation (parallel fit, fallback chain, unified predict loop).
- **New metric** → add pure function to `evaluation/evaluation.py` and
  register it in `ALL_METRICS`.
- **New temporal aggregation level** → extend `TEMPORAL_FREQ_MAP` in
  `aggregation.py`.
- **New structural aggregation level** → just add its name to
  `config.yaml → hierarchy.structural_levels`; no code change needed.

---

## 6. Red flags / things to investigate before editing

- **`_check_coverage` failing for a specific model** → that model is
  silently dropping series. Fix the model, not the check.
- **`save()` / `load()` mismatch** → if a resumed `predict()` crashes,
  it usually means `load` didn't restore enough state. Compare against
  what `predict` reads.
- **Lag-model alignment** in `random_forest.py` / `lightgbm_model.py`
  was previously buggy for `strategy: "direct"` (per-series X / y
  lengths mismatched). The current fix trims every series to
  `min_length = N - n_lags - (horizon - 1)`. Don't regress this.
- **DeepAR / TimesFM freq handling** — `_freq` must be restored by
  `load()`. There used to be a missing `ESTIMATOR_FREQ_MAP` attribute in
  DeepAR; now the estimator takes `self._freq` directly.

---

## 7. Running locally (what the agent needs to know)

```powershell
poetry install
poetry shell

# edit top-of-file constants if needed
python train.py      # writes outputs/ and logs/train_<ts>.log
python evaluate.py   # writes per-level summaries and logs/evaluate_<ts>.log
```

No test suite exists. Validate changes by running the affected entry
point on the real `data/data.csv` and inspecting the newest file under
`logs/` plus the `evaluation_summary.csv` diff.

---

## 8. When unsure — ask

When a change would alter any of the items in §2 or §4, surface the
trade-off to the user before editing. Don't silently revert user
decisions (e.g. re-adding naive fallbacks, re-introducing CLI flags).

