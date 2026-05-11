from __future__ import annotations

import copy
import logging
import sys
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import GridSampler, TPESampler

from utils.aggregation_utils import get_level_sample_frac, iter_levels
from utils.utils import load_config, load_hpo_results, load_raw_data, write_hpo_results
from utils.evaluation_utils import compute_metrics
from utils.logging_utils import setup_logging, timed
from models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Script-level configuration
# ---------------------------------------------------------------------------
CONFIG_PATH: str = "config/config.yaml"
LOG_FILE: str = "logs/tune.log"
RESUME: bool = True  # Skip (level, model) pairs already present in the output tuned config

logger = logging.getLogger("tune")

_TRIAL_BUDGET_CAPS: Dict[str, Tuple[str, str]] = {
    "deepar": ("max_epochs_hpo", "max_epochs"),
    "nhits": ("max_steps_hpo", "max_steps"),
}


def compute_val_dates(config: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (hpo_train_cutoff, train_end_date)."""
    num_val_periods: int = config["hpo"]["num_val_periods"]
    train_end_date = pd.Timestamp(config["experiment"]["train_end_date"])
    hpo_train_cutoff = train_end_date - pd.DateOffset(months=num_val_periods)
    return pd.Timestamp(hpo_train_cutoff), train_end_date


def _enumerable_grid(
    search_space: Dict[str, Any],
    max_size: int,
) -> Dict[str, List[Any]] | None:
    """If every param is categorical (or a small linear int range) and the
    full Cartesian product has <= max_size points, return the grid for use
    with GridSampler. Otherwise return None to fall back to TPE.
    """
    if not search_space:
        return None
    grid: Dict[str, List[Any]] = {}
    total = 1
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "categorical":
            choices = list(spec["choices"])
        elif ptype == "int" and not spec.get("log", False):
            low, high = int(spec["low"]), int(spec["high"])
            step = int(spec.get("step", 1))
            if step <= 0:
                return None
            choices = list(range(low, high + 1, step))
        else:
            return None
        if not choices:
            return None
        grid[name] = choices
        total *= len(choices)
        if total > max_size:
            return None
    return grid


def suggest_params(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a params dict with trial suggestions merged over base_params defaults."""
    params = copy.deepcopy(base_params)
    for param_name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"]
            )
        elif param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, spec["choices"]
            )
    return params


def _score_forecast(
    forecast_df: pd.DataFrame,
    val_df: pd.DataFrame,
    time_col: str,
    target_col: str,
) -> float:
    """Merge forecast with val actuals and return WMAPE. Raises on failure."""
    forecast_df = forecast_df.copy()
    forecast_df[time_col] = pd.to_datetime(forecast_df[time_col])
    val = val_df.copy()
    val[time_col] = pd.to_datetime(val[time_col])

    merged = pd.merge(
        forecast_df,
        val[["ts_id", time_col, target_col]],
        on=["ts_id", time_col],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError(
            f"Empty forecast/val intersection (forecast rows={len(forecast_df)}, "
            f"val rows={len(val_df)}); cannot score trial."
        )

    metrics = compute_metrics(
        merged[target_col].values.astype(float),
        merged["forecast"].values.astype(float),
    )
    score = metrics.get("WMAPE")
    if score is None or not np.isfinite(score):
        raise RuntimeError(f"Non-finite WMAPE returned: {score!r}")
    return float(score)


def build_objective(
    model_name: str,
    model_type: str,
    base_params: Dict[str, Any],
    model_hpo: Dict[str, Any],
    trial_train_df: pd.DataFrame,
    trial_val_df: pd.DataFrame,
    trial_config: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective closure for one model at one hierarchy level."""
    search_space = model_hpo.get("search_space", {})
    time_col = trial_config["data"]["time_col"]
    target_col = trial_config["data"]["target_col"]
    horizon = trial_config["experiment"]["horizon"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space, base_params)

        # Cap DL training budget for trials so HPO runs in reasonable time.
        # The final train.py run always uses the full budget from params.
        budget_cap = _TRIAL_BUDGET_CAPS.get(model_type)
        if budget_cap is not None:
            hpo_field, params_field = budget_cap
            if hpo_field in model_hpo:
                params[params_field] = model_hpo[hpo_field]

        model = MODEL_REGISTRY[model_type](params=params)
        model.fit(trial_train_df, trial_config)
        forecast_df = model.predict(horizon, trial_config)
        forecast_df["forecast"] = forecast_df["forecast"].clip(lower=0)
        return _score_forecast(forecast_df, trial_val_df, time_col, target_col)

    return objective


def tune_level(
    level_label: str,
    model_configs: List[Dict[str, Any]],
    global_hpo: Dict[str, Any],
    trial_train_df: pd.DataFrame,
    trial_val_df: pd.DataFrame,
    level_config: Dict[str, Any],
    existing_results: Dict[str, Dict[str, Any]] | None = None,
    on_model_done: Callable[[str, str, Dict[str, Any]], None] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Run HPO for all HPO-enabled models at one hierarchy level."""
    n_trials: int = global_hpo.get("n_trials", 30)
    seed: int = level_config["experiment"].get("seed", 42)

    if trial_val_df.empty:
        logger.warning(f"  [{level_label}] val df is empty — skipping level.")
        return {}

    # Infer trial horizon dynamically from val rows per ts_id.
    trial_horizon = int(trial_val_df.groupby("ts_id").size().max())

    trial_config = copy.deepcopy(level_config)
    trial_config["experiment"]["horizon"] = trial_horizon

    best_params_map: Dict[str, Dict[str, Any]] = dict(existing_results or {})
    optuna.logging.set_verbosity(optuna.logging.INFO)

    for model_config in model_configs:
        model_name = model_config["name"]
        model_type = model_config.get("type", model_name)
        model_hpo = model_config.get("hpo", {})

        if not model_hpo.get("enabled", False):
            logger.info(f"  [{level_label}/{model_name}] HPO disabled, skipping.")
            continue

        if model_type not in MODEL_REGISTRY:
            logger.warning(
                f"  [{level_label}/{model_name}] type '{model_type}' not in MODEL_REGISTRY, skipping."
            )
            continue

        if existing_results and model_name in existing_results:
            logger.info(
                f"  [SKIP] {level_label}/{model_name} — already tuned "
                f"(params={existing_results[model_name]})"
            )
            continue

        base_params = copy.deepcopy(model_config.get("params", {}))

        horizon_override = model_hpo.get("horizon_override")
        if horizon_override is not None and horizon_override < trial_horizon:
            time_col = level_config["data"]["time_col"]
            model_trial_horizon = int(horizon_override)
            model_trial_val_df = (
                trial_val_df.sort_values([time_col])
                .groupby("ts_id", sort=False)
                .head(model_trial_horizon)
                .copy()
            )
            model_trial_config = copy.deepcopy(level_config)
            model_trial_config["experiment"]["horizon"] = model_trial_horizon
            logger.info(
                f"  [{level_label}/{model_name}] horizon_override={model_trial_horizon} "
                f"(val rows {len(trial_val_df)} → {len(model_trial_val_df)})"
            )
        else:
            model_trial_horizon = trial_horizon
            model_trial_val_df = trial_val_df
            model_trial_config = trial_config

        logger.info(
            f"  [{level_label}/{model_name}] starting {n_trials} trials "
            f"(trial_horizon={model_trial_horizon})"
        )

        with timed(f"hpo {level_label}/{model_name}", logger):
            search_space = model_hpo.get("search_space", {})
            grid = _enumerable_grid(search_space, max_size=n_trials)
            if grid is not None:
                sampler = GridSampler(grid, seed=seed)
                effective_n_trials = int(np.prod([len(v) for v in grid.values()]))
                logger.info(
                    f"  [{level_label}/{model_name}] using GridSampler "
                    f"({effective_n_trials} combinations <= n_trials={n_trials})"
                )
            else:
                n_startup = max(5, n_trials // 4)
                sampler = TPESampler(seed=seed, n_startup_trials=n_startup)
                effective_n_trials = n_trials
            study = optuna.create_study(direction="minimize", sampler=sampler)
            objective = build_objective(
                model_name=model_name,
                model_type=model_type,
                base_params=base_params,
                model_hpo=model_hpo,
                trial_train_df=trial_train_df,
                trial_val_df=model_trial_val_df,
                trial_config=model_trial_config,
            )

            def _trial_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial,
                                 _model=model_name, _level=level_label,
                                 _total=effective_n_trials) -> None:
                value = trial.value if trial.value is not None else float("nan")
                best = study.best_value if study.best_trial is not None else float("nan")
                logger.info(
                    f"    [{_level}/{_model}] trial {trial.number + 1}/{_total} "
                    f"WMAPE={value:.3f}%  best={best:.3f}%  params={trial.params}"
                )

            study.optimize(
                objective,
                n_trials=effective_n_trials,
                show_progress_bar=False,
                callbacks=[_trial_progress],
            )

        logger.info(
            f"  [{level_label}/{model_name}] best WMAPE={study.best_value:.3f}%  "
            f"params={study.best_params}"
        )
        best_params_map[model_name] = study.best_params
        if on_model_done is not None:
            on_model_done(level_label, model_name, study.best_params)

    return best_params_map


def _iter_tuning_levels(
    config: Dict[str, Any],
    dataframe: pd.DataFrame,
    train_end_date: pd.Timestamp,
) -> Iterator[Tuple[str, Dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    """Yield (level_label, level_config, level_train, level_val) for every
    hierarchy level including base.    """
    time_col = config["data"]["time_col"]
    seed = config["experiment"].get("seed", 42)
    in_range = dataframe[dataframe[time_col] <= train_end_date].copy()

    for level_label, level_config, level_frame in iter_levels(config, in_range):
        sample_frac = get_level_sample_frac(level_label, config)
        if sample_frac is not None:
            all_ids = level_frame["ts_id"].unique()
            sample_n = max(1, int(round(len(all_ids) * sample_frac)))
            rng = np.random.default_rng(seed)
            sampled = rng.choice(all_ids, size=sample_n, replace=False)
            level_frame = level_frame[level_frame["ts_id"].isin(sampled)].copy()
            logger.info(
                f"  [{level_label}] HPO sampling: {sample_n}/{len(all_ids)} ts_ids "
                f"({sample_frac:.0%}, seed={seed})"
            )

        level_num_val = level_config["hpo"]["num_val_periods"]
        level_cutoff = train_end_date - pd.DateOffset(months=level_num_val)
        train = level_frame[level_frame[time_col] <= level_cutoff].copy()
        val = level_frame[
            (level_frame[time_col] > level_cutoff)
            & (level_frame[time_col] <= train_end_date)
        ].copy()
        yield level_label, level_config, train, val


def run_tuning(config: Dict[str, Any], resume: bool = False) -> None:
    global_hpo = config["hpo"]

    dataframe = load_raw_data(config)
    hpo_train_cutoff, train_end_date = compute_val_dates(config)

    logger.info(
        f"HPO split — train: <= {hpo_train_cutoff.date()}, "
        f"val: ({hpo_train_cutoff.date()}, {train_end_date.date()}]"
    )

    # Load existing tuned results if resuming.
    hpo_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if resume:
        hpo_results = load_hpo_results(config)

    def _persist(level_label: str, model_name: str, best_params: Dict[str, Any]) -> None:
        hpo_results.setdefault(level_label, {})[model_name] = best_params
        write_hpo_results(config, hpo_results)

    for level_label, level_config, level_train, level_val in _iter_tuning_levels(
        config, dataframe, train_end_date
    ):
        with timed(f"hpo level={level_label}", logger):
            logger.info(
                f"  {level_label}: train={len(level_train)} rows, "
                f"val={len(level_val)} rows, "
                f"series={level_train['ts_id'].nunique()}"
            )
            best = tune_level(
                level_label=level_label,
                model_configs=level_config["models"],
                global_hpo=global_hpo,
                trial_train_df=level_train,
                trial_val_df=level_val,
                level_config=level_config,
                existing_results=hpo_results.get(level_label),
                on_model_done=_persist,
            )
            if best:
                hpo_results[level_label] = best

    written = write_hpo_results(config, hpo_results)
    logger.info(f"Tuned config written → {written.resolve()}")


def main() -> int:
    setup_logging(LOG_FILE)
    config = load_config(CONFIG_PATH)
    if "hpo" not in config:
        logger.error("No 'hpo' section found in config — add it before running tune.py.")
        return 1
    try:
        with timed("full HPO run", logger):
            run_tuning(config, resume=RESUME)
    except Exception:
        logger.exception("HPO run failed at top level.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
