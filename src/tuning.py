import sqlite3
import time
from pathlib import Path

import optuna
import pandas as pd

import boosting
import evaluate
from forest import prepare_folds
from preprocess import load_features
from split import DB

STUDY_DB = Path("results/optuna_lightgbm.db")
TRIALS = Path("results/optuna_trials.csv")
IMPORTANCE = Path("results/optuna_importance.csv")
CHECK = Path("results/optuna_check.csv")
SEED = 42
CHECK_SEED = 7
N_TRIALS = 60
TUNE_RATE = 0.05
CHECK_RATES = [0.05, 0.02]
DEFAULTS = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 1e-3,
    "reg_lambda": 1e-3,
}


def suggest(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def lightgbm_params(params):
    return {**params, "subsample_freq": 1 if params["subsample"] < 1.0 else 0}


def fold_scores(part, params, learning_rate=TUNE_RATE):
    booster = boosting.fit_with_early_stopping(part["Z_fit"], part["y_fit"], learning_rate, **lightgbm_params(params))
    best = booster.best_iteration_
    return {"best_round": best, **boosting.score_at(booster, part["Z_val"], part["y_val"], best)}


def objective(trial, parts):
    params = suggest(trial)
    rows = []
    for step, part in enumerate(parts):
        rows.append(fold_scores(part, params))
        trial.report(pd.DataFrame(rows)["pr_auc"].mean(), step)
        if trial.should_prune():
            trial.set_user_attr("folds_run", len(rows))
            raise optuna.TrialPruned()
    table = pd.DataFrame(rows)
    trial.set_user_attr("folds_run", len(rows))
    trial.set_user_attr("pr_auc_std", float(table["pr_auc"].std()))
    for column in ("roc_auc", "log_loss", "best_round"):
        trial.set_user_attr(column, float(table[column].mean()))
    return float(table["pr_auc"].mean())


def make_study(storage=None, n_startup_trials=10):
    study = optuna.create_study(
        study_name="lightgbm",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1),
    )
    if not study.trials:
        study.enqueue_trial(DEFAULTS)
    return study


def tune(parts, n_trials=N_TRIALS, storage=None, **study_args):
    study = make_study(storage, **study_args)
    finished = sum(trial.state.is_finished() for trial in study.trials)
    remaining = n_trials - finished
    if remaining > 0:
        study.optimize(lambda trial: objective(trial, parts), n_trials=remaining)
    return study


def trials_table(study):
    table = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    table["duration"] = table["duration"].dt.total_seconds()
    return table.rename(columns=lambda name: name.replace("params_", "").replace("user_attrs_", ""))


def best_params(study):
    return dict(study.best_params)


def check_on_fresh_folds(X, y, params, rates=CHECK_RATES):
    parts = prepare_folds(X, y, seed=CHECK_SEED)
    rows = []
    for name, chosen in (("default", DEFAULTS), ("tuned", params)):
        for rate in rates:
            for part in parts:
                rows.append({"model": name, "learning_rate": rate, "fold": part["fold"], **fold_scores(part, chosen, rate)})
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    STUDY_DB.parent.mkdir(exist_ok=True)
    start = time.time()
    study = tune(prepare_folds(X, y), storage=f"sqlite:///{STUDY_DB}")
    print(f"tuning took {time.time() - start:.0f}s")
    table = trials_table(study)
    table.to_csv(TRIALS, index=False, float_format="%.6f")
    importance = pd.Series(optuna.importance.get_param_importances(study), name="importance")
    importance.rename_axis("parameter").reset_index().to_csv(IMPORTANCE, index=False, float_format="%.6f")
    print(importance.round(3).to_string())
    print(f"best trial {study.best_trial.number}: PR-AUC {study.best_value:.4f}, {best_params(study)}")
    check = check_on_fresh_folds(X, y, best_params(study))
    check.to_csv(CHECK, index=False, float_format="%.6f")
    print(check.groupby(["model", "learning_rate"])[["best_round", "pr_auc", "roc_auc", "log_loss"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
