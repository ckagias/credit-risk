import sqlite3
import time
from pathlib import Path
from typing import Callable, NamedTuple

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import evaluate
from baselines import THRESHOLD
from preprocess import load_features, make_preprocessor
from split import DB

CURVE = Path("results/lightgbm_curve.csv")
STOPPED = Path("results/lightgbm_stopped.csv")
SEED = 42
LEARNING_RATES = [0.3, 0.1, 0.05, 0.02]
MAX_ROUNDS = 3000
PATIENCE = 100
STOP_SHARE = 0.1
CURVE_ROUNDS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
TRAIN_SAMPLE = 30000
CURVE_REACH = 3


def make_booster(learning_rate, n_estimators=MAX_ROUNDS, **params):
    return lgb.LGBMClassifier(
        learning_rate=learning_rate, n_estimators=n_estimators, random_state=SEED, verbose=-1, **params
    )


def stopping_split(Z, y):
    Z_fit, Z_stop, y_fit, y_stop = train_test_split(Z, y, test_size=STOP_SHARE, stratify=y, random_state=SEED)
    return Z_fit, Z_stop, y_fit, y_stop


def fit_with_early_stopping(Z, y, learning_rate, patience=PATIENCE, max_rounds=MAX_ROUNDS):
    Z_fit, Z_stop, y_fit, y_stop = stopping_split(Z, y)
    booster = make_booster(learning_rate, max_rounds)
    booster.fit(
        Z_fit,
        y_fit,
        eval_X=Z_stop,
        eval_y=y_stop,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return booster


def fit_fixed_rounds(Z, y, learning_rate, rounds):
    Z_fit, _, y_fit, _ = stopping_split(Z, y)
    return make_booster(learning_rate, rounds).fit(Z_fit, y_fit)


def score_at(booster, Z, y, rounds):
    return evaluate.score(y, booster.predict_proba(Z, num_iteration=rounds)[:, 1], THRESHOLD)


class Backend(NamedTuple):
    fit_with_early_stopping: Callable
    fit_fixed_rounds: Callable
    score_at: Callable
    best_round: Callable


LIGHTGBM = Backend(fit_with_early_stopping, fit_fixed_rounds, score_at, lambda booster: booster.best_iteration_)


def score_fold(fold, Z_fit, y_fit, Z_val, y_val, learning_rates=LEARNING_RATES, rounds=CURVE_ROUNDS, backend=LIGHTGBM, **fit_args):
    sample = pd.Series(range(len(y_fit))).sample(min(TRAIN_SAMPLE, len(y_fit)), random_state=SEED).to_numpy()
    Z_sample, y_sample = Z_fit[sample], y_fit[sample]
    curve, stopped = [], []
    for learning_rate in learning_rates:
        start = time.time()
        stopper = backend.fit_with_early_stopping(Z_fit, y_fit, learning_rate, **fit_args)
        seconds = time.time() - start
        best = backend.best_round(stopper)
        stopped.append(
            {
                "fold": fold,
                "learning_rate": learning_rate,
                "best_round": best,
                "seconds": seconds,
                **backend.score_at(stopper, Z_val, y_val, best),
            }
        )
        reach = min(CURVE_REACH * best, fit_args.get("max_rounds", MAX_ROUNDS))
        booster = backend.fit_fixed_rounds(Z_fit, y_fit, learning_rate, reach)
        for k in [r for r in rounds if r <= reach]:
            for split, Z, y in (("train", Z_sample, y_sample), ("validation", Z_val, y_val)):
                curve.append({"fold": fold, "learning_rate": learning_rate, "rounds": k, "split": split, **backend.score_at(booster, Z, y, k)})
    return curve, stopped


def cross_validate_boosting(X, y, backend=LIGHTGBM, **kwargs):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    curve, stopped = [], []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        pre = make_preprocessor(scale=False).fit(X.iloc[fit_idx])
        c, s = score_fold(fold, pre.transform(X.iloc[fit_idx]), y[fit_idx], pre.transform(X.iloc[val_idx]), y[val_idx], backend=backend, **kwargs)
        curve += c
        stopped += s
        print(f"fold {fold} done", flush=True)
    return pd.DataFrame(curve), pd.DataFrame(stopped)


def run(backend, curve_path, stopped_path):
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    start = time.time()
    curve, stopped = cross_validate_boosting(X, y, backend=backend)
    print(f"boosting took {time.time() - start:.0f}s")
    curve_path.parent.mkdir(exist_ok=True)
    curve.to_csv(curve_path, index=False, float_format="%.6f")
    stopped.to_csv(stopped_path, index=False, float_format="%.6f")
    table = stopped.groupby("learning_rate")[["best_round", "seconds", "pr_auc", "roc_auc", "log_loss"]].agg(["mean", "std"])
    print(table.round(4).to_string())


def main():
    run(LIGHTGBM, CURVE, STOPPED)


if __name__ == "__main__":
    main()
