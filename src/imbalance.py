import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import boosting
import evaluate
import tuning
from baselines import THRESHOLD
from forest import prepare_folds
from knn_svm import make_logistic
from preprocess import load_features
from split import DB

RESULTS = Path("results/imbalance.csv")
SEED = 42
METHODS = ["none", "class_weight", "undersample", "oversample", "smote"]
MODELS = ["logistic", "lightgbm"]
LEARNING_RATE = 0.05
STOP_METRIC = "average_precision"


def resample(method, Z, y, seed=SEED):
    if method in ("none", "class_weight"):
        return Z, y
    sampler = {
        "undersample": RandomUnderSampler,
        "oversample": RandomOverSampler,
        "smote": SMOTE,
    }[method](sampling_strategy=1.0, random_state=seed)
    return sampler.fit_resample(Z, y)


def odds_correction(p, true_ratio, trained_ratio):
    odds = p / (1 - p) * (true_ratio / trained_ratio)
    return odds / (1 + odds)


def odds_ratio(y):
    return y.sum() / (len(y) - y.sum())


def trained_ratio(method, y_fit):
    return odds_ratio(y_fit) if method == "none" else 1.0


def positive_weight(y):
    return (len(y) - y.sum()) / y.sum()


def fit_logistic(Z, y, method):
    model = make_logistic()
    if method == "class_weight":
        model.set_params(class_weight="balanced")
    Z_train, y_train = resample(method, Z, y)
    return model.fit(Z_train, y_train)


def fit_lightgbm(Z, y, method, learning_rate=LEARNING_RATE):
    Z_fit, Z_stop, y_fit, y_stop = boosting.stopping_split(Z, y)
    Z_train, y_train = resample(method, Z_fit, y_fit)
    params = {**tuning.lightgbm_params(tuning.DEFAULTS), "metric": "None"}
    if method == "class_weight":
        params["scale_pos_weight"] = positive_weight(y_fit)
    booster = boosting.make_booster(learning_rate, **params)
    booster.fit(Z_train, y_train, eval_X=Z_stop, eval_y=y_stop, eval_metric=STOP_METRIC, callbacks=[lgb.early_stopping(boosting.PATIENCE, verbose=False)])
    return booster


def clip_probability(p):
    return np.clip(p, evaluate.EPS, 1 - evaluate.EPS)


def fold_row(part, model_name, method):
    start = time.time()
    fit = fit_logistic if model_name == "logistic" else fit_lightgbm
    model = fit(part["Z_fit"], part["y_fit"], method)
    fit_seconds = time.time() - start
    p = clip_probability(model.predict_proba(part["Z_val"])[:, 1])
    p_corrected = odds_correction(p, odds_ratio(part["y_fit"]), trained_ratio(method, part["y_fit"]))
    scored = evaluate.score(part["y_val"], p, THRESHOLD)
    return {
        "fold": part["fold"],
        "model": model_name,
        "method": method,
        **{key: scored[key] for key in ("pr_auc", "roc_auc", "log_loss")},
        "log_loss_corrected": evaluate.log_loss(part["y_val"], clip_probability(p_corrected)),
        "mean_probability": float(p.mean()),
        "share_above_half": float((p > 0.5).mean()),
        "fit_seconds": fit_seconds,
    }


def run(parts, models=MODELS, methods=METHODS):
    rows = []
    for part in parts:
        for model_name in models:
            for method in methods:
                rows.append(fold_row(part, model_name, method))
        print(f"fold {part['fold']} done", flush=True)
    return pd.DataFrame(rows)


def paired_difference(table, column="pr_auc"):
    wide = table.pivot_table(index=["model", "fold"], columns="method", values=column)
    out = {}
    for model_name in wide.index.get_level_values("model").unique():
        part = wide.loc[model_name]
        for method in part.columns:
            diff = part[method] - part["none"]
            out[(model_name, method)] = {"difference": diff.mean(), "std": diff.std(), "better_folds": int((diff > 0).sum())}
    return pd.DataFrame(out).T


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = prepare_folds(X, y, scale=True, clip=True)
    RESULTS.parent.mkdir(exist_ok=True)
    table = run(parts)
    table.to_csv(RESULTS, index=False, float_format="%.8g")
    columns = ["pr_auc", "roc_auc", "log_loss", "log_loss_corrected", "mean_probability", "share_above_half", "fit_seconds"]
    print(table.groupby(["model", "method"])[columns].mean().round(4).to_string())
    print(paired_difference(table).round(4).to_string())


if __name__ == "__main__":
    main()
