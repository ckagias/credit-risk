import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import boosting
import evaluate
import features
import forest_sql
import logistic
import threshold
import tuning
from feature_gain import tuning_best_params
from preprocess import load_features, make_preprocessor
from split import DB

RESULTS = Path("results/holdout.csv")
COST_RESULTS = Path("results/holdout_cost.csv")
SEED = 42
BOOTSTRAPS = 1000
LOGISTIC_C = 0.03
FOREST_FEATURES = 30
CUT = threshold.bayes_threshold(threshold.AMOUNT)
FAMILIES = ["base rate", "EXT_SOURCE mean, logistic", "logistic regression", "random forest", "LightGBM"]
SCORES = ["pr_auc", "roc_auc", "log_loss"]


def fit_lightgbm(X, y):
    pre = make_preprocessor(scale=False).fit(X)
    booster = boosting.fit_with_early_stopping(pre.transform(X), y, tuning.TUNE_RATE, **tuning.lightgbm_params(tuning_best_params()))
    return lambda X_new: booster.predict_proba(pre.transform(X_new))[:, 1]


def fit_sklearn(model, columns=None):
    def fit(X, y):
        selected = X if columns is None else X[columns]
        model.fit(selected, y)
        return lambda X_new: model.predict_proba(X_new if columns is None else X_new[columns])[:, 1]

    return fit


def fitters():
    return {
        "base rate": fit_sklearn(baselines.base_rate_model(), baselines.EXT_SOURCES),
        "EXT_SOURCE mean, logistic": fit_sklearn(baselines.ext_source_model(), baselines.EXT_SOURCES),
        "logistic regression": fit_sklearn(logistic.make_model(C=LOGISTIC_C, clip=True)),
        "random forest": fit_sklearn(forest_sql.make_model(FOREST_FEATURES)),
        "LightGBM": fit_lightgbm,
    }


def interval_rows(name, y, p, boots=BOOTSTRAPS):
    scored = evaluate.bootstrap(y, p, n_boot=boots, seed=SEED, threshold=CUT)
    return [{"model": name, "metric": metric, "estimate": scored.loc[metric, "estimate"], "low": scored.loc[metric, "low"], "high": scored.loc[metric, "high"]} for metric in SCORES]


def cost_row(y, p, size, cut=CUT, boots=BOOTSTRAPS, seed=SEED):
    y, p, size = np.asarray(y), np.asarray(p), np.asarray(size)
    scheme = threshold.AMOUNT

    def measures(index):
        yy, pp, ss = y[index], p[index], size[index]
        cost = 1000 * threshold.cost_per_applicant(yy, pp, np.array([cut]), scheme, ss)[0]
        nobody = 1000 * threshold.cost_per_applicant(yy, pp, np.array([1.0 + 1e-9]), scheme, ss)[0]
        refused = pp >= cut
        return np.array([cost, nobody, 1 - cost / nobody, refused.mean(), refused[yy == 1].mean()])

    point = measures(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.array([measures(rng.integers(0, len(y), len(y))) for _ in range(boots)])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    names = ["cost_per_1000", "cost_refuse_nobody", "saving", "share_refused", "defaulters_refused"]
    row = {"threshold": cut}
    for k, name in enumerate(names):
        row.update({name: point[k], f"{name}_low": low[k], f"{name}_high": high[k]})
    return row


def run(X_train, y_train, X_eval, y_eval, size_train_mean, models=None, boots=BOOTSTRAPS):
    rows, predictions = [], {}
    for name, fit in (models or fitters()).items():
        predict = fit(X_train, y_train)
        predictions[name] = predict(X_eval)
        rows += interval_rows(name, y_eval, predictions[name], boots)
        print(f"{name}: PR-AUC {rows[-3]['estimate']:.4f} ({rows[-3]['low']:.4f} to {rows[-3]['high']:.4f})", flush=True)
    size = (X_eval["AMT_CREDIT"] / size_train_mean).to_numpy()
    return pd.DataFrame(rows), pd.DataFrame([{"model": "LightGBM", **cost_row(y_eval, predictions["LightGBM"], size, boots=boots)}])


def main():
    if RESULTS.exists():
        sys.exit(f"{RESULTS} exists: the holdout has already been scored and is not scored twice")
    con = sqlite3.connect(DB)
    X_train, y_train = load_features(con, "train", features.GROUPS)
    X_hold, y_hold = load_features(con, "holdout", features.GROUPS)
    con.close()
    print(f"train {len(y_train):,} applicants, holdout {len(y_hold):,}", flush=True)
    table, cost = run(X_train, y_train, X_hold, y_hold, X_train["AMT_CREDIT"].mean())
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.6f")
    cost.to_csv(COST_RESULTS, index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print(cost.round(3).T.to_string())


if __name__ == "__main__":
    main()
