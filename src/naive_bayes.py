import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB

import evaluate
from baselines import THRESHOLD
from forest import prepare_folds
from knn_svm import EXT_COLUMNS
from preprocess import load_features
from split import DB

RESULTS = Path("results/naive_bayes.csv")
SEED = 42
SMOOTHING_GRID = [1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def make_bayes(var_smoothing=1e-9):
    return GaussianNB(var_smoothing=var_smoothing)


def make_calibrated(var_smoothing=1e-9):
    return CalibratedClassifierCV(make_bayes(var_smoothing), method="sigmoid", ensemble=False, cv=3)


def fold_row(part, columns, experiment, setting, model):
    start = time.time()
    model.fit(part["Z_fit"], part["y_fit"])
    fit_seconds = time.time() - start
    p = model.predict_proba(part["Z_val"])[:, 1]
    return {
        "fold": part["fold"],
        "columns": columns,
        "experiment": experiment,
        "setting": setting,
        "n_columns": part["Z_fit"].shape[1],
        **evaluate.score(part["y_val"], p, THRESHOLD),
        "mean_p": float(p.mean()),
        "share_p_above_half": float((p > 0.5).mean()),
        "fit_seconds": fit_seconds,
    }


def continuous_mask(Z):
    return np.array([len(np.unique(column)) > 2 for column in Z.T])


def continuous_parts(parts):
    subset = []
    for part in parts:
        mask = continuous_mask(part["Z_fit"])
        subset.append({**part, "Z_fit": part["Z_fit"][:, mask], "Z_val": part["Z_val"][:, mask]})
    return subset


def best_smoothing(rows):
    table = pd.DataFrame(rows)
    return float(table.groupby("setting")["pr_auc"].mean().idxmax())


def sweep_and_calibrate(parts, columns, grid=SMOOTHING_GRID):
    sweep = [fold_row(part, columns, "smoothing", v, make_bayes(v)) for part in parts for v in grid]
    best = best_smoothing(sweep)
    calibrated = [fold_row(part, columns, "calibrated", best, make_calibrated(best)) for part in parts]
    print(f"{columns}: best var_smoothing = {best:g}", flush=True)
    return sweep + calibrated


def run(X, y):
    parts = prepare_folds(X, y, scale=True, clip=True)
    ext_parts = prepare_folds(X[EXT_COLUMNS], y, scale=True, clip=True)
    rows = sweep_and_calibrate(parts, "all")
    rows += sweep_and_calibrate(continuous_parts(parts), "continuous")
    rows += sweep_and_calibrate(ext_parts, "ext")
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    results = run(X, y)
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False, float_format="%.8g")
    table = results.groupby(["columns", "experiment", "setting"])[["n_columns", "pr_auc", "roc_auc", "log_loss", "precision", "recall", "mean_p", "share_p_above_half", "fit_seconds"]].mean()
    print(table.round(4).to_string())
    print(f"base rate {y.mean():.4f}")


if __name__ == "__main__":
    main()
