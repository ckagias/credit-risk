import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import evaluate
from baselines import THRESHOLD
from preprocess import load_features, make_preprocessor
from split import DB

CURVE = Path("results/regularization_curve.csv")
PATH = Path("results/coefficient_path.csv")
SEED = 42
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
L1_MAX_C = 0.1
SETTINGS = {
    "L2": {"l1_ratio": 0.0, "solver": "lbfgs"},
    "L1": {"l1_ratio": 1.0, "solver": "liblinear"},
}


def c_values(penalty):
    return [c for c in C_GRID if penalty == "L2" or c <= L1_MAX_C]


def fit(Z, y, penalty, C):
    return LogisticRegression(C=C, max_iter=1000, **SETTINGS[penalty]).fit(Z, y)


def score_fold(fold, X_fit, y_fit, X_val, y_val):
    pre = make_preprocessor().fit(X_fit)
    Z_fit, Z_val = pre.transform(X_fit), pre.transform(X_val)
    rows = []
    for penalty in SETTINGS:
        for C in c_values(penalty):
            model = fit(Z_fit, y_fit, penalty, C)
            for split, Z, y in (("train", Z_fit, y_fit), ("validation", Z_val, y_val)):
                scores = evaluate.score(y, model.predict_proba(Z)[:, 1], THRESHOLD)
                rows.append({"fold": fold, "penalty": penalty, "C": C, "split": split, **scores})
    return rows


def validation_curve(X, y, n_jobs=5):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    jobs = (
        delayed(score_fold)(k, X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx])
        for k, (fit_idx, val_idx) in enumerate(folds)
    )
    return pd.DataFrame([row for rows in Parallel(n_jobs=n_jobs)(jobs) for row in rows])


def coefficient_path(X, y):
    pre = make_preprocessor().fit(X)
    Z = pre.transform(X)
    names = pre.get_feature_names_out()
    rows = []
    for penalty in SETTINGS:
        for C in c_values(penalty):
            weights = fit(Z, y, penalty, C).coef_.ravel()
            rows.append(pd.DataFrame({"penalty": penalty, "C": C, "feature": names, "coefficient": weights}))
    return pd.concat(rows, ignore_index=True)


def summarize(curve):
    return (
        curve.groupby(["penalty", "split", "C"])[evaluate.METRICS]
        .agg(["mean", "std"])
        .round(6)
    )


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    start = time.time()
    curve = validation_curve(X, y)
    print(f"validation curve took {time.time() - start:.0f}s", flush=True)
    CURVE.parent.mkdir(exist_ok=True)
    curve.to_csv(CURVE, index=False, float_format="%.6f")
    validation = curve[curve["split"] == "validation"]
    table = validation.groupby(["penalty", "C"])[["pr_auc", "roc_auc", "log_loss"]].agg(["mean", "std"])
    print(table.round(4).to_string())
    start = time.time()
    path = coefficient_path(X, y)
    print(f"coefficient path took {time.time() - start:.0f}s", flush=True)
    path.to_csv(PATH, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
