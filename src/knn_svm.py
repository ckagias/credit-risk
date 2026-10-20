import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

import evaluate
from baselines import THRESHOLD
from forest import prepare_folds
from preprocess import load_features
from split import DB

RESULTS = Path("results/knn_svm.csv")
EXT_RESULTS = Path("results/knn_ext.csv")
SEED = 42
SIZES = [2000, 5000, 10000, 20000]
K_GRID = [5, 15, 50, 150, 500, 1500]
KNN_SWEEP_SIZE = 20000
C_GRID = [0.1, 1, 10]
GAMMA_GRID = [0.01, 0.1, 1.0]
SVM_SWEEP_SIZE = 5000
LINEAR_C_GRID = [0.001, 0.01]
LOGISTIC_C = 0.03
EXT_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def make_knn(k):
    return KNeighborsClassifier(n_neighbors=k, algorithm="brute", n_jobs=-1)


def make_svm(Z, C=1.0, gamma_multiplier=1.0, class_weight="balanced"):
    gamma = gamma_multiplier / (Z.shape[1] * Z.var())
    return CalibratedClassifierCV(SVC(C=C, kernel="rbf", gamma=gamma, class_weight=class_weight), method="sigmoid", ensemble=False, cv=3)


def make_linear_svm(C):
    return CalibratedClassifierCV(LinearSVC(C=C, dual=False), method="sigmoid", cv=3)


def make_logistic():
    return LogisticRegression(C=LOGISTIC_C, max_iter=1000)


def subsample(part, n, seed=SEED):
    if n >= len(part["y_fit"]):
        return part["Z_fit"], part["y_fit"]
    Z, _, y, _ = train_test_split(part["Z_fit"], part["y_fit"], train_size=n, stratify=part["y_fit"], random_state=seed)
    return Z, y


def timed_score(model, Z_fit, y_fit, Z_val, y_val):
    start = time.time()
    model.fit(Z_fit, y_fit)
    fit_seconds = time.time() - start
    start = time.time()
    p = model.predict_proba(Z_val)[:, 1]
    predict_seconds = time.time() - start
    return {**evaluate.score(y_val, p, THRESHOLD), "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}


def row(part, experiment, model, setting, n, scored):
    return {"fold": part["fold"], "experiment": experiment, "model": model, "setting": setting, "n_fit": n, **scored}


def knn_sweep(parts, k_grid=K_GRID, n=KNN_SWEEP_SIZE):
    rows = []
    for part in parts:
        Z, y = subsample(part, n)
        for k in k_grid:
            rows.append(row(part, "knn_k", "knn", k, len(y), timed_score(make_knn(k), Z, y, part["Z_val"], part["y_val"])))
    return rows


def svm_setting(C, gamma_multiplier):
    return f"C={C:g}, gamma x{gamma_multiplier:g}"


def parse_svm_setting(setting):
    C, gamma = setting.split(", gamma x")
    return float(C.removeprefix("C=")), float(gamma)


def svm_sweep(parts, c_grid=C_GRID, gamma_grid=GAMMA_GRID, n=SVM_SWEEP_SIZE):
    rows = []
    for part in parts:
        Z, y = subsample(part, n)
        for C in c_grid:
            for gamma in gamma_grid:
                scored = timed_score(make_svm(Z, C, gamma), Z, y, part["Z_val"], part["y_val"])
                rows.append(row(part, "svm_c_gamma", "svm", svm_setting(C, gamma), len(y), scored))
    return rows


def svm_unweighted_rows(parts, C, gamma_multiplier, n=SVM_SWEEP_SIZE):
    rows = []
    for part in parts:
        Z, y = subsample(part, n)
        scored = timed_score(make_svm(Z, C, gamma_multiplier, class_weight=None), Z, y, part["Z_val"], part["y_val"])
        rows.append(row(part, "svm_unweighted", "svm", svm_setting(C, gamma_multiplier), len(y), scored))
    return rows


def size_curve(parts, k, C, gamma_multiplier, sizes=SIZES):
    rows = []
    for part in parts:
        for n in sizes:
            Z, y = subsample(part, n)
            models = {"knn": (make_knn(k), k), "svm": (make_svm(Z, C, gamma_multiplier), svm_setting(C, gamma_multiplier)), "logistic": (make_logistic(), LOGISTIC_C)}
            for name, (model, setting) in models.items():
                rows.append(row(part, "size", name, setting, len(y), timed_score(model, Z, y, part["Z_val"], part["y_val"])))
    return rows


def linear_svm_rows(parts, c_grid=LINEAR_C_GRID):
    rows = []
    for part in parts:
        n = len(part["y_fit"])
        for C in c_grid:
            rows.append(row(part, "linear_svm", "linear_svm", C, n, timed_score(make_linear_svm(C), part["Z_fit"], part["y_fit"], part["Z_val"], part["y_val"])))
        rows.append(row(part, "linear_svm", "logistic", LOGISTIC_C, n, timed_score(make_logistic(), part["Z_fit"], part["y_fit"], part["Z_val"], part["y_val"])))
    return rows


def ext_source_rows(X, y, k_grid=K_GRID, n=KNN_SWEEP_SIZE):
    parts = prepare_folds(X[EXT_COLUMNS], y, scale=True, clip=True)
    rows = []
    for part in parts:
        Z, y_fit = subsample(part, n)
        for k in k_grid:
            rows.append(row(part, "ext_columns", "knn", k, len(y_fit), timed_score(make_knn(k), Z, y_fit, part["Z_val"], part["y_val"])))
        rows.append(row(part, "ext_columns", "logistic", LOGISTIC_C, len(y_fit), timed_score(make_logistic(), Z, y_fit, part["Z_val"], part["y_val"])))
    return rows


def best_setting(rows, experiment):
    table = pd.DataFrame(rows).query("experiment == @experiment")
    return table.groupby("setting")["pr_auc"].mean().idxmax()


def run(parts):
    start = time.time()
    knn_rows = knn_sweep(parts)
    k = int(best_setting(knn_rows, "knn_k"))
    print(f"k sweep took {time.time() - start:.0f}s, best k = {k}", flush=True)
    start = time.time()
    svm_rows = svm_sweep(parts)
    best_svm = best_setting(svm_rows, "svm_c_gamma")
    C, gamma = parse_svm_setting(best_svm)
    svm_rows += svm_unweighted_rows(parts, C, gamma)
    print(f"SVM sweep took {time.time() - start:.0f}s, best {best_svm}", flush=True)
    start = time.time()
    size_rows = size_curve(parts, k, C, gamma)
    print(f"size curve took {time.time() - start:.0f}s", flush=True)
    start = time.time()
    linear_rows = linear_svm_rows(parts)
    print(f"linear SVM on the full fit fold took {time.time() - start:.0f}s", flush=True)
    return pd.DataFrame(knn_rows + svm_rows + size_rows + linear_rows)


def main(only_ext_sources=False):
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    RESULTS.parent.mkdir(exist_ok=True)
    if only_ext_sources:
        results = pd.DataFrame(ext_source_rows(X, y))
        results.to_csv(EXT_RESULTS, index=False, float_format="%.6f")
    else:
        results = run(prepare_folds(X, y, scale=True, clip=True))
        results.to_csv(RESULTS, index=False, float_format="%.6f")
    table = results.groupby(["experiment", "model", "setting", "n_fit"])[["pr_auc", "roc_auc", "log_loss", "fit_seconds", "predict_seconds"]].mean()
    print(table.round(4).to_string())


if __name__ == "__main__":
    main(only_ext_sources=sys.argv[1:] == ["ext"])
