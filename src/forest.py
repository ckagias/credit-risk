import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import evaluate
from baselines import THRESHOLD
from preprocess import load_features, make_preprocessor
from split import DB

RESULTS = Path("results/forest.csv")
SEED = 42
N_TREES = 100
TREE_COUNTS = [1, 2, 5, 10, 25, 50, 100, 200, 300, 500]
LEAF_GRID = [1, 3, 10, 30, 100, 300, 1000]
FEATURE_GRID = [1, 3, 10, 17, 30, 60, 100, 150, 300]
DEFAULT_FEATURES = 17


def make_forest(n_estimators=N_TREES, max_features=DEFAULT_FEATURES, min_samples_leaf=1, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=SEED,
    )


def prepare_folds(X, y, n_splits=5, seed=SEED, **preprocess):
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    prepared = []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        pre = make_preprocessor(**{"scale": False, **preprocess}).fit(X.iloc[fit_idx])
        prepared.append(
            {
                "fold": fold,
                "Z_fit": pre.transform(X.iloc[fit_idx]),
                "y_fit": y[fit_idx],
                "Z_val": pre.transform(X.iloc[val_idx]),
                "y_val": y[val_idx],
                "val_idx": val_idx,
                "features": list(pre.get_feature_names_out()),
            }
        )
    return prepared


def tree_probabilities(forest, Z):
    def one(tree):
        return tree.predict_proba(Z)[:, 1].astype(np.float32)

    return Parallel(n_jobs=-1, prefer="threads")(delayed(one)(tree) for tree in forest.estimators_)


def running_means(probabilities, counts):
    total = np.zeros(len(probabilities[0]))
    wanted = set(counts)
    means = {}
    for k, p in enumerate(probabilities, start=1):
        total += p
        if k in wanted:
            means[k] = total / k
    return means


def score_rows(part, experiment, setting, p_fit, p_val):
    rows = []
    for split, y, p in (("train", part["y_fit"], p_fit), ("validation", part["y_val"], p_val)):
        rows.append(
            {"fold": part["fold"], "experiment": experiment, "setting": setting, "split": split, **evaluate.score(y, p, THRESHOLD)}
        )
    return rows


def tree_count_curve(parts, counts=TREE_COUNTS, **forest_params):
    rows = []
    for part in parts:
        forest = make_forest(n_estimators=max(counts), **forest_params).fit(part["Z_fit"], part["y_fit"])
        fit_means = running_means(tree_probabilities(forest, part["Z_fit"]), counts)
        val_means = running_means(tree_probabilities(forest, part["Z_val"]), counts)
        for k in counts:
            rows += score_rows(part, "trees", k, fit_means[k], val_means[k])
    return rows


def setting_curve(parts, experiment, grid, n_estimators=N_TREES, **fixed):
    rows = []
    for part in parts:
        for value in grid:
            forest = make_forest(n_estimators=n_estimators, **{**fixed, experiment: value}).fit(part["Z_fit"], part["y_fit"])
            rows += score_rows(
                part, experiment, value, forest.predict_proba(part["Z_fit"])[:, 1], forest.predict_proba(part["Z_val"])[:, 1]
            )
    return rows


def best_setting(rows, experiment):
    table = pd.DataFrame(rows).query("experiment == @experiment and split == 'validation'")
    return table.groupby("setting")["pr_auc"].mean().idxmax()


def run(parts):
    rows = []
    start = time.time()
    leaf_rows = setting_curve(parts, "min_samples_leaf", LEAF_GRID)
    leaf = int(best_setting(leaf_rows, "min_samples_leaf"))
    print(f"leaf size sweep took {time.time() - start:.0f}s, best min_samples_leaf = {leaf}", flush=True)
    start = time.time()
    feature_rows = setting_curve(parts, "max_features", FEATURE_GRID, min_samples_leaf=leaf)
    features = int(best_setting(feature_rows, "max_features"))
    print(f"feature sweep took {time.time() - start:.0f}s, best max_features = {features}", flush=True)
    start = time.time()
    tree_rows = tree_count_curve(parts, max_features=features, min_samples_leaf=leaf)
    print(f"tree count sweep took {time.time() - start:.0f}s", flush=True)
    return pd.DataFrame(leaf_rows + feature_rows + tree_rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    results = run(prepare_folds(X, y))
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False, float_format="%.6f")
    table = results.groupby(["experiment", "setting", "split"])[["pr_auc", "roc_auc", "log_loss"]].mean()
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
