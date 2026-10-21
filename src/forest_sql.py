import sqlite3
import time
from pathlib import Path

import pandas as pd
from sklearn.pipeline import make_pipeline

import evaluate
import features
from baselines import THRESHOLD
from forest import make_forest
from preprocess import load_features, make_preprocessor
from split import DB

RESULTS = Path("results/forest_sql.csv")
SEED = 42
TREES = 200
LEAF = 30
FEATURE_CHOICES = [30, 60]


def make_model(max_features, n_estimators=TREES, min_samples_leaf=LEAF):
    return make_pipeline(make_preprocessor(scale=False), make_forest(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf))


def best_max_features(table):
    return int(table.groupby("max_features")["pr_auc"].mean().idxmax())


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train", features.GROUPS)
    con.close()
    rows = []
    for max_features in FEATURE_CHOICES:
        start = time.time()
        folds = evaluate.cross_validate(make_model(max_features), X, y, n_splits=5, seed=SEED, threshold=THRESHOLD)
        rows.append(folds.assign(max_features=max_features, seconds=time.time() - start))
        print(f"max_features {max_features}: PR-AUC {folds['pr_auc'].mean():.4f} in {time.time() - start:.0f}s", flush=True)
    table = pd.concat(rows, ignore_index=True)
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.6f")
    print(table.groupby("max_features")[["pr_auc", "roc_auc", "log_loss"]].agg(["mean", "std"]).round(4).to_string())
    print("chosen max_features", best_max_features(table))


if __name__ == "__main__":
    main()
