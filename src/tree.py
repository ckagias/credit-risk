import sqlite3
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import evaluate
from baselines import THRESHOLD
from preprocess import load_features, make_preprocessor
from split import DB

CURVE = Path("results/tree_depth.csv")
SEED = 42
DEPTH_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, None]


def make_tree(max_depth=None, criterion="gini", min_samples_leaf=1):
    return DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, random_state=SEED
    )


def score_fold(fold, X_fit, y_fit, X_val, y_val, depths=DEPTH_GRID):
    pre = make_preprocessor(scale=False).fit(X_fit)
    Z_fit, Z_val = pre.transform(X_fit), pre.transform(X_val)
    rows = []
    for depth in depths:
        tree = make_tree(depth).fit(Z_fit, y_fit)
        for split, Z, y in (("train", Z_fit, y_fit), ("validation", Z_val, y_val)):
            scores = evaluate.score(y, tree.predict_proba(Z)[:, 1], THRESHOLD)
            rows.append(
                {
                    "fold": fold,
                    "max_depth": depth,
                    "depth": tree.get_depth(),
                    "leaves": tree.get_n_leaves(),
                    "split": split,
                    **scores,
                }
            )
    return rows


def depth_curve(X, y, depths=DEPTH_GRID, n_jobs=5):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    jobs = (
        delayed(score_fold)(k, X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx], depths)
        for k, (fit_idx, val_idx) in enumerate(folds)
    )
    return pd.DataFrame([row for rows in Parallel(n_jobs=n_jobs)(jobs) for row in rows])


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    start = time.time()
    curve = depth_curve(X, y)
    print(f"depth curve took {time.time() - start:.0f}s", flush=True)
    CURVE.parent.mkdir(exist_ok=True)
    curve.to_csv(CURVE, index=False, float_format="%.6f")
    table = curve.groupby(["max_depth", "split"], dropna=False)[["depth", "leaves", "pr_auc", "roc_auc", "log_loss"]].mean()
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
