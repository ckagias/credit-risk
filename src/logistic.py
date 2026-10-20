import sqlite3
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import evaluate
from baselines import THRESHOLD
from preprocess import load_features, make_preprocessor
from split import DB

RESULTS = Path("results/logistic.csv")
SEED = 42


def make_model(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000, clip=False):
    return make_pipeline(
        make_preprocessor(scale=True, clip=clip),
        LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=max_iter),
    )


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    start = time.time()
    folds = evaluate.cross_validate(make_model(), X, y, n_splits=5, seed=SEED, threshold=THRESHOLD)
    print(f"5-fold CV took {time.time() - start:.0f}s")
    table = evaluate.fold_table(folds, "logistic regression, L2, C=1")
    print(table.drop(columns="model").round(4).to_string(index=False))
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.6f")
    baseline = pd.read_csv("results/baselines.csv")
    floor = baseline[baseline["model"].str.startswith("EXT_SOURCE")].set_index("metric")["mean"]
    print("\nagainst the EXT_SOURCE baseline (mean over folds):")
    for metric in ("pr_auc", "roc_auc", "log_loss"):
        mine = table.set_index("metric").loc[metric, "mean"]
        print(f"  {metric:9s} {floor[metric]:.4f} -> {mine:.4f}")


if __name__ == "__main__":
    main()
