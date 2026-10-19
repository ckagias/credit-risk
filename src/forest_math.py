import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import forest
from preprocess import load_features
from split import DB

VARIANCE = Path("results/forest_variance.csv")
SEED = 42
DATASETS = 8
TREES = 30
SUBSAMPLE = 100_000
LEAF = 30
FEATURES = [10, 30, 300]
GROUP_SIZES = [1, 2, 3, 5, 6, 10, 15, 30]


def group_variances(P, sizes):
    datasets, trees, rows = P.shape
    variances = {}
    for b in sizes:
        if trees % b:
            raise ValueError("the group size must divide the number of trees")
        means = P.reshape(datasets, trees // b, b, rows).mean(axis=2).reshape(-1, rows)
        variances[b] = float(means.var(axis=0, ddof=1).mean())
    return variances


def fit_correlation(variances):
    inverse = 1.0 / np.array(list(variances))
    slope, intercept = np.polyfit(inverse, np.array(list(variances.values())), 1)
    sigma2 = intercept + slope
    return float(intercept / sigma2), float(sigma2)


def predicted_variance(b, rho, sigma2):
    return rho * sigma2 + (1.0 - rho) * sigma2 / b


def forests_on_subsamples(part, max_features, datasets=DATASETS, trees=TREES, size=SUBSAMPLE, leaf=LEAF):
    rng = np.random.default_rng(SEED)
    P = np.empty((datasets, trees, len(part["Z_val"])), dtype=np.float32)
    for k in range(datasets):
        rows = rng.choice(len(part["y_fit"]), size=size, replace=False)
        model = forest.make_forest(n_estimators=trees, max_features=max_features, min_samples_leaf=leaf)
        model.set_params(random_state=SEED + k)
        model.fit(part["Z_fit"][rows], part["y_fit"][rows])
        P[k] = np.stack(forest.tree_probabilities(model, part["Z_val"]))
    return P


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = forest.prepare_folds(X, y)[0]
    rows = []
    for max_features in FEATURES:
        start = time.time()
        P = forests_on_subsamples(part, max_features)
        variances = group_variances(P, GROUP_SIZES)
        rho, sigma2 = fit_correlation(variances)
        print(f"max_features {max_features}: sigma2 {sigma2:.5f}, rho {rho:.3f}, floor {rho * sigma2:.5f} ({time.time() - start:.0f}s)", flush=True)
        for b, variance in variances.items():
            rows.append(
                {
                    "max_features": max_features,
                    "trees": b,
                    "variance": variance,
                    "formula": predicted_variance(b, rho, sigma2),
                    "rho": rho,
                    "sigma2": sigma2,
                }
            )
    table = pd.DataFrame(rows)
    VARIANCE.parent.mkdir(exist_ok=True)
    table.to_csv(VARIANCE, index=False, float_format="%.8f")
    print(table.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
