import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from forest import prepare_folds
from kmeans import SEED, fit_kmeans
from naive_bayes import continuous_parts
from preprocess import load_features
from split import DB

LLOYD = Path("results/kmeans_lloyd.csv")
K = 5
STARTS = 5
MAX_ITERATIONS = 300


def squared_distances(Z, centers):
    return (Z**2).sum(axis=1)[:, None] - 2 * Z @ centers.T + (centers**2).sum(axis=1)[None, :]


def objective(Z, centers, labels):
    return float(((Z - centers[labels]) ** 2).sum())


def lloyd(Z, centers, max_iterations=MAX_ITERATIONS):
    centers = np.array(centers, dtype=float)
    labels = np.full(len(Z), -1)
    history = []
    for iteration in range(1, max_iterations + 1):
        new_labels = squared_distances(Z, centers).argmin(axis=1)
        after_assign = objective(Z, centers, new_labels)
        for c in range(len(centers)):
            if (new_labels == c).any():
                centers[c] = Z[new_labels == c].mean(axis=0)
        history.append({"iteration": iteration, "after_assign": after_assign, "after_update": objective(Z, centers, new_labels)})
        if (new_labels == labels).all():
            break
        labels = new_labels
    return centers, labels, history


def random_starts(Z, k=K, starts=STARTS, seed=SEED):
    rng = np.random.default_rng(seed)
    return [Z[rng.choice(len(Z), size=k, replace=False)] for _ in range(starts)]


def lloyd_rows(Z, k=K, starts=STARTS):
    rows = []
    for start, init in enumerate(random_starts(Z, k, starts)):
        centers, labels, history = lloyd(Z, init)
        reference = fit_kmeans_from(Z, init)
        for h in history:
            rows.append({"start": start, **h, "final": objective(Z, centers, labels), "sklearn_final": reference})
    return rows


def fit_kmeans_from(Z, init):
    km = KMeans(n_clusters=len(init), init=init, n_init=1, algorithm="lloyd", tol=0, max_iter=MAX_ITERATIONS).fit(Z)
    return float(km.inertia_)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = continuous_parts(prepare_folds(X, y, scale=True, clip=True))[0]
    LLOYD.parent.mkdir(exist_ok=True)
    table = pd.DataFrame(lloyd_rows(part["Z_fit"]))
    table.to_csv(LLOYD, index=False, float_format="%.8g")
    summary = table.groupby("start").agg(iterations=("iteration", "max"), first=("after_assign", "first"), final=("final", "first"), sklearn_final=("sklearn_final", "first"))
    print(summary.round(1).to_string())
    print("k-means++ default:", round(fit_kmeans(part["Z_fit"], K).inertia_, 1))


if __name__ == "__main__":
    main()
