import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import evaluate
from baselines import THRESHOLD
from forest import prepare_folds
from naive_bayes import continuous_mask, continuous_parts
from preprocess import load_features, make_preprocessor
from split import DB

CLUSTERS = Path("results/kmeans_clusters.csv")
SUMMARY = Path("results/kmeans_summary.csv")
PROFILE = Path("results/kmeans_profile.csv")
SEED = 42
K_GRID = [2, 3, 5, 8, 12, 20]
N_INIT = 3
SILHOUETTE_SAMPLE = 10000
PROFILE_K = 5
PROFILE_COLUMNS = 4


def fit_kmeans(Z, k):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(Z)


def silhouette(Z, labels, seed=SEED):
    return float(silhouette_score(Z, labels, sample_size=min(SILHOUETTE_SAMPLE, len(Z)), random_state=seed))


def cluster_table(labels_fit, y_fit, labels_val, y_val, k):
    rows = []
    for c in range(k):
        in_fit = labels_fit == c
        in_val = labels_val == c
        rows.append(
            {
                "cluster": c,
                "n_fit": int(in_fit.sum()),
                "rate_fit": float(y_fit[in_fit].mean()) if in_fit.any() else np.nan,
                "n_val": int(in_val.sum()),
                "rate_val": float(y_val[in_val].mean()) if in_val.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def chi_square_p(labels, y, k):
    counts = np.array([[np.sum((labels == c) & (y == 0)), np.sum((labels == c) & (y == 1))] for c in range(k)])
    counts = counts[counts.sum(axis=1) > 0]
    return float(chi2_contingency(counts)[1])


def fold_rows(part, k):
    start = time.time()
    km = fit_kmeans(part["Z_fit"], k)
    fit_seconds = time.time() - start
    labels_fit = km.labels_
    labels_val = km.predict(part["Z_val"])
    table = cluster_table(labels_fit, part["y_fit"], labels_val, part["y_val"], k)
    rate_by_cluster = table["rate_fit"].fillna(part["y_fit"].mean()).to_numpy()
    scored = evaluate.score(part["y_val"], rate_by_cluster[labels_val], THRESHOLD)
    summary = {
        "fold": part["fold"],
        "k": k,
        "inertia_per_row": km.inertia_ / len(part["Z_fit"]),
        "silhouette": silhouette(part["Z_fit"], labels_fit),
        "rate_min_val": table["rate_val"].min(),
        "rate_max_val": table["rate_val"].max(),
        "smallest_cluster_share": table["n_fit"].min() / len(part["Z_fit"]),
        "chi_square_p_val": chi_square_p(labels_val, part["y_val"], k),
        "pr_auc": scored["pr_auc"],
        "roc_auc": scored["roc_auc"],
        "fit_seconds": fit_seconds,
    }
    return summary, table.assign(fold=part["fold"], k=k)


def sweep(parts, grid=K_GRID):
    summaries, tables = [], []
    for part in parts:
        for k in grid:
            summary, table = fold_rows(part, k)
            summaries.append(summary)
            tables.append(table)
        print(f"fold {part['fold']} done", flush=True)
    return pd.DataFrame(summaries), pd.concat(tables, ignore_index=True)


def centroid_profile(X, y, k=PROFILE_K, per_cluster=PROFILE_COLUMNS):
    Z = make_preprocessor(scale=True, clip=True).fit_transform(X)
    numeric = np.array(X.select_dtypes("number").columns)
    mask = continuous_mask(Z[:, : len(numeric)])
    names = numeric[mask]
    Zc = Z[:, : len(numeric)][:, mask]
    km = fit_kmeans(Zc, k)
    rows = []
    for c in range(k):
        in_cluster = km.labels_ == c
        order = np.argsort(-np.abs(km.cluster_centers_[c]))[:per_cluster]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "cluster": c,
                    "size": int(in_cluster.sum()),
                    "default_rate": float(y[in_cluster].mean()),
                    "rank": rank,
                    "column": names[j],
                    "centroid": km.cluster_centers_[c][j],
                }
            )
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = continuous_parts(prepare_folds(X, y, scale=True, clip=True))
    CLUSTERS.parent.mkdir(exist_ok=True)
    summary, clusters = sweep(parts)
    summary.to_csv(SUMMARY, index=False, float_format="%.8g")
    clusters.to_csv(CLUSTERS, index=False, float_format="%.8g")
    print(summary.drop(columns="fold").groupby("k").mean().round(4).to_string())
    centroid_profile(X, y).to_csv(PROFILE, index=False, float_format="%.6g")
    print(pd.read_csv(PROFILE).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
