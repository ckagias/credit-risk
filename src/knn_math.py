import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from forest import prepare_folds
from knn_svm import subsample
from naive_bayes import continuous_parts
from preprocess import load_features
from split import DB

DISTANCE = Path("results/knn_distance.csv")
SEED = 42
COLUMN_COUNTS = [1, 2, 5, 10, 20, 40, 70]
DRAWS = 5
FIT_ROWS = 20000
QUERY_ROWS = 300


def squared_distances(A, B):
    return (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None, :]


def knn_probability(Z_fit, y_fit, Z_query, k):
    nearest = np.argsort(squared_distances(Z_query, Z_fit), axis=1)[:, :k]
    return y_fit[nearest].mean(axis=1)


def distance_contrast(Z_fit, Z_query):
    distances = np.sqrt(np.maximum(squared_distances(Z_query, Z_fit), 0))
    nearest = distances.min(axis=1)
    median = np.median(distances, axis=1)
    usable = median > 0
    return float(nearest.mean()), float(median.mean()), float((nearest[usable] / median[usable]).mean())


def concentration_rows(part, column_counts=COLUMN_COUNTS, draws=DRAWS, fit_rows=FIT_ROWS, query_rows=QUERY_ROWS):
    rng = np.random.default_rng(SEED)
    Z_fit, _ = subsample(part, fit_rows)
    Z_query = part["Z_val"][rng.choice(len(part["Z_val"]), size=query_rows, replace=False)]
    rows = []
    for d in column_counts:
        for draw in range(draws if d < Z_fit.shape[1] else 1):
            columns = rng.choice(Z_fit.shape[1], size=d, replace=False)
            nearest, median, ratio = distance_contrast(Z_fit[:, columns], Z_query[:, columns])
            rows.append({"columns": d, "draw": draw, "nearest": nearest, "median": median, "ratio": ratio})
    return rows


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = continuous_parts(prepare_folds(X, y, scale=True, clip=True))[0]
    table = pd.DataFrame(concentration_rows(part))
    DISTANCE.parent.mkdir(exist_ok=True)
    table.to_csv(DISTANCE, index=False, float_format="%.6g")
    print(table.groupby("columns")[["nearest", "median", "ratio"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
