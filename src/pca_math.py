import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from forest import prepare_folds
from naive_bayes import continuous_parts
from pca import fit_pca
from preprocess import load_features
from split import DB

RECONSTRUCTION = Path("results/pca_reconstruction.csv")
DECILES = Path("results/pca_deciles.csv")
COMPONENTS = 3
BINS = 10


def eigen_decomposition(Z):
    values, vectors = np.linalg.eigh(np.cov(Z, rowvar=False))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def reconstruction_errors(Z_fit, Z_val):
    mean = Z_fit.mean(axis=0)
    values, vectors = eigen_decomposition(Z_fit)
    centered_fit = Z_fit - mean
    centered_val = Z_val - mean
    rows = []
    for k in range(len(values) + 1):
        V = vectors[:, :k]
        fit_error = ((centered_fit - centered_fit @ V @ V.T) ** 2).sum(axis=1).mean()
        val_error = ((centered_val - centered_val @ V @ V.T) ** 2).sum(axis=1).mean()
        rows.append(
            {
                "components": k,
                "fit_error": fit_error,
                "val_error": val_error,
                "tail_eigenvalues": values[k:].sum() * (len(Z_fit) - 1) / len(Z_fit),
            }
        )
    return rows


def decile_rows(Z_fit, Z_val, y_val, components=COMPONENTS, bins=BINS):
    pca = fit_pca(Z_fit)
    scores = pca.transform(Z_val)
    rows = []
    for c in range(components):
        edges = np.quantile(scores[:, c], np.linspace(0, 1, bins + 1))
        which = np.clip(np.searchsorted(edges, scores[:, c], side="right") - 1, 0, bins - 1)
        for b in range(bins):
            in_bin = which == b
            rows.append({"component": c + 1, "decile": b + 1, "n": int(in_bin.sum()), "default_rate": float(y_val[in_bin].mean())})
    return rows


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = continuous_parts(prepare_folds(X, y, scale=True, clip=True))[0]
    RECONSTRUCTION.parent.mkdir(exist_ok=True)
    reconstruction = pd.DataFrame(reconstruction_errors(part["Z_fit"], part["Z_val"]))
    reconstruction.to_csv(RECONSTRUCTION, index=False, float_format="%.8g")
    print(reconstruction.iloc[[0, 1, 2, 5, 10, 18, 30, 70]].round(3).to_string(index=False))
    deciles = pd.DataFrame(decile_rows(part["Z_fit"], part["Z_val"], part["y_val"]))
    deciles.to_csv(DECILES, index=False, float_format="%.6g")
    print(deciles.pivot(index="decile", columns="component", values="default_rate").round(4).to_string())


if __name__ == "__main__":
    main()
