import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import evaluate
import tuning
from baselines import THRESHOLD
from forest import prepare_folds
from knn_svm import make_logistic
from naive_bayes import continuous_mask, continuous_parts, make_bayes
from preprocess import load_features, make_preprocessor
from split import DB

VARIANCE = Path("results/pca_variance.csv")
MODELS = Path("results/pca_models.csv")
LOADINGS = Path("results/pca_loadings.csv")
SEED = 42
COMPONENTS = [2, 5, 10, 20, 30, 50]
VARIANCE_TARGETS = [0.5, 0.8, 0.9, 0.95, 0.99]
LOADING_COMPONENTS = 3
LOADINGS_PER_COMPONENT = 6


def fit_pca(Z):
    return PCA(svd_solver="full", random_state=SEED).fit(Z)


def variance_rows(parts):
    rows = []
    for part in parts:
        ratio = fit_pca(part["Z_fit"]).explained_variance_ratio_
        for i, (r, c) in enumerate(zip(ratio, np.cumsum(ratio)), start=1):
            rows.append({"fold": part["fold"], "component": i, "explained_ratio": r, "cumulative": c})
    return rows


def components_needed(variance, targets=VARIANCE_TARGETS):
    mean = pd.DataFrame(variance).groupby("component")["cumulative"].mean()
    return {target: int((mean < target).sum() + 1) for target in targets}


def scored(part, representation, k, model_name, Z_fit, Z_val):
    start = time.time()
    if model_name == "lightgbm":
        result = tuning.fold_scores({**part, "Z_fit": Z_fit, "Z_val": Z_val}, tuning.DEFAULTS)
        result = {key: value for key, value in result.items() if key != "best_round"}
        fit_seconds = time.time() - start
    else:
        model = make_logistic() if model_name == "logistic" else make_bayes()
        model.fit(Z_fit, part["y_fit"])
        fit_seconds = time.time() - start
        result = evaluate.score(part["y_val"], model.predict_proba(Z_val)[:, 1], THRESHOLD)
    return {
        "fold": part["fold"],
        "model": model_name,
        "representation": representation,
        "n_columns": k,
        **{key: result[key] for key in ("pr_auc", "roc_auc", "log_loss")},
        "fit_seconds": fit_seconds,
    }


def model_rows(parts, components=COMPONENTS, models=("logistic", "lightgbm", "naive_bayes")):
    rows = []
    for part in parts:
        pca = fit_pca(part["Z_fit"])
        scores_fit = pca.transform(part["Z_fit"])
        scores_val = pca.transform(part["Z_val"])
        for model_name in models:
            n = part["Z_fit"].shape[1]
            rows.append(scored(part, "raw", n, model_name, part["Z_fit"], part["Z_val"]))
            for k in [*components, n]:
                rows.append(scored(part, "pca", k, model_name, scores_fit[:, :k], scores_val[:, :k]))
    return rows


def loadings(X, y):
    Z = make_preprocessor(scale=True, clip=True).fit_transform(X)
    numeric = np.array(X.select_dtypes("number").columns)
    mask = continuous_mask(Z[:, : len(numeric)])
    if mask.sum() != continuous_mask(Z).sum():
        raise ValueError("the continuous columns are not all in the numeric block of the preprocessed table")
    names = numeric[mask]
    pca = fit_pca(Z[:, : len(numeric)][:, mask])
    rows = []
    for c in range(LOADING_COMPONENTS):
        order = np.argsort(-np.abs(pca.components_[c]))[:LOADINGS_PER_COMPONENT]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "component": c + 1,
                    "rank": rank,
                    "column": names[j],
                    "loading": pca.components_[c][j],
                    "explained_ratio": pca.explained_variance_ratio_[c],
                }
            )
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = continuous_parts(prepare_folds(X, y, scale=True, clip=True))
    VARIANCE.parent.mkdir(exist_ok=True)
    variance = variance_rows(parts)
    pd.DataFrame(variance).to_csv(VARIANCE, index=False, float_format="%.8g")
    print("components for cumulative variance:", components_needed(variance), flush=True)
    models = pd.DataFrame(model_rows(parts))
    models.to_csv(MODELS, index=False, float_format="%.8g")
    table = models.groupby(["model", "representation", "n_columns"])[["pr_auc", "roc_auc", "log_loss", "fit_seconds"]].mean()
    print(table.round(4).to_string())
    loadings(X, y).to_csv(LOADINGS, index=False, float_format="%.6g")
    print(pd.read_csv(LOADINGS).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
