import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

EPS = 1e-15
METRICS = ["pr_auc", "roc_auc", "log_loss", "precision", "recall"]


def check_inputs(y, p):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y and p must be one-dimensional and the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p must be probabilities between 0 and 1")
    return y.astype(int), p


def counts_by_threshold(y, p):
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]
    tp = np.cumsum(y_sorted)[last_of_tie]
    fp = (last_of_tie + 1) - tp
    return tp, fp


def positives_and_negatives(y):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y needs both classes")
    return n_pos, n_neg


def pr_curve(y, p):
    y, p = check_inputs(y, p)
    n_pos, _ = positives_and_negatives(y)
    tp, fp = counts_by_threshold(y, p)
    return tp / n_pos, tp / (tp + fp)


def roc_curve(y, p):
    y, p = check_inputs(y, p)
    n_pos, n_neg = positives_and_negatives(y)
    tp, fp = counts_by_threshold(y, p)
    return np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos]


def pr_auc(y, p):
    recall, precision = pr_curve(y, p)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, p):
    fpr, tpr = roc_curve(y, p)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def log_loss(y, p):
    y, p = check_inputs(y, p)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def precision_recall_at(y, p, threshold):
    y, p = check_inputs(y, p)
    flagged = p >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    n_flagged = int(flagged.sum())
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / int(y.sum()) if y.sum() else 0.0
    return precision, recall


def score(y, p, threshold=0.5):
    precision, recall = precision_recall_at(y, p, threshold)
    return {
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "log_loss": log_loss(y, p),
        "precision": precision,
        "recall": recall,
    }


def cross_validate(estimator, X, y, n_splits=5, seed=42, threshold=0.5):
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(X, y)):
        model = clone(estimator)
        model.fit(X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx], y[fit_idx])
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        rows.append({"fold": fold, **score(y[val_idx], model.predict_proba(X_val)[:, 1], threshold)})
    return pd.DataFrame(rows)


def out_of_fold_predictions(estimator, X, y, n_splits=5, seed=42):
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.empty(len(y))
    for fit_idx, val_idx in folds.split(X, y):
        model = clone(estimator)
        model.fit(X.iloc[fit_idx] if hasattr(X, "iloc") else X[fit_idx], y[fit_idx])
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    return predictions


def summarize_folds(folds):
    table = folds[METRICS].agg(["mean", "std"]).T
    table.columns = ["mean", "std"]
    return table


def fold_table(folds, model):
    table = folds[METRICS].agg(["mean", "std", "min", "max"]).T
    table.index.name = "metric"
    return table.reset_index().assign(model=model)[["model", "metric", "mean", "std", "min", "max"]]


def bootstrap(y, p, n_boot=1000, seed=42, alpha=0.05, threshold=0.5):
    y, p = check_inputs(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    while len(draws) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        draws.append(score(y[idx], p[idx], threshold))
    draws = pd.DataFrame(draws)
    estimate = score(y, p, threshold)
    return pd.DataFrame(
        {
            "estimate": pd.Series(estimate),
            "low": draws.quantile(alpha / 2),
            "high": draws.quantile(1 - alpha / 2),
        }
    ).loc[METRICS]
