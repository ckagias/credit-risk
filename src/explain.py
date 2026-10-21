import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

import boosting
import evaluate
import features
import tuning
from baselines import THRESHOLD
from feature_gain import tuning_best_params
from forest import prepare_folds
from preprocess import load_features
from split import DB

PERMUTATION = Path("results/permutation_importance.csv")
SHAP_IMPORTANCE = Path("results/shap_importance.csv")
SHAP_SAMPLE = Path("results/shap_sample.csv")
APPLICANTS = Path("results/shap_applicants.csv")
SEED = 42
FOLD = 0
REPEATS = 5
SHAP_ROWS = 5000
PLOT_ROWS = 2000
PLOT_FEATURES = 15
APPLICANT_FEATURES = 8
OVERLAP = 20


def fit_best(part, params=None):
    params = tuning.lightgbm_params(params or tuning_best_params())
    return boosting.fit_with_early_stopping(part["Z_fit"], part["y_fit"], tuning.TUNE_RATE, **params)


def permutation_table(model, Z, y, names, repeats=REPEATS, seed=SEED):
    result = permutation_importance(model, Z, y, scoring="average_precision", n_repeats=repeats, random_state=seed, n_jobs=1)
    table = pd.DataFrame({"feature": names, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
    return table.sort_values("importance_mean", ascending=False, ignore_index=True)


def shap_values(model, Z):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Z)
    values = values[1] if isinstance(values, list) else values
    return values, float(np.ravel(explainer.expected_value)[-1])


def shap_table(values, names):
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs, "share": mean_abs / mean_abs.sum()})
    return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def pick_applicants(y, p):
    order = np.argsort(p, kind="mergesort")
    defaulters, repaid = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return {
        "highest risk score": int(order[-1]),
        "lowest risk score": int(order[0]),
        "defaulter scored lowest": int(defaulters[np.argmin(p[defaulters])]),
        "repaid, scored highest": int(repaid[np.argmax(p[repaid])]),
    }


def applicant_rows(model, Z, y, p, names, top=APPLICANT_FEATURES):
    rows = []
    for label, index in pick_applicants(y, p).items():
        values, base = shap_values(model, Z[index : index + 1])
        contribution = values[0]
        for rank, feature in enumerate(np.argsort(-np.abs(contribution))[:top], start=1):
            rows.append(
                {
                    "applicant": label,
                    "row": index,
                    "default": int(y[index]),
                    "probability": float(p[index]),
                    "base_log_odds": base,
                    "log_odds": base + float(contribution.sum()),
                    "rank": rank,
                    "feature": names[feature],
                    "value": float(Z[index, feature]),
                    "shap": float(contribution[feature]),
                }
            )
    return pd.DataFrame(rows)


def sample_rows(values, Z, names, top_features, rows=PLOT_ROWS):
    keep = np.random.default_rng(SEED).choice(len(values), size=min(rows, len(values)), replace=False)
    columns = [names.index(f) for f in top_features]
    return pd.concat(
        [pd.DataFrame({"feature": names[c], "row": keep, "value": Z[keep, c], "shap": values[keep, c]}) for c in columns],
        ignore_index=True,
    )


def agreement(permutation, shapley, top=OVERLAP):
    merged = permutation.merge(shapley, on="feature")
    return {
        "spearman": float(spearmanr(merged["importance_mean"], merged["mean_abs_shap"])[0]),
        "top_overlap": len(set(permutation["feature"][:top]) & set(shapley["feature"][:top])),
    }


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train", features.GROUPS)
    con.close()
    part = prepare_folds(X, y, seed=SEED)[FOLD]
    names = part["features"]
    model = fit_best(part)
    Z, y_val = part["Z_val"], part["y_val"]
    p = model.predict_proba(Z)[:, 1]
    print(f"fold {FOLD}: {len(names)} features, {model.best_iteration_} rounds, {evaluate.score(y_val, p, THRESHOLD)}", flush=True)

    start = time.time()
    permutation = permutation_table(model, Z, y_val, names)
    PERMUTATION.parent.mkdir(exist_ok=True)
    permutation.to_csv(PERMUTATION, index=False, float_format="%.6g")
    print(f"permutation importance took {time.time() - start:.0f}s", flush=True)

    keep = np.random.default_rng(SEED).choice(len(y_val), size=SHAP_ROWS, replace=False)
    values, base = shap_values(model, Z[keep])
    shapley = shap_table(values, names)
    shapley.to_csv(SHAP_IMPORTANCE, index=False, float_format="%.6g")
    sample_rows(values, Z[keep], names, list(shapley["feature"][:PLOT_FEATURES])).to_csv(SHAP_SAMPLE, index=False, float_format="%.5g")
    applicant_rows(model, Z, y_val, p, names).to_csv(APPLICANTS, index=False, float_format="%.6g")

    print(f"base log-odds {base:.3f}, probability {1 / (1 + np.exp(-base)):.4f}")
    print(permutation.head(15).round(5).to_string())
    print(shapley.head(15).round(4).to_string())
    print(agreement(permutation, shapley))


if __name__ == "__main__":
    main()
