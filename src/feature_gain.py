import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

import evaluate
import features
import logistic
import tuning
from baselines import THRESHOLD
from forest import prepare_folds
from preprocess import load_features
from split import DB

RESULTS = Path("results/feature_gain.csv")
SEED = 42
RATE = tuning.TUNE_RATE
LOGISTIC_C = 0.03
BASE = "base"
ALL = "all groups"
TUNED = "all groups, tuned"


def configs(groups=features.GROUPS):
    return [(BASE, [])] + [(group, [group]) for group in groups] + [(ALL, list(groups))]


def lightgbm_rows(X, y, params, rate=RATE):
    rows = []
    for part in prepare_folds(X, y, seed=SEED):
        rows.append({"fold": part["fold"], "n_features": part["Z_fit"].shape[1], **tuning.fold_scores(part, params, rate)})
    return pd.DataFrame(rows)


def logistic_rows(X, y):
    folds = evaluate.cross_validate(logistic.make_model(C=LOGISTIC_C, clip=True), X, y, n_splits=5, seed=SEED, threshold=THRESHOLD)
    return folds.assign(best_round=float("nan"))


def gain_table(results):
    keep = ["pr_auc", "roc_auc", "log_loss"]
    rows = []
    for model, part in results.groupby("model", sort=False):
        base = part[part["config"] == BASE].set_index("fold")[keep]
        for config, scored in part.groupby("config", sort=False):
            scored = scored.set_index("fold")
            difference = scored[keep] - base
            row = {"model": model, "config": config, "n_features": scored["n_features"].mean()}
            for metric in keep:
                row[f"{metric}_mean"] = scored[metric].mean()
                row[f"{metric}_std"] = scored[metric].std()
                row[f"{metric}_gain"] = difference[metric].mean()
                row[f"{metric}_gain_std"] = difference[metric].std()
            row["folds_better"] = int((difference["pr_auc"] > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def run(models, con):
    results = []
    for name in models:
        for config, groups in configs():
            X, y = load_features(con, "train", groups)
            start = time.time()
            if name == "lightgbm":
                scored = lightgbm_rows(X, y, tuning.DEFAULTS)
            else:
                scored = logistic_rows(X, y)
                scored["n_features"] = float("nan")
            results.append(scored.assign(model=name, config=config))
            print(f"{name} / {config}: PR-AUC {scored['pr_auc'].mean():.4f} in {time.time() - start:.0f}s", flush=True)
            pd.concat(results).to_csv(RESULTS, index=False, float_format="%.6f")
        if name == "lightgbm":
            X, y = load_features(con, "train", features.GROUPS)
            tuned = lightgbm_rows(X, y, tuning_best_params())
            results.append(tuned.assign(model=name, config=TUNED))
            print(f"lightgbm / {TUNED}: PR-AUC {tuned['pr_auc'].mean():.4f}", flush=True)
            pd.concat(results).to_csv(RESULTS, index=False, float_format="%.6f")
    return pd.concat(results, ignore_index=True)


def tuning_best_params():
    trials = pd.read_csv("results/optuna_trials.csv")
    best = trials[trials["state"] == "COMPLETE"].sort_values("value", ascending=False).iloc[0]
    return {name: (int(best[name]) if name in ("num_leaves", "min_child_samples") else float(best[name])) for name in tuning.DEFAULTS}


def main():
    models = sys.argv[1:] or ["lightgbm", "logistic"]
    con = sqlite3.connect(DB)
    results = run(models, con)
    con.close()
    table = gain_table(results)
    pd.set_option("display.width", 200)
    print(table[["model", "config", "n_features", "pr_auc_mean", "pr_auc_gain", "pr_auc_gain_std", "roc_auc_gain", "log_loss_gain", "folds_better"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
