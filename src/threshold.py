import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import calibration
from forest import prepare_folds
from preprocess import load_features
from split import DB

CURVE = Path("results/threshold_curve.csv")
FOLDS = Path("results/threshold_folds.csv")
GRID = np.round(np.arange(0.005, 0.9951, 0.005), 3)
RATIOS = [2, 5, 10, 20]
MAIN_RATIO = 10
SCORINGS = {
    "lightgbm": ("lightgbm", "none"),
    "lightgbm_weighted": ("lightgbm_weighted", "none"),
    "lightgbm_weighted_platt": ("lightgbm_weighted", "platt"),
}
FIXED = {"fixed_0.5": 0.5, "fixed_0.1": 0.1}


def outcomes(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    sorted_defaults = np.sort(p[y == 1])
    sorted_good = np.sort(p[y == 0])
    missed = np.searchsorted(sorted_defaults, thresholds, side="left")
    refused_good = len(sorted_good) - np.searchsorted(sorted_good, thresholds, side="left")
    return missed, refused_good


def cost_per_applicant(y, p, thresholds, ratio):
    missed, refused_good = outcomes(y, p, thresholds)
    return (ratio * missed + refused_good) / len(y)


def bayes_threshold(ratio):
    return 1.0 / (1.0 + ratio)


def best_threshold(y, p, ratio, grid=GRID):
    costs = cost_per_applicant(y, p, grid, ratio)
    return float(grid[np.argmin(costs)])


def best_f1_threshold(y, p, grid=GRID):
    missed, refused_good = outcomes(y, p, grid)
    positives = int(np.sum(y))
    caught = positives - missed
    flagged = caught + refused_good
    f1 = np.where(flagged > 0, 2 * caught / np.maximum(flagged + positives, 1), 0.0)
    return float(grid[np.argmax(f1)])


def rule_row(y, p, threshold, ratio):
    missed, refused_good = outcomes(y, p, np.array([threshold]))
    missed, refused_good = int(missed[0]), int(refused_good[0])
    positives = int(np.sum(y))
    return {
        "threshold": threshold,
        "cost_per_1000": 1000 * (ratio * missed + refused_good) / len(y),
        "share_refused": (positives - missed + refused_good) / len(y),
        "recall": (positives - missed) / positives,
    }


def fold_predictions_by_scoring(parts, scorings=SCORINGS):
    cache = {}
    out = {name: [] for name in scorings}
    for part in parts:
        for name, (model, calibrator) in scorings.items():
            if (part["fold"], model) not in cache:
                cache[(part["fold"], model)] = calibration.fold_predictions(part, model)
            out[name].append((part["y_val"], cache[(part["fold"], model)][calibrator]))
        print(f"fold {part['fold']} done", flush=True)
    return out


def rule_rows(predictions, ratios=RATIOS):
    rows = []
    for scoring, folds in predictions.items():
        for ratio in ratios:
            for k, (y, p) in enumerate(folds):
                y_other = np.concatenate([f[0] for j, f in enumerate(folds) if j != k])
                p_other = np.concatenate([f[1] for j, f in enumerate(folds) if j != k])
                thresholds = {
                    "bayes": bayes_threshold(ratio),
                    "cv_tuned": best_threshold(y_other, p_other, ratio),
                    "best_f1": best_f1_threshold(y_other, p_other),
                    **FIXED,
                    "refuse_nobody": 1.0 + 1e-9,
                    "refuse_everybody": 0.0,
                }
                for rule, threshold in thresholds.items():
                    rows.append({"fold": k, "scoring": scoring, "ratio": ratio, "rule": rule, **rule_row(y, p, threshold, ratio)})
    return pd.DataFrame(rows)


def curve_rows(predictions, ratios=RATIOS, grid=GRID):
    rows = []
    for scoring, folds in predictions.items():
        y = np.concatenate([f[0] for f in folds])
        p = np.concatenate([f[1] for f in folds])
        for ratio in ratios:
            costs = cost_per_applicant(y, p, grid, ratio)
            rows.append(pd.DataFrame({"scoring": scoring, "ratio": ratio, "threshold": grid, "cost_per_1000": 1000 * costs}))
    return pd.concat(rows, ignore_index=True)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = prepare_folds(X, y, scale=True, clip=True)
    predictions = fold_predictions_by_scoring(parts)
    FOLDS.parent.mkdir(exist_ok=True)
    rules = rule_rows(predictions)
    rules.to_csv(FOLDS, index=False, float_format="%.8g")
    curve = curve_rows(predictions)
    curve.to_csv(CURVE, index=False, float_format="%.8g")
    summary = rules.groupby(["scoring", "ratio", "rule"])[["threshold", "cost_per_1000", "share_refused", "recall"]].mean()
    print(summary.round(3).to_string())
    pooled = curve.loc[curve.groupby(["scoring", "ratio"])["cost_per_1000"].idxmin()]
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
