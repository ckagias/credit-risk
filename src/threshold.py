import sqlite3
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import calibration
from forest import prepare_folds
from preprocess import load_features
from split import DB

CURVE = Path("results/threshold_curve.csv")
FOLDS = Path("results/threshold_folds.csv")
GRID = np.round(np.arange(0.005, 0.9951, 0.005), 3)
LOSS_GIVEN_DEFAULT = 0.45
MARGIN = 0.10


class Scheme(NamedTuple):
    name: str
    miss: float
    refuse: float
    by_size: bool

    @property
    def ratio(self):
        return self.miss / self.refuse


AMOUNT = Scheme("amount", LOSS_GIVEN_DEFAULT / MARGIN, 1.0, True)
SCHEMES = [AMOUNT, Scheme("flat_4.5", AMOUNT.ratio, 1.0, False)] + [Scheme(f"flat_{r}", float(r), 1.0, False) for r in (2, 5, 10, 20)]
MAIN_SCHEME = AMOUNT.name
SCORINGS = {
    "lightgbm": ("lightgbm", "none"),
    "lightgbm_weighted": ("lightgbm_weighted", "none"),
    "lightgbm_weighted_platt": ("lightgbm_weighted", "platt"),
}
FIXED = {"fixed_0.5": 0.5, "fixed_0.1": 0.1}


def outcomes(y, p, thresholds, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    totals, sorted_scores = [], []
    for label in (1, 0):
        order = np.argsort(p[y == label], kind="mergesort")
        sorted_scores.append(p[y == label][order])
        totals.append(np.concatenate([[0.0], np.cumsum(w[y == label][order])]))
    missed = totals[0][np.searchsorted(sorted_scores[0], thresholds, side="left")]
    refused_good = totals[1][-1] - totals[1][np.searchsorted(sorted_scores[1], thresholds, side="left")]
    return missed, refused_good


def sizes_for(scheme, size):
    return size if scheme.by_size else None


def cost_per_applicant(y, p, thresholds, scheme, size=None):
    missed, refused_good = outcomes(y, p, thresholds, sizes_for(scheme, size))
    return (scheme.miss * missed + scheme.refuse * refused_good) / len(y)


def bayes_threshold(scheme):
    return scheme.refuse / (scheme.refuse + scheme.miss)


def best_threshold(y, p, scheme, size=None, grid=GRID):
    costs = cost_per_applicant(y, p, grid, scheme, size)
    return float(grid[np.argmin(costs)])


def best_f1_threshold(y, p, grid=GRID):
    missed, refused_good = outcomes(y, p, grid)
    positives = int(np.sum(y))
    caught = positives - missed
    flagged = caught + refused_good
    f1 = np.where(flagged > 0, 2 * caught / np.maximum(flagged + positives, 1), 0.0)
    return float(grid[np.argmax(f1)])


def expected_cost_per_1000(p, threshold, scheme, size=None):
    p = np.asarray(p)
    weight = 1.0 if size is None or not scheme.by_size else np.asarray(size)
    return float(1000 * np.mean(weight * np.where(p >= threshold, scheme.refuse * (1 - p), scheme.miss * p)))


def rule_row(y, p, threshold, scheme, size=None):
    missed, refused_good = outcomes(y, p, np.array([threshold]))
    missed, refused_good = int(missed[0]), int(refused_good[0])
    positives = int(np.sum(y))
    return {
        "threshold": threshold,
        "cost_per_1000": 1000 * cost_per_applicant(y, p, np.array([threshold]), scheme, size)[0],
        "expected_cost_per_1000": expected_cost_per_1000(p, threshold, scheme, size),
        "share_refused": (positives - missed + refused_good) / len(y),
        "recall": (positives - missed) / positives,
    }


def fold_predictions_by_scoring(parts, sizes, scorings=SCORINGS):
    cache = {}
    out = {name: [] for name in scorings}
    for part in parts:
        for name, (model, calibrator) in scorings.items():
            if (part["fold"], model) not in cache:
                cache[(part["fold"], model)] = calibration.fold_predictions(part, model)
            out[name].append((part["y_val"], cache[(part["fold"], model)][calibrator], sizes[part["fold"]]))
        print(f"fold {part['fold']} done", flush=True)
    return out


def rule_rows(predictions, schemes=SCHEMES):
    rows = []
    for scoring, folds in predictions.items():
        for scheme in schemes:
            for k, (y, p, size) in enumerate(folds):
                others = [f for j, f in enumerate(folds) if j != k]
                y_other, p_other, size_other = (np.concatenate([f[i] for f in others]) for i in range(3))
                thresholds = {
                    "bayes": bayes_threshold(scheme),
                    "cv_tuned": best_threshold(y_other, p_other, scheme, size_other),
                    "best_f1": best_f1_threshold(y_other, p_other),
                    **FIXED,
                    "refuse_nobody": 1.0 + 1e-9,
                    "refuse_everybody": 0.0,
                }
                for rule, threshold in thresholds.items():
                    rows.append({"fold": k, "scoring": scoring, "scheme": scheme.name, "ratio": scheme.ratio, "rule": rule, **rule_row(y, p, threshold, scheme, size)})
    return pd.DataFrame(rows)


def curve_rows(predictions, schemes=SCHEMES, grid=GRID):
    rows = []
    for scoring, folds in predictions.items():
        y, p, size = (np.concatenate([f[i] for f in folds]) for i in range(3))
        for scheme in schemes:
            costs = cost_per_applicant(y, p, grid, scheme, size)
            rows.append(pd.DataFrame({"scoring": scoring, "scheme": scheme.name, "ratio": scheme.ratio, "threshold": grid, "cost_per_1000": 1000 * costs}))
    return pd.concat(rows, ignore_index=True)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = prepare_folds(X, y, scale=True, clip=True)
    amount = X["AMT_CREDIT"].to_numpy()
    sizes = [amount[part["val_idx"]] / amount.mean() for part in parts]
    predictions = fold_predictions_by_scoring(parts, sizes)
    FOLDS.parent.mkdir(exist_ok=True)
    rules = rule_rows(predictions)
    rules.to_csv(FOLDS, index=False, float_format="%.8g")
    curve = curve_rows(predictions)
    curve.to_csv(CURVE, index=False, float_format="%.8g")
    summary = rules.groupby(["scoring", "scheme", "rule"])[["threshold", "cost_per_1000", "share_refused", "recall"]].mean()
    print(summary.round(3).to_string())
    pooled = curve.loc[curve.groupby(["scoring", "scheme"])["cost_per_1000"].idxmin()]
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
