import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import boosting
import calibration
import evaluate
import imbalance
import tuning
from forest import prepare_folds
from knn_svm import make_logistic
from preprocess import load_features
from split import DB

SHIFT = Path("results/weighted_shift.csv")
SAMPLE = Path("results/weighted_shift_sample.csv")
GAP = Path("results/weighted_gap.csv")
GAP_ROUNDS = 300
GAP_LEAVES = [4, 8, 31]
SEED = 42
SAMPLE_ROWS = 5000


def weighted_log_loss(y, p, positive_weight):
    y, p = evaluate.check_inputs(y, p)
    p = np.clip(p, evaluate.EPS, 1 - evaluate.EPS)
    weights = np.where(y == 1, positive_weight, 1.0)
    return float(-np.mean(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def shifted_probability(p, positive_weight):
    odds = p / (1 - p) * positive_weight
    return odds / (1 + odds)


def weighted_optimum(p, positive_weight):
    return positive_weight * p / (positive_weight * p + 1 - p)


def shift_row(part):
    Z_model, _, y_model, _ = calibration.calibration_split(part)
    w = float((len(y_model) - y_model.sum()) / y_model.sum())
    plain = calibration.fold_predictions(part, "lightgbm")["none"]
    predictions, calibrators = calibration.fold_predictions(part, "lightgbm_weighted", with_calibrators=True)
    weighted = predictions["none"]
    platt = calibrators["platt"].model_
    x, z = calibration.logit(plain), calibration.logit(weighted)
    slope, intercept = np.polyfit(x, z, 1)
    row = {
        "fold": part["fold"],
        "weight": w,
        "log_weight": np.log(w),
        "slope": slope,
        "intercept": intercept,
        "platt_slope": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "median_log_odds_shift": float(np.median(z - x)),
        "correlation": float(np.corrcoef(x, z)[0, 1]),
        "mean_probability_weighted": float(weighted.mean()),
        "mean_probability_formula": float(shifted_probability(plain, w).mean()),
        "log_loss_weighted": evaluate.log_loss(part["y_val"], weighted),
        "log_loss_formula": evaluate.log_loss(part["y_val"], shifted_probability(plain, w)),
        "log_loss_plain": evaluate.log_loss(part["y_val"], plain),
        "weighted_loss_plain": weighted_log_loss(part["y_val"], plain, w),
        "weighted_loss_weighted": weighted_log_loss(part["y_val"], weighted, w),
    }
    keep = np.random.default_rng(SEED).choice(len(x), size=SAMPLE_ROWS, replace=False)
    sample = pd.DataFrame({"fold": part["fold"], "logit_plain": x[keep], "logit_weighted": z[keep], "y": part["y_val"][keep]})
    return row, sample


def shift_decomposition(x, z):
    slope, intercept = np.polyfit(x, z, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_shift": float(np.mean(z - x)),
        "mean_log_odds": float(np.mean(x)),
        "stretch_part": float((slope - 1) * np.mean(x)),
    }


def lightgbm_at_rounds(part, weighted, num_leaves, positive_weight):
    Z_model, _, y_model, _ = calibration.calibration_split(part)
    Z_fit, _, y_fit, _ = boosting.stopping_split(Z_model, y_model)
    params = {**tuning.lightgbm_params({**tuning.DEFAULTS, "num_leaves": num_leaves}), "metric": "None"}
    if weighted:
        params["scale_pos_weight"] = positive_weight
    model = boosting.make_booster(imbalance.LEARNING_RATE, n_estimators=GAP_ROUNDS, **params).fit(Z_fit, y_fit)
    return calibration.logit(np.clip(model.predict_proba(part["Z_val"])[:, 1], evaluate.EPS, 1 - evaluate.EPS))


def gap_rows(part):
    Z_model, _, y_model, _ = calibration.calibration_split(part)
    w = float((len(y_model) - y_model.sum()) / y_model.sum())
    rows = []
    plain = calibration.fit_model("logistic", Z_model, y_model)
    weighted = make_logistic().set_params(class_weight="balanced").fit(Z_model, y_model)
    to_logit = lambda model: calibration.logit(np.clip(model.predict_proba(part["Z_val"])[:, 1], evaluate.EPS, 1 - evaluate.EPS))
    rows.append({"model": "logistic", "num_leaves": 0, "log_weight": np.log(w), **shift_decomposition(to_logit(plain), to_logit(weighted))})
    for leaves in GAP_LEAVES:
        x = lightgbm_at_rounds(part, False, leaves, w)
        z = lightgbm_at_rounds(part, True, leaves, w)
        rows.append({"model": "lightgbm", "num_leaves": leaves, "log_weight": np.log(w), **shift_decomposition(x, z)})
    return pd.DataFrame(rows)


def gap_main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = prepare_folds(X, y, scale=True, clip=True)[0]
    table = gap_rows(part)
    table.to_csv(GAP, index=False, float_format="%.6g")
    print(table.round(3).to_string())


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = prepare_folds(X, y, scale=True, clip=True)
    rows, samples = [], []
    for part in parts:
        row, sample = shift_row(part)
        rows.append(row)
        if part["fold"] == 0:
            samples.append(sample)
        print(f"fold {part['fold']} done", flush=True)
    SHIFT.parent.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(SHIFT, index=False, float_format="%.8g")
    pd.concat(samples).to_csv(SAMPLE, index=False, float_format="%.5g")
    print(table.round(4).T.to_string())


if __name__ == "__main__":
    import sys

    gap_main() if "gap" in sys.argv[1:] else main()
