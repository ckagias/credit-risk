import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import evaluate
import imbalance
from baselines import THRESHOLD
from forest import make_forest, prepare_folds
from knn_svm import make_logistic
from preprocess import load_features
from split import DB

RESULTS = Path("results/calibration.csv")
CURVE = Path("results/calibration_curve.csv")
SEED = 42
CALIBRATION_SHARE = 0.2
BINS = 10
MODELS = ["logistic", "lightgbm", "lightgbm_weighted", "forest"]
CALIBRATORS = ["none", "platt", "isotonic"]
FOREST_TREES = 200
FOREST_FEATURES = 30
FOREST_LEAF = 30


def logit(p):
    p = np.clip(p, evaluate.EPS, 1 - evaluate.EPS)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.model_.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return self.model_.predict(p)


def make_calibrator(name):
    return {"platt": Platt, "isotonic": Isotonic}[name]()


def brier(y, p):
    y, p = evaluate.check_inputs(y, p)
    return float(np.mean((p - y) ** 2))


def reliability(y, p, bins=BINS):
    y, p = evaluate.check_inputs(y, p)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for index in np.array_split(order, bins):
        rows.append({"mean_predicted": float(p[index].mean()), "observed": float(y[index].mean()), "n": len(index)})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, bins=BINS):
    table = reliability(y, p, bins)
    return float(np.sum(table["n"] * (table["mean_predicted"] - table["observed"]).abs()) / table["n"].sum())


def fit_model(name, Z, y):
    if name == "logistic":
        return make_logistic().fit(Z, y)
    if name == "lightgbm":
        return imbalance.fit_lightgbm(Z, y, "none")
    if name == "lightgbm_weighted":
        return imbalance.fit_lightgbm(Z, y, "class_weight")
    return make_forest(n_estimators=FOREST_TREES, max_features=FOREST_FEATURES, min_samples_leaf=FOREST_LEAF).fit(Z, y)


def calibration_split(part, share=CALIBRATION_SHARE, seed=SEED):
    Z_model, Z_cal, y_model, y_cal = train_test_split(part["Z_fit"], part["y_fit"], test_size=share, stratify=part["y_fit"], random_state=seed)
    return Z_model, Z_cal, y_model, y_cal


def fold_predictions(part, model_name):
    Z_model, Z_cal, y_model, y_cal = calibration_split(part)
    model = fit_model(model_name, Z_model, y_model)
    p_cal = np.clip(model.predict_proba(Z_cal)[:, 1], evaluate.EPS, 1 - evaluate.EPS)
    p_val = np.clip(model.predict_proba(part["Z_val"])[:, 1], evaluate.EPS, 1 - evaluate.EPS)
    out = {"none": p_val}
    for name in CALIBRATORS[1:]:
        out[name] = make_calibrator(name).fit(p_cal, y_cal).predict(p_val)
    return out


def run(parts, models=MODELS):
    rows, curves = [], []
    for part in parts:
        for model_name in models:
            start = time.time()
            predictions = fold_predictions(part, model_name)
            seconds = time.time() - start
            for calibrator, p in predictions.items():
                scored = evaluate.score(part["y_val"], p, THRESHOLD)
                rows.append(
                    {
                        "fold": part["fold"],
                        "model": model_name,
                        "calibrator": calibrator,
                        **{key: scored[key] for key in ("pr_auc", "roc_auc", "log_loss")},
                        "brier": brier(part["y_val"], p),
                        "ece": expected_calibration_error(part["y_val"], p),
                        "mean_probability": float(p.mean()),
                        "seconds": seconds,
                    }
                )
                curves.append(pd.DataFrame({"fold": part["fold"], "model": model_name, "calibrator": calibrator, "y": part["y_val"], "p": p}))
        print(f"fold {part['fold']} done", flush=True)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def pooled_curves(predictions, bins=BINS):
    rows = []
    for (model, calibrator), part in predictions.groupby(["model", "calibrator"]):
        table = reliability(part["y"].to_numpy(), part["p"].to_numpy(), bins)
        rows.append(table.assign(model=model, calibrator=calibrator, bin=range(1, len(table) + 1)))
    return pd.concat(rows, ignore_index=True)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    parts = prepare_folds(X, y, scale=True, clip=True)
    table, predictions = run(parts)
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.8g")
    pooled_curves(predictions).to_csv(CURVE, index=False, float_format="%.8g")
    columns = ["pr_auc", "roc_auc", "log_loss", "brier", "ece", "mean_probability"]
    print(table.groupby(["model", "calibrator"])[columns].mean().round(4).to_string())


if __name__ == "__main__":
    main()
