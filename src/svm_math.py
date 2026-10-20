import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

import evaluate
from baselines import THRESHOLD
from forest import prepare_folds
from knn_svm import SVM_SWEEP_SIZE, subsample
from preprocess import load_features
from split import DB

SUPPORT = Path("results/svm_support.csv")
MARGINS = Path("results/svm_margins.csv")
SEED = 42
C_GRID = [0.1, 1, 10]
GAMMA_MULTIPLIER = 0.1
MARGIN_C = 1
MARGIN_ROWS = 5000


def signed(y):
    return 2 * np.asarray(y) - 1


def hinge(y, f, weights=None):
    loss = np.maximum(0.0, 1.0 - signed(y) * f)
    return loss if weights is None else loss * weights


def balanced_weights(y):
    y = np.asarray(y)
    return np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))


def make_svc(Z, C, gamma_multiplier=GAMMA_MULTIPLIER, kernel="rbf"):
    gamma = gamma_multiplier / (Z.shape[1] * Z.var())
    return SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced")


def decision_by_hand(model, Z_query, Z_fit):
    K = rbf_kernel(Z_query, Z_fit[model.support_], gamma=model._gamma)
    return K @ model.dual_coef_[0] + model.intercept_[0]


def squared_norm_of_w(model, Z_fit):
    K = rbf_kernel(Z_fit[model.support_], gamma=model._gamma)
    alpha = model.dual_coef_[0]
    return float(alpha @ K @ alpha)


def primal_and_dual(model, Z_fit, y_fit, C):
    weights = balanced_weights(y_fit)
    f = model.decision_function(Z_fit)
    w2 = squared_norm_of_w(model, Z_fit)
    primal = 0.5 * w2 + C * hinge(y_fit, f, weights).sum()
    dual = np.abs(model.dual_coef_[0]).sum() - 0.5 * w2
    return float(primal), float(dual), w2


def support_rows(part, c_grid=C_GRID, n=SVM_SWEEP_SIZE):
    Z, y = subsample(part, n)
    weights = balanced_weights(y)
    rows = []
    for C in c_grid:
        model = make_svc(Z, C).fit(Z, y)
        alpha = np.abs(model.dual_coef_[0])
        upper = C * weights[model.support_]
        primal, dual, w2 = primal_and_dual(model, Z, y, C)
        f_val = model.decision_function(part["Z_val"])
        rows.append(
            {
                "C": C,
                "n_fit": len(y),
                "n_support": len(model.support_),
                "share_support": len(model.support_) / len(y),
                "share_at_upper_bound": float((alpha >= upper - 1e-8).sum() / len(y)),
                "margin_width": 2 / np.sqrt(w2),
                "primal": primal,
                "dual": dual,
                "train_hinge": float(hinge(y, model.decision_function(Z), weights).mean()),
                "val_hinge": float(hinge(part["y_val"], f_val, balanced_weights(part["y_val"])).mean()),
                "val_roc_auc": evaluate.roc_auc(part["y_val"], 1 / (1 + np.exp(-f_val))),
            }
        )
    return rows


def margin_rows(part, C=MARGIN_C, n=SVM_SWEEP_SIZE, rows=MARGIN_ROWS):
    Z, y = subsample(part, n)
    model = make_svc(Z, C).fit(Z, y)
    keep = np.random.default_rng(SEED).choice(len(part["y_val"]), size=rows, replace=False)
    f = model.decision_function(part["Z_val"][keep])
    return pd.DataFrame({"y": part["y_val"][keep], "f": f})


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = prepare_folds(X, y, scale=True, clip=True)[0]
    SUPPORT.parent.mkdir(exist_ok=True)
    table = pd.DataFrame(support_rows(part))
    table.to_csv(SUPPORT, index=False, float_format="%.6g")
    print(table.round(4).to_string(index=False))
    margins = margin_rows(part)
    margins.to_csv(MARGINS, index=False, float_format="%.5g")
    print(margins.groupby("y")["f"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
