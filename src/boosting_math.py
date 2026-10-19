import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import boosting
import evaluate
import forest
from baselines import THRESHOLD
from preprocess import load_features
from split import DB

DEMO = Path("results/boosting_demo.csv")
SEED = 42
ROUNDS = 150
ETA = 0.1
LEAVES = 31
MIN_LEAF = 20
RECORD_EVERY = 5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_odds(rate):
    return float(np.log(rate / (1.0 - rate)))


def log_loss_of(y, F):
    p = np.clip(sigmoid(F), evaluate.EPS, 1 - evaluate.EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def residuals(y, F):
    return y - sigmoid(F)


def newton_leaf_values(leaf_of, r, p, size, penalty=0.0):
    numerator = np.bincount(leaf_of, weights=r, minlength=size)
    denominator = np.bincount(leaf_of, weights=p * (1.0 - p), minlength=size) + penalty
    return np.divide(numerator, denominator, out=np.zeros(size), where=denominator > 0)


def fit_boosting(Z, y, rounds=ROUNDS, eta=ETA, max_leaf_nodes=LEAVES, min_samples_leaf=MIN_LEAF, penalty=0.0, watch=None, record_every=RECORD_EVERY):
    y = np.asarray(y, dtype=float)
    F = np.full(len(y), log_odds(y.mean()))
    model = {"F0": float(F[0]), "eta": eta, "trees": [], "values": []}
    watched = None if watch is None else np.full(len(watch[1]), model["F0"])
    history = []
    for m in range(1, rounds + 1):
        p = sigmoid(F)
        r = y - p
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, random_state=SEED).fit(Z, r)
        leaf_of = tree.apply(Z)
        values = newton_leaf_values(leaf_of, r, p, tree.tree_.node_count, penalty)
        F = F + eta * values[leaf_of]
        model["trees"].append(tree)
        model["values"].append(values)
        if watched is not None:
            watched = watched + eta * values[tree.apply(watch[0])]
        if m % record_every == 0 or m == 1:
            row = {"round": m, "train_log_loss": log_loss_of(y, F)}
            if watched is not None:
                row.update(validation_log_loss=log_loss_of(watch[1], watched), validation_pr_auc=evaluate.pr_auc(watch[1], sigmoid(watched)))
            history.append(row)
    return model, pd.DataFrame(history)


def decision_function(model, Z, rounds=None):
    F = np.full(len(Z), model["F0"])
    for tree, values in list(zip(model["trees"], model["values"]))[:rounds]:
        F = F + model["eta"] * values[tree.apply(Z)]
    return F


def lightgbm_reference(part, rounds=ROUNDS, eta=ETA, record_every=RECORD_EVERY):
    booster = boosting.make_booster(eta, rounds, num_leaves=LEAVES, min_child_samples=MIN_LEAF).fit(part["Z_fit"], part["y_fit"])
    rows = []
    for k in [1] + list(range(record_every, rounds + 1, record_every)):
        p_val = booster.predict_proba(part["Z_val"], num_iteration=k)[:, 1]
        p_fit = booster.predict_proba(part["Z_fit"], num_iteration=k)[:, 1]
        rows.append(
            {
                "round": k,
                "lightgbm_train_log_loss": evaluate.log_loss(part["y_fit"], p_fit),
                "lightgbm_validation_log_loss": evaluate.log_loss(part["y_val"], p_val),
                "lightgbm_validation_pr_auc": evaluate.pr_auc(part["y_val"], p_val),
            }
        )
    return pd.DataFrame(rows)


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    part = forest.prepare_folds(X, y)[0]
    start = time.time()
    model, history = fit_boosting(part["Z_fit"], part["y_fit"], watch=(part["Z_val"], part["y_val"]))
    print(f"hand-written boosting, {ROUNDS} rounds: {time.time() - start:.0f}s")
    table = history.merge(lightgbm_reference(part), on="round")
    DEMO.parent.mkdir(exist_ok=True)
    table.to_csv(DEMO, index=False, float_format="%.6f")
    print(table.round(4).to_string(index=False))
    first = model["trees"][0]
    r0 = residuals(part["y_fit"], np.full(len(part["y_fit"]), model["F0"]))
    print(f"F0 {model['F0']:.4f} (probability {sigmoid(model['F0']):.4f}); residuals: default {r0[part['y_fit'] == 1][0]:.4f}, no default {r0[part['y_fit'] == 0][0]:.4f}")
    values = model["values"][0][first.tree_.children_left == -1]
    print(f"round 1: {first.get_n_leaves()} leaves, leaf values {values.min():.3f} to {values.max():.3f}, probability range after round 1: {sigmoid(model['F0'] + ETA * values.min()):.4f} to {sigmoid(model['F0'] + ETA * values.max()):.4f}")


if __name__ == "__main__":
    main()
