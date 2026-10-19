import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from preprocess import load_features, make_preprocessor
from split import DB
from tree import SEED

ROOT_SPLIT = Path("results/tree_root_split.csv")
NODES = Path("results/tree_depth3_nodes.csv")
DIAGRAM_DEPTH = 3


def gini(p):
    p = np.asarray(p, dtype=float)
    return 2.0 * p * (1.0 - p)


def entropy(p):
    p = np.asarray(p, dtype=float)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(q > 0, q * np.log2(q), 0.0))
    return bits


IMPURITY = {"gini": gini, "entropy": entropy}


def node_impurity(n, positives, criterion):
    return float(IMPURITY[criterion](positives / n))


def impurity_decrease(n_left, pos_left, n_right, pos_right, criterion):
    n = n_left + n_right
    parent = node_impurity(n, pos_left + pos_right, criterion)
    left = node_impurity(n_left, pos_left, criterion)
    right = node_impurity(n_right, pos_right, criterion)
    return parent - (n_left / n) * left - (n_right / n) * right


def feature_names(pre):
    return [name.split("__", 1)[1] for name in pre.get_feature_names_out()]


def fit_example(Z, y, criterion, max_depth):
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=SEED).fit(Z, y)


def node_table(tree, Z, y, names):
    structure = tree.tree_
    paths = tree.decision_path(Z).tocsc()
    rows = []
    for node in range(structure.node_count):
        members = paths[:, node].toarray().ravel().astype(bool)
        n = int(members.sum())
        positives = int(y[members].sum())
        leaf = structure.children_left[node] == -1
        rows.append(
            {
                "node": node,
                "left": int(structure.children_left[node]),
                "right": int(structure.children_right[node]),
                "feature": "" if leaf else names[structure.feature[node]],
                "threshold": np.nan if leaf else float(structure.threshold[node]),
                "samples": n,
                "defaults": positives,
                "default_rate": positives / n,
                "impurity": float(structure.impurity[node]),
            }
        )
    return pd.DataFrame(rows)


def root_split(table, criterion):
    root, left, right = table.iloc[0], table.iloc[table.iloc[0]["left"]], table.iloc[table.iloc[0]["right"]]
    decrease = impurity_decrease(left["samples"], left["defaults"], right["samples"], right["defaults"], criterion)
    return {
        "criterion": criterion,
        "feature": root["feature"],
        "threshold": root["threshold"],
        "samples": root["samples"],
        "defaults": root["defaults"],
        "impurity_parent": root["impurity"],
        "n_left": left["samples"],
        "defaults_left": left["defaults"],
        "impurity_left": left["impurity"],
        "n_right": right["samples"],
        "defaults_right": right["defaults"],
        "impurity_right": right["impurity"],
        "decrease": decrease,
    }


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    pre = make_preprocessor(scale=False).fit(X)
    Z, names = pre.transform(X), feature_names(pre)
    splits = []
    for criterion in ("gini", "entropy"):
        table = node_table(fit_example(Z, y, criterion, 1), Z, y, names)
        splits.append(root_split(table, criterion))
    ROOT_SPLIT.parent.mkdir(exist_ok=True)
    pd.DataFrame(splits).to_csv(ROOT_SPLIT, index=False, float_format="%.6f")
    print(pd.DataFrame(splits).T.to_string())
    diagram = node_table(fit_example(Z, y, "gini", DIAGRAM_DEPTH), Z, y, names)
    diagram.to_csv(NODES, index=False, float_format="%.6f")
    print(diagram.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
