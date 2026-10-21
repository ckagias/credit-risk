from itertools import combinations, permutations
from math import factorial

import numpy as np


def shapley_values(value, n_players):
    values = np.zeros(n_players)
    for i in range(n_players):
        others = [j for j in range(n_players) if j != i]
        for size in range(n_players):
            weight = factorial(size) * factorial(n_players - size - 1) / factorial(n_players)
            for subset in combinations(others, size):
                values[i] += weight * (value(frozenset(subset) | {i}) - value(frozenset(subset)))
    return values


def shapley_by_orderings(value, n_players):
    values = np.zeros(n_players)
    orderings = list(permutations(range(n_players)))
    for order in orderings:
        present = frozenset()
        for i in order:
            values[i] += value(present | {i}) - value(present)
            present = present | {i}
    return values / len(orderings)


def tree_tables(booster):
    table = booster.trees_to_dataframe()
    table["feature"] = table["split_feature"].str.removeprefix("Column_").fillna("-1").astype(int)
    return [group.set_index("node_index") for _, group in table.groupby("tree_index")]


def tree_expectation(nodes, x, known, node=None):
    node = nodes.index[0] if node is None else node
    row = nodes.loc[node]
    if row["feature"] < 0:
        return float(row["value"])
    left, right = row["left_child"], row["right_child"]
    if row["feature"] in known:
        return tree_expectation(nodes, x, known, left if x[row["feature"]] <= row["threshold"] else right)
    left_share = nodes.loc[left, "count"] / row["count"]
    return left_share * tree_expectation(nodes, x, known, left) + (1 - left_share) * tree_expectation(nodes, x, known, right)


def tree_value_function(booster, x):
    trees = tree_tables(booster)
    return lambda known: sum(tree_expectation(nodes, x, known) for nodes in trees)
