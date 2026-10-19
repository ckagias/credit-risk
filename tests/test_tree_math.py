import math

import numpy as np
import pytest

import tree_math as tm


def test_gini_and_entropy_at_known_points():
    assert tm.gini(0.0) == 0.0 and tm.gini(1.0) == 0.0
    assert tm.gini(0.5) == pytest.approx(0.5)
    assert tm.gini(0.1) == pytest.approx(0.18)
    assert tm.entropy(0.0) == 0.0 and tm.entropy(1.0) == 0.0
    assert tm.entropy(0.5) == pytest.approx(1.0)
    assert tm.entropy(0.25) == pytest.approx(-0.25 * math.log2(0.25) - 0.75 * math.log2(0.75))


def test_both_impurities_are_largest_at_one_half_and_symmetric():
    p = np.linspace(0, 1, 101)
    for impurity in (tm.gini, tm.entropy):
        assert impurity(p).argmax() == 50
        assert impurity(p) == pytest.approx(impurity(1 - p))


def test_gini_is_one_minus_the_sum_of_squared_class_shares():
    p = 0.3
    assert tm.gini(p) == pytest.approx(1 - p**2 - (1 - p) ** 2)


def test_a_split_worked_out_by_hand():
    # 10 applicants, 4 defaults. Left: 5 applicants with 4 defaults. Right: 5 applicants with 0.
    assert tm.impurity_decrease(5, 4, 5, 0, "gini") == pytest.approx(0.48 - 0.5 * 0.32)
    assert tm.impurity_decrease(5, 4, 5, 0, "entropy") == pytest.approx(
        tm.entropy(0.4) - 0.5 * tm.entropy(0.8)
    )


def test_a_split_that_changes_nothing_has_zero_decrease():
    assert tm.impurity_decrease(50, 10, 50, 10, "gini") == pytest.approx(0.0)
    assert tm.impurity_decrease(50, 10, 50, 10, "entropy") == pytest.approx(0.0)


def test_the_decrease_is_never_negative():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_left, n_right = rng.integers(1, 200, 2)
        pos_left, pos_right = rng.integers(0, n_left + 1), rng.integers(0, n_right + 1)
        for criterion in ("gini", "entropy"):
            assert tm.impurity_decrease(n_left, pos_left, n_right, pos_right, criterion) >= -1e-12


def test_the_hand_computed_decrease_matches_the_one_scikit_learn_makes():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(2000, 4))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(-1.5 + 2.0 * Z[:, 0])))).astype(int)
    for criterion in ("gini", "entropy"):
        tree = tm.fit_example(Z, y, criterion, 1)
        table = tm.node_table(tree, Z, y, ["a", "b", "c", "d"])
        split = tm.root_split(table, criterion)
        structure = tree.tree_
        library = structure.impurity[0] - (
            structure.n_node_samples[1] * structure.impurity[1] + structure.n_node_samples[2] * structure.impurity[2]
        ) / structure.n_node_samples[0]
        assert split["decrease"] == pytest.approx(library)
        assert split["feature"] == "a"
        assert split["impurity_parent"] == pytest.approx(tm.node_impurity(2000, int(y.sum()), criterion))


def test_node_counts_add_up_and_children_partition_the_parent():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(1500, 3))
    y = (rng.random(1500) < 0.2).astype(int)
    table = tm.node_table(tm.fit_example(Z, y, "gini", 3), Z, y, ["a", "b", "c"])
    assert table.loc[0, "samples"] == 1500 and table.loc[0, "defaults"] == y.sum()
    for _, node in table[table["left"] >= 0].iterrows():
        left, right = table.loc[node["left"]], table.loc[node["right"]]
        assert left["samples"] + right["samples"] == node["samples"]
        assert left["defaults"] + right["defaults"] == node["defaults"]
