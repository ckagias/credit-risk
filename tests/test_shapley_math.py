import lightgbm as lgb
import numpy as np
import pytest

import explain
import shapley_math as sm


def toy_value(subset):
    payoff = {frozenset(): 0.0, frozenset({0}): 10.0, frozenset({1}): 20.0, frozenset({2}): 0.0}
    payoff.update({frozenset({0, 1}): 40.0, frozenset({0, 2}): 10.0, frozenset({1, 2}): 20.0, frozenset({0, 1, 2}): 60.0})
    return payoff[frozenset(subset)]


def test_the_worked_three_player_example_has_the_hand_computed_shapley_values():
    values = sm.shapley_values(toy_value, 3)
    assert values == pytest.approx([130 / 6, 190 / 6, 40 / 6])


def test_the_subset_formula_and_the_average_over_orderings_are_the_same_numbers():
    rng = np.random.default_rng(0)
    table = {frozenset(s): rng.normal() for s in [(), (0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]}
    value = lambda subset: table[frozenset(subset)]
    assert sm.shapley_values(value, 4) == pytest.approx(sm.shapley_by_orderings(value, 4))


def test_efficiency_the_values_add_up_to_the_full_value_minus_the_empty_value():
    values = sm.shapley_values(toy_value, 3)
    assert values.sum() == pytest.approx(toy_value({0, 1, 2}) - toy_value(set()))


def test_a_player_that_never_changes_the_value_gets_zero_and_equal_players_get_equal_values():
    value = lambda subset: 3.0 * (0 in subset) + 3.0 * (1 in subset) + 5.0 * (0 in subset and 1 in subset)
    values = sm.shapley_values(value, 3)
    assert values[2] == pytest.approx(0.0)
    assert values[0] == pytest.approx(values[1])


def test_the_values_of_a_sum_of_two_games_are_the_sums_of_their_values():
    a = lambda subset: float(len(subset)) ** 2
    b = lambda subset: 2.0 * (1 in subset) - (0 in subset)
    both = sm.shapley_values(lambda s: a(s) + b(s), 3)
    assert both == pytest.approx(sm.shapley_values(a, 3) + sm.shapley_values(b, 3))


def test_a_linear_model_with_other_features_at_their_mean_gives_weight_times_distance_from_the_mean():
    rng = np.random.default_rng(1)
    w, mean, x = np.array([2.0, -1.0, 0.5, 3.0]), rng.normal(size=4), rng.normal(size=4)

    def value(subset):
        z = mean.copy()
        for j in subset:
            z[j] = x[j]
        return float(w @ z)

    assert sm.shapley_values(value, 4) == pytest.approx(w * (x - mean))


@pytest.fixture(scope="module")
def forest():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(4000, 4))
    y = (Z[:, 0] + Z[:, 1] * Z[:, 2] + 0.5 * rng.normal(size=4000) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=15, num_leaves=6, learning_rate=0.2, random_state=0, verbose=-1).fit(Z, y)
    return model, Z


def test_the_tree_expectation_with_every_feature_known_is_the_raw_score_of_the_model(forest):
    model, Z = forest
    value = sm.tree_value_function(model.booster_, Z[0])
    assert value({0, 1, 2, 3}) == pytest.approx(model.predict(Z[:1], raw_score=True)[0], abs=1e-9)


def test_the_tree_expectation_with_nothing_known_is_the_expected_value_of_shap(forest):
    model, Z = forest
    _, base = explain.shap_values(model, Z[:1])
    assert sm.tree_value_function(model.booster_, Z[0])(set()) == pytest.approx(base, abs=1e-6)


@pytest.mark.parametrize("row", [0, 1, 2])
def test_exact_shapley_values_of_the_trees_equal_what_the_shap_library_returns(forest, row):
    model, Z = forest
    exact = sm.shapley_values(sm.tree_value_function(model.booster_, Z[row]), 4)
    library, base = explain.shap_values(model, Z[row : row + 1])
    assert exact == pytest.approx(library[0], abs=1e-6)
    assert exact.sum() + base == pytest.approx(model.predict(Z[row : row + 1], raw_score=True)[0], abs=1e-6)
