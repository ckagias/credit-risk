import numpy as np
import pytest

import forest
import forest_math as fm


def correlated_trees(rho, datasets=40, trees=30, rows=3000, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(datasets, 1, rows))
    own = rng.normal(size=(datasets, trees, rows))
    return np.sqrt(rho) * shared + np.sqrt(1 - rho) * own


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7])
def test_variance_of_averages_follows_the_correlation_formula(rho):
    P = correlated_trees(rho)
    variances = fm.group_variances(P, [1, 2, 5, 10, 30])
    assert variances[1] == pytest.approx(1.0, abs=0.03)
    for b, value in variances.items():
        assert value == pytest.approx(fm.predicted_variance(b, rho, 1.0), abs=0.04)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7])
def test_the_fitted_correlation_recovers_the_one_that_was_built_in(rho):
    P = correlated_trees(rho)
    fitted, sigma2 = fm.fit_correlation(fm.group_variances(P, fm.GROUP_SIZES))
    assert fitted == pytest.approx(rho, abs=0.04)
    assert sigma2 == pytest.approx(1.0, abs=0.05)


def test_independent_trees_lose_all_their_variance_as_trees_are_added():
    P = correlated_trees(0.0)
    variances = fm.group_variances(P, [1, 30])
    assert variances[30] == pytest.approx(variances[1] / 30, rel=0.15)


def test_perfectly_correlated_trees_gain_nothing_from_averaging():
    P = correlated_trees(1.0)
    variances = fm.group_variances(P, [1, 30])
    assert variances[30] == pytest.approx(variances[1], rel=0.05)


def test_a_group_size_that_does_not_divide_the_trees_is_refused():
    with pytest.raises(ValueError):
        fm.group_variances(correlated_trees(0.3, trees=30), [4])


def test_fewer_columns_per_split_make_the_trees_less_alike():
    rng = np.random.default_rng(0)
    n = 6000
    Z = rng.normal(size=(n, 12)).astype(np.float32)
    logit = -1.5 + 1.2 * Z[:, 0] + 0.8 * Z[:, 1] - 0.8 * Z[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    part = {"Z_fit": Z[:4500], "y_fit": y[:4500], "Z_val": Z[4500:], "y_val": y[4500:]}
    found = {}
    for max_features in (2, 12):
        P = fm.forests_on_subsamples(part, max_features, datasets=6, trees=12, size=2500, leaf=20)
        found[max_features] = fm.fit_correlation(fm.group_variances(P, [1, 2, 3, 4, 6, 12]))
    assert found[2][0] < found[12][0]
