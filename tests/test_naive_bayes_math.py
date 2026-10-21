import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

import naive_bayes_math as nb


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.2).astype(int)
    X = np.column_stack([rng.normal(0.8 * y, 1.0), rng.normal(-0.5 * y, 1.5), rng.normal(0.3 * y, 0.7)])
    return X, y


def test_the_hand_written_log_odds_equal_those_of_sklearn_gaussian_naive_bayes(data):
    X, y = data
    model = GaussianNB(var_smoothing=0.0).fit(X, y)
    log_proba = model.predict_log_proba(X)
    sklearn_log_odds = log_proba[:, 1] - log_proba[:, 0]
    assert nb.log_odds(nb.fit_gaussian(X, y), X) == pytest.approx(sklearn_log_odds, abs=1e-9)


def test_the_fitted_parameters_are_the_class_priors_means_and_variances(data):
    X, y = data
    params = nb.fit_gaussian(X, y)
    model = GaussianNB(var_smoothing=0.0).fit(X, y)
    assert params["prior"] == pytest.approx(model.class_prior_[1])
    assert params["mean_1"] == pytest.approx(model.theta_[1])
    assert params["var_0"] == pytest.approx(model.var_[0])


@pytest.mark.parametrize("smoothing", [1e-9, 1e-2, 1.0])
def test_var_smoothing_adds_its_share_of_the_largest_column_variance_to_every_class_variance(data, smoothing):
    X, y = data
    model = GaussianNB(var_smoothing=smoothing).fit(X, y)
    params = nb.fit_gaussian(X, y, smoothing)
    assert params["epsilon"] == pytest.approx(model.epsilon_)
    assert params["epsilon"] == pytest.approx(smoothing * X.var(axis=0).max())
    assert params["var_1"] == pytest.approx(model.var_[1])
    assert nb.log_odds(params, X) == pytest.approx(model.predict_log_proba(X)[:, 1] - model.predict_log_proba(X)[:, 0], abs=1e-9)


def test_every_column_adds_its_own_log_likelihood_ratio_to_the_prior_log_odds(data):
    X, y = data
    params = nb.fit_gaussian(X, y)
    parts = nb.evidence(params, X)
    assert parts.shape == X.shape
    assert nb.log_odds(params, X) == pytest.approx(np.log(params["prior"] / (1 - params["prior"])) + parts.sum(axis=1))


def test_with_equal_class_variances_the_evidence_of_a_column_is_linear_in_its_value():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(20000), np.ones(20000)].astype(int)
    X = np.concatenate([rng.normal(0, 1, 20000), rng.normal(1, 1, 20000)]).reshape(-1, 1)
    params = nb.fit_gaussian(X, y)
    params["var_1"] = params["var_0"] = np.array([1.0])
    grid = np.linspace(-3, 3, 7).reshape(-1, 1)
    second_difference = np.diff(nb.evidence(params, grid).ravel(), n=2)
    assert second_difference == pytest.approx(0.0, abs=1e-9)


def test_copying_a_column_counts_its_evidence_twice_and_makes_the_probabilities_overconfident():
    rng = np.random.default_rng(2)
    n = 40000
    y = (rng.random(n) < 0.2).astype(int)
    one = rng.normal(0.8 * y, 1.0).reshape(-1, 1)
    two = np.hstack([one, one])
    single, double = nb.fit_gaussian(one, y), nb.fit_gaussian(two, y)
    prior = np.log(single["prior"] / (1 - single["prior"]))
    assert nb.log_odds(double, two) - prior == pytest.approx(2 * (nb.log_odds(single, one) - prior))

    def log_loss(log_odds_values):
        p = 1 / (1 + np.exp(-log_odds_values))
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    assert log_loss(nb.log_odds(double, two)) > log_loss(nb.log_odds(single, one))


def test_for_columns_that_really_are_independent_given_the_class_the_probabilities_are_the_true_ones():
    rng = np.random.default_rng(3)
    n = 400000
    y = (rng.random(n) < 0.2).astype(int)
    X = np.column_stack([rng.normal(0.8 * y, 1.0), rng.normal(-0.5 * y, 1.5)])
    true_log_odds = np.log(0.2 / 0.8) + (-(X[:, 0] - 0.8) ** 2 / 2 + X[:, 0] ** 2 / 2) + (-(X[:, 1] + 0.5) ** 2 / (2 * 1.5**2) + X[:, 1] ** 2 / (2 * 1.5**2))
    estimated = nb.log_odds(nb.fit_gaussian(X, y), X)
    assert np.abs(estimated - true_log_odds).mean() < 0.01
