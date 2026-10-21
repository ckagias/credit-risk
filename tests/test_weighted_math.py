import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

import evaluate
import weighted_math as wm


def test_weight_one_gives_the_ordinary_log_loss():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    assert wm.weighted_log_loss(y, p, 1.0) == pytest.approx(evaluate.log_loss(y, p))


def test_a_weight_of_three_on_the_positives_is_the_same_as_copying_each_defaulter_three_times():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < 0.3).astype(int)
    copied_y = np.concatenate([y, y[y == 1], y[y == 1]])
    copied_p = np.concatenate([p, p[y == 1], p[y == 1]])
    assert wm.weighted_log_loss(y, p, 3.0) * len(y) == pytest.approx(evaluate.log_loss(copied_y, copied_p) * len(copied_y))


@pytest.mark.parametrize("p_true", [0.02, 0.081, 0.3])
@pytest.mark.parametrize("weight", [1.0, 3.0, 11.4])
def test_the_best_constant_prediction_under_the_weighted_loss_is_the_shifted_probability(p_true, weight):
    y = np.r_[np.ones(int(p_true * 100000)), np.zeros(100000 - int(p_true * 100000))].astype(int)
    result = minimize_scalar(lambda q: wm.weighted_log_loss(y, np.full(len(y), q), weight), bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-10})
    assert result.x == pytest.approx(wm.weighted_optimum(y.mean(), weight), abs=1e-5)


def test_the_shift_formula_is_a_multiplication_of_the_odds_and_is_undone_by_the_inverse_weight():
    p = np.array([0.01, 0.081, 0.4])
    shifted = wm.shifted_probability(p, 11.4)
    assert shifted / (1 - shifted) == pytest.approx(11.4 * p / (1 - p))
    assert wm.shifted_probability(shifted, 1 / 11.4) == pytest.approx(p)
    assert wm.weighted_optimum(p, 11.4) == pytest.approx(shifted)


def test_logistic_regression_with_class_weights_shifts_only_the_intercept_when_the_model_is_right():
    rng = np.random.default_rng(2)
    n = 200000
    x = rng.normal(size=(n, 1))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.4 + 0.9 * x[:, 0])))).astype(int)
    plain = LogisticRegression(C=1e6, max_iter=500).fit(x, y)
    w = (n - y.sum()) / y.sum()
    weighted = LogisticRegression(C=1e6, max_iter=500, class_weight={0: 1.0, 1: w}).fit(x, y)
    assert weighted.coef_[0, 0] == pytest.approx(plain.coef_[0, 0], abs=0.03)
    assert weighted.intercept_[0] - plain.intercept_[0] == pytest.approx(np.log(w), abs=0.05)


def test_platt_scaling_on_shifted_logodds_recovers_a_slope_of_one_and_an_intercept_of_minus_log_w():
    import calibration

    rng = np.random.default_rng(3)
    p = rng.beta(1, 9, size=300000)
    y = (rng.random(300000) < p).astype(int)
    shifted = wm.shifted_probability(p, 11.4)
    platt = calibration.Platt().fit(shifted, y).model_
    assert platt.coef_[0, 0] == pytest.approx(1.0, abs=0.04)
    assert platt.intercept_[0] == pytest.approx(-np.log(11.4), abs=0.08)


def test_the_mean_shift_is_the_intercept_plus_the_stretch_times_the_mean_log_odds():
    rng = np.random.default_rng(3)
    x = rng.normal(-2.5, 0.9, 5000)
    z = 2.4 + 1.07 * x + rng.normal(0, 0.3, 5000)
    d = wm.shift_decomposition(x, z)
    assert d["slope"] == pytest.approx(1.07, abs=0.02)
    assert d["mean_shift"] == pytest.approx(d["intercept"] + d["stretch_part"])
