import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import logistic_math as lm


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(600, 5))
    y = (rng.random(600) < lm.sigmoid(-1.0 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1])).astype(int)
    return Z, y


def test_sigmoid_at_known_points():
    assert lm.sigmoid(0.0) == 0.5
    assert lm.sigmoid(math.log(3)) == pytest.approx(0.75)
    assert lm.sigmoid(-math.log(3)) == pytest.approx(0.25)
    assert lm.sigmoid(np.array([-1000.0, 1000.0])).tolist() == pytest.approx([0.0, 1.0], abs=1e-100)


def test_sigmoid_is_symmetric_and_increasing():
    z = np.linspace(-8, 8, 101)
    assert lm.sigmoid(z) + lm.sigmoid(-z) == pytest.approx(np.ones_like(z))
    assert (np.diff(lm.sigmoid(z)) > 0).all()


def test_loss_and_gradient_at_zero_weights_worked_out_by_hand():
    Z = np.array([[2.0], [-1.0]])
    y = np.array([1, 0])
    w, b = np.zeros(1), 0.0
    assert lm.objective(Z, y, w, b, lam=0.0) == pytest.approx(math.log(2))
    grad_w, grad_b = lm.gradient(Z, y, w, b, lam=0.0)
    assert grad_w == pytest.approx([-0.75])
    assert grad_b == pytest.approx(0.0)


def test_the_penalty_adds_lambda_times_the_weights_to_the_gradient():
    Z = np.array([[2.0], [-1.0]])
    y = np.array([1, 0])
    grad_w, _ = lm.gradient(Z, y, np.array([0.4]), 0.0, lam=0.5)
    plain, _ = lm.gradient(Z, y, np.array([0.4]), 0.0, lam=0.0)
    assert grad_w - plain == pytest.approx([0.5 * 0.4])


def test_gradient_matches_finite_differences(toy):
    Z, y = toy
    rng = np.random.default_rng(1)
    w, b, lam = rng.normal(size=5) * 0.3, 0.2, 0.01
    grad_w, grad_b = lm.gradient(Z, y, w, b, lam)
    step = 1e-6
    for j in range(5):
        up, down = w.copy(), w.copy()
        up[j] += step
        down[j] -= step
        numeric = (lm.objective(Z, y, up, b, lam) - lm.objective(Z, y, down, b, lam)) / (2 * step)
        assert grad_w[j] == pytest.approx(numeric, abs=1e-7)
    numeric_b = (lm.objective(Z, y, w, b + step, lam) - lm.objective(Z, y, w, b - step, lam)) / (2 * step)
    assert grad_b == pytest.approx(numeric_b, abs=1e-7)


def test_the_loss_never_goes_up_during_gradient_descent(toy):
    Z, y = toy
    _, _, history, _ = lm.gradient_descent(Z, y, lam=0.01, steps=200)
    assert (np.diff(history[:, 2]) <= 1e-12).all()
    assert history[0, 1] == pytest.approx(math.log(2))


@pytest.mark.parametrize("C", [0.01, 1.0])
def test_gradient_descent_reaches_the_scikit_learn_solution(toy, C):
    Z, y = toy
    lam = lm.penalty_weight(C, len(y))
    model = LogisticRegression(C=C, l1_ratio=0.0, tol=1e-10, max_iter=5000).fit(Z, y)
    w, b, history, _ = lm.gradient_descent(Z, y, lam, steps=20000, record_every=20000)
    assert w == pytest.approx(model.coef_.ravel(), abs=2e-3)
    assert b == pytest.approx(float(model.intercept_[0]), abs=2e-3)
    optimum = lm.objective(Z, y, model.coef_.ravel(), float(model.intercept_[0]), lam)
    assert history[-1, 2] == pytest.approx(optimum, abs=1e-6)


def test_penalty_weight_follows_the_scikit_learn_convention():
    assert lm.penalty_weight(0.03, 246008) == pytest.approx(1 / (0.03 * 246008))
