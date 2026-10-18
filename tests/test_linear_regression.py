import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge

import linear_regression as lr


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = 5.0 + X @ np.array([3.0, -2.0, 0.0, 0.0]) + rng.normal(0, 0.5, 400)
    return X, y


def test_normal_equation_on_three_points_worked_out_by_hand():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    assert lr.normal_equation(X, y).tolist() == pytest.approx([0.0, 2.0], abs=1e-9)


def test_normal_equation_with_noise_on_three_points_worked_out_by_hand():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 2.0])
    assert lr.normal_equation(X, y).tolist() == pytest.approx([7 / 6, 0.5])


def test_normal_equation_matches_scikit_learn(toy):
    X, y = toy
    model = LinearRegression().fit(X, y)
    mine = lr.normal_equation(X, y)
    assert mine[0] == pytest.approx(model.intercept_)
    assert mine[1:] == pytest.approx(model.coef_)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 50.0, 1e4])
def test_ridge_closed_form_matches_scikit_learn(toy, alpha):
    X, y = toy
    intercept, weights = lr.ridge_closed_form(X, y, alpha)
    model = Ridge(alpha=alpha).fit(X, y)
    assert intercept == pytest.approx(model.intercept_, rel=1e-6)
    assert weights == pytest.approx(model.coef_, rel=1e-6)


def test_the_residuals_of_a_least_squares_fit_are_orthogonal_to_every_column(toy):
    X, y = toy
    w = lr.normal_equation(X, y)
    residual = y - np.column_stack([np.ones(len(X)), X]) @ w
    assert residual.sum() == pytest.approx(0, abs=1e-8)
    assert X.T @ residual == pytest.approx(np.zeros(4), abs=1e-8)


def test_a_larger_ridge_alpha_shrinks_the_weights(toy):
    X, y = toy
    sizes = [np.linalg.norm(lr.ridge_closed_form(X, y, a)[1]) for a in (0, 10, 1000, 1e6)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] < 0.01 * sizes[0]


def test_lasso_zeroes_the_uninformative_columns_first(toy):
    X, y = toy
    weights = Lasso(alpha=0.3).fit(X, y).coef_
    assert weights[2] == 0 and weights[3] == 0
    assert weights[0] > 0 and weights[1] < 0


def test_models_cover_ols_ridge_and_lasso_grids():
    found = lr.models()
    assert sum(k[0] == "OLS" for k in found) == 1
    assert sum(k[0] == "Ridge" for k in found) == len(lr.RIDGE_ALPHAS)
    assert sum(k[0] == "Lasso" for k in found) == len(lr.LASSO_ALPHAS)


def test_target_and_its_near_copies_are_not_among_the_columns():
    assert set(lr.DROPPED) == {"AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"}
