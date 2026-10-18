import numpy as np
import pandas as pd
import pytest

import evaluate as ev
import logistic


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = -2.5 + 1.5 * X["f0"] - 1.0 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def weights(model):
    return model.steps[-1][1].coef_.ravel()


def test_make_model_passes_its_settings_to_the_regression():
    regression = logistic.make_model(C=0.3, l1_ratio=1.0, solver="saga").steps[-1][1]
    assert (regression.C, regression.l1_ratio, regression.solver) == (0.3, 1.0, "saga")


def test_the_model_learns_a_real_signal(toy):
    X, y = toy
    folds = ev.cross_validate(logistic.make_model(), X, y, seed=1)
    assert folds["roc_auc"].mean() > 0.75
    assert folds["pr_auc"].mean() > y.mean() * 1.5


def test_a_smaller_C_shrinks_the_weights(toy):
    X, y = toy
    loose = logistic.make_model(C=10).fit(X, y)
    tight = logistic.make_model(C=0.001).fit(X, y)
    assert np.linalg.norm(weights(tight)) < 0.2 * np.linalg.norm(weights(loose))


def test_the_two_informative_columns_get_the_largest_weights_with_the_right_signs(toy):
    X, y = toy
    w = weights(logistic.make_model(C=1).fit(X, y))
    assert w[0] > 0 and w[1] < 0
    assert set(np.argsort(-np.abs(w))[:2]) == {0, 1}
