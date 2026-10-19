import numpy as np
import pandas as pd
import pytest

import boosting
import evaluate
import xgboosting as xb


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 6000
    Z = rng.normal(size=(n, 6)).astype(np.float32)
    logit = -2.0 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1] + 0.8 * Z[:, 0] * Z[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return Z[:4500], y[:4500], Z[4500:], y[4500:]


def test_the_best_round_counts_rounds_not_the_zero_based_index(toy):
    Z, y, _, _ = toy
    booster = xb.fit_with_early_stopping(Z, y, 0.3, patience=20, max_rounds=1000)
    best = xb.BACKEND.best_round(booster)
    assert best == booster.best_iteration + 1
    assert best < 200


def test_early_stopping_scores_the_best_round_like_the_library_default(toy):
    Z, y, Zv, yv = toy
    booster = xb.fit_with_early_stopping(Z, y, 0.3, patience=20, max_rounds=1000)
    best = xb.BACKEND.best_round(booster)
    expected = evaluate.score(yv, booster.predict_proba(Zv)[:, 1], boosting.THRESHOLD)
    assert xb.score_at(booster, Zv, yv, best)["pr_auc"] == pytest.approx(expected["pr_auc"])


def test_scoring_fewer_rounds_uses_fewer_trees(toy):
    Z, y, Zv, yv = toy
    booster = xb.fit_fixed_rounds(Z, y, 0.3, 100)
    train = [xb.score_at(booster, Z, y, k)["pr_auc"] for k in (1, 30, 100)]
    assert train[0] < train[1] < train[2]


def test_a_smaller_learning_rate_needs_more_rounds(toy):
    Z, y, _, _ = toy
    fast = xb.fit_with_early_stopping(Z, y, 0.3, patience=20, max_rounds=2000)
    slow = xb.fit_with_early_stopping(Z, y, 0.03, patience=20, max_rounds=2000)
    assert xb.BACKEND.best_round(slow) > 3 * xb.BACKEND.best_round(fast)


def test_the_shared_fold_loop_runs_with_the_xgboost_backend(toy):
    Z, y, Zv, yv = toy
    curve, stopped = boosting.score_fold(
        0, Z, y, Zv, yv, learning_rates=[0.3], rounds=[1, 3, 100000], backend=xb.BACKEND, patience=20, max_rounds=500
    )
    curve = pd.DataFrame(curve)
    assert sorted(curve["rounds"].unique()) == [1, 3]
    assert len(curve) == 4
    assert stopped[0]["best_round"] >= 1
    assert 0 < stopped[0]["pr_auc"] <= 1

