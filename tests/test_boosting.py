import numpy as np
import pandas as pd
import pytest

import boosting
import evaluate


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 6000
    Z = rng.normal(size=(n, 6)).astype(np.float32)
    logit = -2.0 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1] + 0.8 * Z[:, 0] * Z[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return Z[:4500], y[:4500], Z[4500:], y[4500:]


def test_early_stopping_stops_before_the_round_limit(toy):
    Z, y, _, _ = toy
    booster = boosting.fit_with_early_stopping(Z, y, 0.3, patience=20, max_rounds=1000)
    assert booster.best_iteration_ < 200
    assert booster.booster_.num_trees() == booster.best_iteration_


def test_a_smaller_learning_rate_needs_more_rounds(toy):
    Z, y, _, _ = toy
    fast = boosting.fit_with_early_stopping(Z, y, 0.3, patience=20, max_rounds=2000)
    slow = boosting.fit_with_early_stopping(Z, y, 0.03, patience=20, max_rounds=2000)
    assert slow.best_iteration_ > 3 * fast.best_iteration_


def test_the_stopped_row_scores_the_best_round_on_the_validation_fold(toy):
    Z, y, Zv, yv = toy
    curve, stopped = boosting.score_fold(0, Z, y, Zv, yv, learning_rates=[0.1], rounds=[10, 50], patience=20, max_rounds=500)
    booster = boosting.fit_with_early_stopping(Z, y, 0.1, patience=20, max_rounds=500)
    expected = evaluate.score(yv, booster.predict_proba(Zv, num_iteration=booster.best_iteration_)[:, 1], boosting.THRESHOLD)
    assert stopped[0]["pr_auc"] == pytest.approx(expected["pr_auc"])
    assert stopped[0]["best_round"] == booster.best_iteration_


def test_the_curve_has_a_train_and_a_validation_row_for_each_round_reached(toy):
    Z, y, Zv, yv = toy
    curve, _ = boosting.score_fold(0, Z, y, Zv, yv, learning_rates=[0.1], rounds=[1, 3, 100000], patience=20, max_rounds=500)
    curve = pd.DataFrame(curve)
    assert sorted(curve["rounds"].unique()) == [1, 3]
    assert set(curve["split"]) == {"train", "validation"}
    assert len(curve) == 4


def test_the_curve_runs_past_the_best_round_so_overfitting_can_be_seen(toy):
    Z, y, Zv, yv = toy
    curve, stopped = boosting.score_fold(0, Z, y, Zv, yv, learning_rates=[0.3], rounds=list(range(1, 400)), patience=20, max_rounds=400)
    best = stopped[0]["best_round"]
    assert max(row["rounds"] for row in curve) > best


def test_boosting_beats_a_single_round_and_the_training_score_keeps_rising(toy):
    Z, y, Zv, yv = toy
    booster = boosting.fit_fixed_rounds(Z, y, 0.3, 200)
    train = [boosting.score_at(booster, Z, y, k)["pr_auc"] for k in (1, 50, 200)]
    validation = [boosting.score_at(booster, Zv, yv, k)["pr_auc"] for k in (1, 50)]
    assert train[0] < train[1] < train[2]
    assert validation[1] > validation[0]
