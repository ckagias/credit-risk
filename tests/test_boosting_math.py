import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

import boosting_math as bm


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(size=(n, 5))
    y = (rng.random(n) < bm.sigmoid(-1.5 + 1.5 * Z[:, 0] - 1.0 * Z[:, 1] + 0.8 * Z[:, 0] * Z[:, 2])).astype(int)
    return Z, y


def test_the_residual_is_minus_the_gradient_of_the_log_loss_with_respect_to_the_score():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    F = np.array([-2.0, -0.5, 0.7, 1.3])
    step = 1e-6
    for i in range(len(y)):
        up, down = F.copy(), F.copy()
        up[i] += step
        down[i] -= step
        numeric = (bm.log_loss_of(y, up) - bm.log_loss_of(y, down)) / (2 * step) * len(y)
        assert -bm.residuals(y, F)[i] == pytest.approx(numeric, abs=1e-6)


def test_the_first_score_is_the_log_odds_of_the_base_rate():
    assert bm.sigmoid(bm.log_odds(0.0807)) == pytest.approx(0.0807)
    assert bm.log_odds(0.5) == 0.0


def test_a_leaf_value_worked_out_by_hand():
    leaf_of = np.array([1, 1, 1, 2, 2])
    p = np.array([0.5, 0.5, 0.5, 0.2, 0.2])
    y = np.array([1, 0, 1, 0, 0])
    values = bm.newton_leaf_values(leaf_of, y - p, p, size=3)
    assert values[1] == pytest.approx((0.5 - 0.5 + 0.5) / (3 * 0.25))
    assert values[2] == pytest.approx((-0.2 - 0.2) / (2 * 0.16))
    assert values[0] == 0.0


def test_a_penalty_shrinks_every_leaf_value():
    leaf_of = np.array([1, 1, 2, 2])
    p = np.array([0.3, 0.3, 0.1, 0.1])
    r = np.array([0.7, -0.3, 0.9, -0.1])
    plain = bm.newton_leaf_values(leaf_of, r, p, 3)
    penalized = bm.newton_leaf_values(leaf_of, r, p, 3, penalty=5.0)
    assert (np.abs(penalized[1:]) < np.abs(plain[1:])).all()


def test_one_round_of_a_stump_lowers_the_training_log_loss(toy):
    Z, y = toy
    start = bm.log_loss_of(y, np.full(len(y), bm.log_odds(y.mean())))
    _, history = bm.fit_boosting(Z, y, rounds=1, eta=1.0, max_leaf_nodes=2, min_samples_leaf=1)
    assert history["train_log_loss"].iloc[0] < start


def test_the_training_loss_falls_at_every_recorded_round(toy):
    Z, y = toy
    _, history = bm.fit_boosting(Z, y, rounds=40, eta=0.1, max_leaf_nodes=8, min_samples_leaf=10, record_every=5)
    assert (np.diff(history["train_log_loss"].to_numpy()) < 0).all()


def test_the_scores_equal_those_of_scikit_learns_gradient_boosting(toy):
    Z, y = toy
    settings = dict(n_estimators=30, learning_rate=0.1, max_depth=None, max_leaf_nodes=8, min_samples_leaf=10, random_state=bm.SEED)
    library = GradientBoostingClassifier(loss="log_loss", **settings).fit(Z, y)
    model, _ = bm.fit_boosting(Z, y, rounds=30, eta=0.1, max_leaf_nodes=8, min_samples_leaf=10)
    assert bm.decision_function(model, Z) == pytest.approx(library.decision_function(Z), abs=1e-6)


def test_scoring_only_the_first_rounds_gives_the_earlier_model(toy):
    Z, y = toy
    model, _ = bm.fit_boosting(Z, y, rounds=20, eta=0.1, max_leaf_nodes=8, min_samples_leaf=10)
    short, _ = bm.fit_boosting(Z, y, rounds=7, eta=0.1, max_leaf_nodes=8, min_samples_leaf=10)
    assert bm.decision_function(model, Z, rounds=7) == pytest.approx(bm.decision_function(short, Z))


def test_the_watched_validation_set_does_not_change_the_fit(toy):
    Z, y = toy
    a, _ = bm.fit_boosting(Z[:1500], y[:1500], rounds=10, max_leaf_nodes=8, min_samples_leaf=10)
    b, history = bm.fit_boosting(Z[:1500], y[:1500], rounds=10, max_leaf_nodes=8, min_samples_leaf=10, watch=(Z[1500:], y[1500:]))
    assert bm.decision_function(a, Z) == pytest.approx(bm.decision_function(b, Z))
    assert {"validation_log_loss", "validation_pr_auc"} <= set(history.columns)
