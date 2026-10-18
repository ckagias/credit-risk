import math

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.dummy import DummyClassifier

import evaluate as ev

Y = np.array([1, 1, 0, 1, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])


def test_roc_auc_on_the_worked_example():
    assert ev.roc_auc(Y, P) == pytest.approx(8 / 9)


def test_pr_auc_on_the_worked_example():
    assert ev.pr_auc(Y, P) == pytest.approx(11 / 12)


def test_log_loss_on_the_worked_example():
    by_hand = -(math.log(0.9) + math.log(0.8) + math.log(0.3) + math.log(0.6) + math.log(0.6) + math.log(0.8)) / 6
    assert ev.log_loss(Y, P) == pytest.approx(by_hand)
    assert ev.log_loss(Y, P) == pytest.approx(0.4629, abs=1e-4)


@pytest.mark.parametrize(
    "threshold, precision, recall",
    [(0.5, 3 / 4, 1.0), (0.75, 1.0, 2 / 3), (0.9, 1.0, 1 / 3), (0.95, 0.0, 0.0), (0.0, 0.5, 1.0)],
)
def test_precision_and_recall_at_a_threshold(threshold, precision, recall):
    assert ev.precision_recall_at(Y, P, threshold) == pytest.approx((precision, recall))


def test_a_score_equal_to_the_threshold_is_flagged():
    assert ev.precision_recall_at(Y, P, 0.6) == pytest.approx((3 / 4, 1.0))


def test_partly_tied_scores_on_the_worked_example():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.8, 0.3, 0.1])
    assert ev.roc_auc(y, p) == pytest.approx(2.5 / 4)
    assert ev.pr_auc(y, p) == pytest.approx(7 / 12)


def test_everyone_with_the_same_score_gets_the_base_rate_and_a_coin_flip():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.full(10, 0.1)
    assert ev.pr_auc(y, p) == pytest.approx(0.1)
    assert ev.roc_auc(y, p) == pytest.approx(0.5)
    assert ev.log_loss(y, p) == pytest.approx(-(0.1 * math.log(0.1) + 0.9 * math.log(0.9)))


def test_a_perfect_ranking_scores_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert ev.pr_auc(y, p) == pytest.approx(1.0)
    assert ev.roc_auc(y, p) == pytest.approx(1.0)


def test_a_reversed_ranking_scores_zero_on_roc():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert ev.roc_auc(y, p) == pytest.approx(0.0)


def test_a_certain_wrong_answer_costs_a_finite_amount():
    assert ev.log_loss([1], [0.0]) == pytest.approx(-math.log(ev.EPS))
    assert math.isfinite(ev.log_loss([0], [1.0]))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rate", [0.08, 0.3])
@pytest.mark.parametrize("decimals", [None, 1])
def test_metrics_match_scikit_learn(seed, rate, decimals):
    rng = np.random.default_rng(seed)
    n = 5000
    y = (rng.random(n) < rate).astype(int)
    p = np.clip(rate + 0.2 * (y - rate) + rng.normal(0, 0.12, n), 0, 1)
    if decimals is not None:
        p = p.round(decimals)
    threshold = 0.25
    mine = ev.score(y, p, threshold)
    assert mine["pr_auc"] == pytest.approx(skm.average_precision_score(y, p), abs=1e-12)
    assert mine["roc_auc"] == pytest.approx(skm.roc_auc_score(y, p), abs=1e-12)
    assert mine["log_loss"] == pytest.approx(skm.log_loss(y, np.clip(p, ev.EPS, 1 - ev.EPS)), abs=1e-12)
    assert mine["precision"] == pytest.approx(skm.precision_score(y, p >= threshold, zero_division=0), abs=1e-12)
    assert mine["recall"] == pytest.approx(skm.recall_score(y, p >= threshold), abs=1e-12)


@pytest.mark.parametrize("y, p", [([1, 1, 1], [0.2, 0.5, 0.9]), ([0, 0, 0], [0.2, 0.5, 0.9])])
def test_a_single_class_is_refused_for_the_rank_metrics(y, p):
    with pytest.raises(ValueError):
        ev.roc_auc(y, p)
    with pytest.raises(ValueError):
        ev.pr_auc(y, p)


@pytest.mark.parametrize(
    "y, p",
    [([0, 1, 2], [0.1, 0.5, 0.9]), ([0, 1], [0.1, 0.5, 0.9]), ([0, 1], [0.1, 1.2]), ([0, 1], [0.1, float("nan")])],
)
def test_bad_inputs_are_refused(y, p):
    with pytest.raises(ValueError):
        ev.score(y, p)


@pytest.fixture
def toy_cv():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    y = np.array([1] * 100 + [0] * 900)
    return X, y


def test_cross_validation_with_the_base_rate_model_is_worked_out_by_hand(toy_cv):
    X, y = toy_cv
    folds = ev.cross_validate(DummyClassifier(strategy="prior"), X, y, n_splits=5)
    entropy = -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))
    assert len(folds) == 5
    assert folds["pr_auc"].tolist() == pytest.approx([0.1] * 5)
    assert folds["roc_auc"].tolist() == pytest.approx([0.5] * 5)
    assert folds["log_loss"].tolist() == pytest.approx([entropy] * 5)


def test_cross_validation_is_the_same_on_every_run_and_changes_with_the_seed(toy_cv):
    X, y = toy_cv
    from sklearn.linear_model import LogisticRegression

    first = ev.cross_validate(LogisticRegression(), X, y, seed=1)
    again = ev.cross_validate(LogisticRegression(), X, y, seed=1)
    other = ev.cross_validate(LogisticRegression(), X, y, seed=2)
    assert first.equals(again)
    assert not first.equals(other)


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.2 * (y - 0.1) + rng.normal(0, 0.1, 2000), 0, 1)
    first = ev.bootstrap(y, p, n_boot=200, seed=5)
    again = ev.bootstrap(y, p, n_boot=200, seed=5)
    assert first.equals(again)
    assert (first["low"] <= first["estimate"]).all() and (first["estimate"] <= first["high"]).all()
    assert first["estimate"].to_dict() == pytest.approx(ev.score(y, p))


def test_bootstrap_interval_is_narrower_with_more_applicants():
    def width(n):
        rng = np.random.default_rng(3)
        y = (rng.random(n) < 0.1).astype(int)
        p = np.clip(0.1 + 0.2 * (y - 0.1) + rng.normal(0, 0.1, n), 0, 1)
        table = ev.bootstrap(y, p, n_boot=200)
        return table.loc["roc_auc", "high"] - table.loc["roc_auc", "low"]

    assert width(8000) < width(800)


def test_bootstrap_interval_of_a_perfect_ranking_has_no_width():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 25)
    p = np.where(y == 1, 0.9, 0.1)
    table = ev.bootstrap(y, p, n_boot=100)
    assert table.loc["roc_auc"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_curves_on_the_worked_example():
    fpr, tpr = ev.roc_curve(Y, P)
    assert fpr.tolist() == pytest.approx([0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 1])
    assert tpr.tolist() == pytest.approx([0, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1])
    recall, precision = ev.pr_curve(Y, P)
    assert recall.tolist() == pytest.approx([1 / 3, 2 / 3, 2 / 3, 1, 1, 1])
    assert precision.tolist() == pytest.approx([1, 1, 2 / 3, 3 / 4, 3 / 5, 1 / 2])


@pytest.mark.parametrize("decimals", [None, 1])
def test_curves_match_scikit_learn(decimals):
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.2 * (y - 0.1) + rng.normal(0, 0.1, 3000), 0, 1)
    if decimals is not None:
        p = p.round(decimals)
    fpr, tpr = ev.roc_curve(y, p)
    sk_fpr, sk_tpr, _ = skm.roc_curve(y, p, drop_intermediate=False)
    assert fpr == pytest.approx(sk_fpr, abs=1e-12)
    assert tpr == pytest.approx(sk_tpr, abs=1e-12)
    recall, precision = ev.pr_curve(y, p)
    sk_precision, sk_recall, _ = skm.precision_recall_curve(y, p)
    sk_precision, sk_recall = sk_precision[::-1][1:], sk_recall[::-1][1:]
    assert recall[: len(sk_recall)] == pytest.approx(sk_recall, abs=1e-12)
    assert precision[: len(sk_precision)] == pytest.approx(sk_precision, abs=1e-12)


def test_out_of_fold_predictions_come_from_a_model_that_did_not_see_the_row(toy_cv):
    X, y = toy_cv
    from sklearn.linear_model import LogisticRegression

    predictions = ev.out_of_fold_predictions(LogisticRegression(), X, y, seed=1)
    assert predictions.shape == y.shape
    assert ((predictions > 0) & (predictions < 1)).all()
    folds = ev.cross_validate(LogisticRegression(), X, y, seed=1)
    from sklearn.model_selection import StratifiedKFold

    split = StratifiedKFold(5, shuffle=True, random_state=1).split(X, y)
    per_fold = [ev.roc_auc(y[val], predictions[val]) for _, val in split]
    assert per_fold == pytest.approx(folds["roc_auc"].tolist())
