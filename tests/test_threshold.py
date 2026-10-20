import numpy as np
import pandas as pd
import pytest

import threshold


def brute_force(y, p, t, ratio):
    flagged = p >= t
    missed = int(((y == 1) & ~flagged).sum())
    refused_good = int(((y == 0) & flagged).sum())
    return missed, refused_good, (ratio * missed + refused_good) / len(y)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 9, size=20000)
    y = (rng.random(20000) < p).astype(int)
    return y, p


def test_outcomes_match_a_direct_count_at_every_threshold(scored):
    y, p = scored
    grid = np.array([0.0, 0.05, 0.0909, 0.2, 0.5, 1.0000001])
    missed, refused = threshold.outcomes(y, p, grid)
    for t, m, r in zip(grid, missed, refused):
        assert (m, r) == brute_force(y, p, t, 1)[:2]


def test_a_score_equal_to_the_threshold_is_refused():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.2, 0.1])
    missed, refused = threshold.outcomes(y, p, np.array([0.5]))
    assert (int(missed[0]), int(refused[0])) == (1, 1)


@pytest.mark.parametrize("ratio", [2, 10])
def test_the_cost_per_applicant_matches_a_direct_count(scored, ratio):
    y, p = scored
    grid = np.array([0.03, 0.1, 0.3])
    for t, cost in zip(grid, threshold.cost_per_applicant(y, p, grid, ratio)):
        assert cost == pytest.approx(brute_force(y, p, t, ratio)[2])


def test_refusing_nobody_costs_the_ratio_times_the_default_rate_and_refusing_everybody_the_good_share(scored):
    y, p = scored
    nobody = threshold.rule_row(y, p, 1.0 + 1e-9, 10)
    everybody = threshold.rule_row(y, p, 0.0, 10)
    assert nobody["cost_per_1000"] == pytest.approx(1000 * 10 * y.mean())
    assert everybody["cost_per_1000"] == pytest.approx(1000 * (1 - y.mean()))
    assert nobody["share_refused"] == 0 and everybody["share_refused"] == pytest.approx(1)
    assert everybody["recall"] == 1


@pytest.mark.parametrize("ratio", [2, 5, 10, 20])
def test_the_best_threshold_of_true_probabilities_is_the_bayes_threshold(ratio):
    rng = np.random.default_rng(1)
    p = rng.beta(1, 9, size=600000)
    y = (rng.random(600000) < p).astype(int)
    assert threshold.bayes_threshold(ratio) == pytest.approx(1 / (1 + ratio))
    assert threshold.best_threshold(y, p, ratio) == pytest.approx(threshold.bayes_threshold(ratio), abs=0.02)


def test_scores_inflated_by_class_weights_have_a_best_threshold_far_above_the_bayes_one():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 9, size=300000)
    y = (rng.random(300000) < p).astype(int)
    odds = p / (1 - p) * 11.4
    inflated = odds / (1 + odds)
    assert threshold.best_threshold(y, inflated, 10) > 3 * threshold.bayes_threshold(10)


def test_best_f1_threshold_agrees_with_a_direct_search(scored):
    y, p = scored
    best = max(threshold.GRID, key=lambda t: 2 * ((y == 1) & (p >= t)).sum() / (((p >= t).sum()) + y.sum()))
    assert threshold.best_f1_threshold(y, p) == pytest.approx(best)


def test_each_rule_is_scored_once_per_fold_and_the_tuned_threshold_never_sees_the_scored_fold():
    rng = np.random.default_rng(3)
    folds = []
    for k in range(3):
        p = rng.beta(1, 9, size=4000)
        folds.append(((rng.random(4000) < p).astype(int), p))
    rows = threshold.rule_rows({"a": folds}, ratios=[10])
    assert len(rows) == 3 * 7
    assert set(rows["rule"]) == {"bayes", "cv_tuned", "best_f1", "fixed_0.5", "fixed_0.1", "refuse_nobody", "refuse_everybody"}
    y_other = np.concatenate([folds[1][0], folds[2][0]])
    p_other = np.concatenate([folds[1][1], folds[2][1]])
    tuned = rows[(rows["fold"] == 0) & (rows["rule"] == "cv_tuned")]["threshold"].iloc[0]
    assert tuned == threshold.best_threshold(y_other, p_other, 10)


def test_the_curve_has_one_row_per_scoring_ratio_and_threshold():
    rng = np.random.default_rng(4)
    p = rng.beta(1, 9, size=2000)
    folds = [((rng.random(2000) < p).astype(int), p)]
    curve = threshold.curve_rows({"a": folds}, ratios=[2, 10])
    assert len(curve) == 2 * len(threshold.GRID)
