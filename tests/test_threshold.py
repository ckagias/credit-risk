import numpy as np
import pandas as pd
import pytest

import threshold


def flat(ratio):
    return threshold.Scheme(f"flat_{ratio}", float(ratio), 1.0, False)


def brute_force(y, p, t, ratio, size=None):
    flagged = p >= t
    w = np.ones(len(y)) if size is None else size
    missed = float(w[(y == 1) & ~flagged].sum())
    refused_good = float(w[(y == 0) & flagged].sum())
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
    for t, cost in zip(grid, threshold.cost_per_applicant(y, p, grid, flat(ratio))):
        assert cost == pytest.approx(brute_force(y, p, t, ratio)[2])


def test_refusing_nobody_costs_the_ratio_times_the_default_rate_and_refusing_everybody_the_good_share(scored):
    y, p = scored
    nobody = threshold.rule_row(y, p, 1.0 + 1e-9, flat(10))
    everybody = threshold.rule_row(y, p, 0.0, flat(10))
    assert nobody["cost_per_1000"] == pytest.approx(1000 * 10 * y.mean())
    assert everybody["cost_per_1000"] == pytest.approx(1000 * (1 - y.mean()))
    assert nobody["share_refused"] == 0 and everybody["share_refused"] == pytest.approx(1)
    assert everybody["recall"] == 1


@pytest.mark.parametrize("ratio", [2, 5, 10, 20])
def test_the_best_threshold_of_true_probabilities_is_the_bayes_threshold(ratio):
    rng = np.random.default_rng(1)
    p = rng.beta(1, 9, size=600000)
    y = (rng.random(600000) < p).astype(int)
    assert threshold.bayes_threshold(flat(ratio)) == pytest.approx(1 / (1 + ratio))
    assert threshold.best_threshold(y, p, flat(ratio)) == pytest.approx(threshold.bayes_threshold(flat(ratio)), abs=0.02)


def test_scores_inflated_by_class_weights_have_a_best_threshold_far_above_the_bayes_one():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 9, size=300000)
    y = (rng.random(300000) < p).astype(int)
    odds = p / (1 - p) * 11.4
    inflated = odds / (1 + odds)
    assert threshold.best_threshold(y, inflated, flat(10)) > 3 * threshold.bayes_threshold(flat(10))


def test_best_f1_threshold_agrees_with_a_direct_search(scored):
    y, p = scored
    best = max(threshold.GRID, key=lambda t: 2 * ((y == 1) & (p >= t)).sum() / (((p >= t).sum()) + y.sum()))
    assert threshold.best_f1_threshold(y, p) == pytest.approx(best)


def test_each_rule_is_scored_once_per_fold_and_the_tuned_threshold_never_sees_the_scored_fold():
    rng = np.random.default_rng(3)
    folds = []
    for k in range(3):
        p = rng.beta(1, 9, size=4000)
        folds.append(((rng.random(4000) < p).astype(int), p, rng.lognormal(0, 0.5, 4000)))
    rows = threshold.rule_rows({"a": folds}, schemes=[flat(10)])
    assert len(rows) == 3 * 7
    assert set(rows["rule"]) == {"bayes", "cv_tuned", "best_f1", "fixed_0.5", "fixed_0.1", "refuse_nobody", "refuse_everybody"}
    y_other = np.concatenate([folds[1][0], folds[2][0]])
    p_other = np.concatenate([folds[1][1], folds[2][1]])
    tuned = rows[(rows["fold"] == 0) & (rows["rule"] == "cv_tuned")]["threshold"].iloc[0]
    assert tuned == threshold.best_threshold(y_other, p_other, flat(10))


def test_the_curve_has_one_row_per_scoring_ratio_and_threshold():
    rng = np.random.default_rng(4)
    p = rng.beta(1, 9, size=2000)
    folds = [((rng.random(2000) < p).astype(int), p, np.ones(2000))]
    curve = threshold.curve_rows({"a": folds}, schemes=[flat(2), flat(10)])
    assert len(curve) == 2 * len(threshold.GRID)


def test_the_expected_cost_of_true_probabilities_matches_the_realised_cost_and_is_lowest_at_the_bayes_threshold():
    rng = np.random.default_rng(5)
    p = rng.beta(1, 9, size=400000)
    y = (rng.random(400000) < p).astype(int)
    for t in (0.05, threshold.bayes_threshold(flat(10)), 0.3):
        row = threshold.rule_row(y, p, t, flat(10))
        assert row["expected_cost_per_1000"] == pytest.approx(row["cost_per_1000"], rel=0.03)
    expected = {t: threshold.expected_cost_per_1000(p, t, flat(10)) for t in (0.05, threshold.bayes_threshold(flat(10)), 0.3)}
    assert min(expected, key=expected.get) == pytest.approx(threshold.bayes_threshold(flat(10)))
    assert threshold.expected_cost_per_1000(p, threshold.bayes_threshold(flat(10)), flat(10)) == pytest.approx(1000 * np.mean(np.minimum(10 * p, 1 - p)))


def test_the_amount_scheme_is_a_loss_of_45_over_a_margin_of_10_weighted_by_loan_size():
    assert threshold.AMOUNT.ratio == pytest.approx(4.5)
    assert threshold.AMOUNT.by_size
    assert threshold.bayes_threshold(threshold.AMOUNT) == pytest.approx(1 / 5.5)


def test_weighted_outcomes_match_a_direct_sum_of_loan_sizes(scored):
    y, p = scored
    size = np.random.default_rng(6).lognormal(0, 0.6, len(y))
    grid = np.array([0.0, 0.05, 0.1818, 0.5, 1.0000001])
    missed, refused = threshold.outcomes(y, p, grid, size)
    for t, m, r in zip(grid, missed, refused):
        assert (m, r) == pytest.approx(brute_force(y, p, t, 1, size)[:2])


def test_the_amount_cost_matches_a_direct_sum_and_equals_the_flat_cost_when_every_loan_is_the_same_size(scored):
    y, p = scored
    size = np.random.default_rng(7).lognormal(0, 0.6, len(y))
    grid = np.array([0.05, 0.1818, 0.4])
    for t, cost in zip(grid, threshold.cost_per_applicant(y, p, grid, threshold.AMOUNT, size)):
        assert cost == pytest.approx(brute_force(y, p, t, 4.5, size)[2])
    same = threshold.cost_per_applicant(y, p, grid, threshold.AMOUNT, np.ones(len(y)))
    assert same == pytest.approx(threshold.cost_per_applicant(y, p, grid, flat(4.5)))


def test_a_flat_scheme_ignores_the_loan_sizes(scored):
    y, p = scored
    size = np.random.default_rng(8).lognormal(0, 0.6, len(y))
    grid = np.array([0.05, 0.3])
    assert threshold.cost_per_applicant(y, p, grid, flat(10), size) == pytest.approx(threshold.cost_per_applicant(y, p, grid, flat(10)))


def test_the_best_threshold_of_true_probabilities_is_the_same_whatever_the_loan_sizes():
    rng = np.random.default_rng(9)
    p = rng.beta(1, 9, size=600000)
    y = (rng.random(600000) < p).astype(int)
    size = rng.lognormal(0, 0.6, 600000)
    assert threshold.best_threshold(y, p, threshold.AMOUNT, size) == pytest.approx(1 / 5.5, abs=0.02)


def test_the_expected_amount_cost_of_true_probabilities_matches_the_realised_cost():
    rng = np.random.default_rng(10)
    p = rng.beta(1, 9, size=400000)
    y = (rng.random(400000) < p).astype(int)
    size = rng.lognormal(0, 0.6, 400000)
    row = threshold.rule_row(y, p, 1 / 5.5, threshold.AMOUNT, size)
    assert row["expected_cost_per_1000"] == pytest.approx(row["cost_per_1000"], rel=0.03)
