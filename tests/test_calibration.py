import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

import calibration
import evaluate
import forest


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    logit = -2.6 + 1.0 * X["a"] - 0.7 * X["b"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return forest.prepare_folds(X, y, scale=True, clip=True)


def test_brier_matches_scikit_learn():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(int)
    p = rng.random(500)
    assert calibration.brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_reliability_bins_hold_equal_counts_and_the_overall_means_match():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    table = calibration.reliability(y, p, bins=10)
    assert (table["n"] == 100).all()
    assert (table["n"] * table["mean_predicted"]).sum() / 1000 == pytest.approx(p.mean())
    assert (table["n"] * table["observed"]).sum() / 1000 == pytest.approx(y.mean())
    assert (np.diff(table["mean_predicted"]) >= 0).all()


def test_expected_calibration_error_is_small_for_true_probabilities_and_large_for_shifted_ones():
    rng = np.random.default_rng(3)
    p = rng.beta(1, 8, size=50000)
    y = (rng.random(50000) < p).astype(int)
    assert calibration.expected_calibration_error(y, p) < 0.005
    shifted = 1 / (1 + np.exp(-(calibration.logit(p) + 2.0)))
    assert calibration.expected_calibration_error(y, shifted) > 0.1


def test_platt_scaling_repairs_a_shift_of_the_log_odds_and_keeps_the_order():
    rng = np.random.default_rng(4)
    p = rng.beta(1, 8, size=40000)
    y = (rng.random(40000) < p).astype(int)
    shifted = 1 / (1 + np.exp(-(calibration.logit(p) + 2.0)))
    platt = calibration.Platt().fit(shifted, y)
    repaired = platt.predict(shifted)
    assert repaired.mean() == pytest.approx(y.mean(), abs=0.002)
    assert evaluate.log_loss(y, repaired) < evaluate.log_loss(y, shifted) - 0.1
    assert evaluate.roc_auc(y, repaired) == pytest.approx(evaluate.roc_auc(y, shifted), abs=1e-9)
    assert platt.model_.coef_[0, 0] == pytest.approx(1.0, abs=0.08)
    assert platt.model_.intercept_[0] == pytest.approx(-2.0, abs=0.12)


def test_isotonic_calibration_is_a_non_decreasing_step_function_inside_zero_and_one():
    rng = np.random.default_rng(5)
    p = rng.random(3000)
    y = (rng.random(3000) < p**2).astype(int)
    iso = calibration.Isotonic().fit(p, y)
    grid = np.linspace(-0.1, 1.1, 500)
    out = iso.predict(grid)
    assert (np.diff(out) >= -1e-12).all()
    assert out.min() >= 0 and out.max() <= 1


def test_the_calibration_rows_are_not_used_to_fit_the_model(toy):
    Z_model, Z_cal, y_model, y_cal = calibration.calibration_split(toy[0])
    assert len(y_model) + len(y_cal) == len(toy[0]["y_fit"])
    assert len(y_cal) == pytest.approx(0.2 * len(toy[0]["y_fit"]), abs=2)
    assert y_cal.mean() == pytest.approx(y_model.mean(), abs=0.003)


def test_every_model_is_scored_with_the_three_calibrators_and_the_ranking_is_kept_by_platt(toy):
    table, predictions = calibration.run(toy[:1], models=["logistic", "lightgbm_weighted"])
    assert len(table) == 2 * 3
    assert len(predictions) == 2 * 3 * len(toy[0]["y_val"])
    weighted = table[table["model"] == "lightgbm_weighted"].set_index("calibrator")
    assert weighted.loc["platt", "roc_auc"] == pytest.approx(weighted.loc["none", "roc_auc"], abs=1e-6)
    assert weighted.loc["platt", "log_loss"] < weighted.loc["none", "log_loss"] - 0.03
    assert weighted.loc["platt", "ece"] < weighted.loc["none", "ece"] / 3


def test_pooled_curves_give_one_row_per_bin_for_each_model_and_calibrator(toy):
    _, predictions = calibration.run(toy[:1], models=["logistic"])
    curves = calibration.pooled_curves(predictions, bins=5)
    assert len(curves) == 3 * 5
    assert set(curves["calibrator"]) == {"none", "platt", "isotonic"}


def test_fold_predictions_can_return_the_fitted_calibrators(toy):
    predictions, calibrators = calibration.fold_predictions(toy[0], "logistic", with_calibrators=True)
    assert set(calibrators) == {"platt", "isotonic"}
    assert set(predictions) == {"none", "platt", "isotonic"}
    assert hasattr(calibrators["platt"], "model_")
