import numpy as np
import pandas as pd
import pytest

import forest
import imbalance


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    logit = -2.8 + 1.0 * X["a"] - 0.7 * X["b"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return forest.prepare_folds(X, y, scale=True, clip=True)


@pytest.mark.parametrize("method", ["undersample", "oversample", "smote"])
def test_resampling_gives_both_classes_the_same_count(toy, method):
    Z, y = imbalance.resample(method, toy[0]["Z_fit"], toy[0]["y_fit"])
    assert y.sum() == len(y) - y.sum()


def test_undersampling_keeps_every_defaulter_and_oversampling_keeps_every_good_loan(toy):
    y = toy[0]["y_fit"]
    _, y_under = imbalance.resample("undersample", toy[0]["Z_fit"], y)
    _, y_over = imbalance.resample("oversample", toy[0]["Z_fit"], y)
    assert y_under.sum() == y.sum()
    assert len(y_over) - y_over.sum() == len(y) - y.sum()


def test_smote_rows_lie_between_two_defaulters(toy):
    Z, y = toy[0]["Z_fit"], toy[0]["y_fit"]
    Z_new, y_new = imbalance.resample("smote", Z, y)
    synthetic = Z_new[len(y):]
    minority = Z[y == 1]
    assert (synthetic >= minority.min(axis=0) - 1e-9).all() and (synthetic <= minority.max(axis=0) + 1e-9).all()
    assert (y_new[len(y):] == 1).all()


def test_none_and_class_weight_leave_the_rows_alone(toy):
    for method in ("none", "class_weight"):
        Z, y = imbalance.resample(method, toy[0]["Z_fit"], toy[0]["y_fit"])
        assert Z is toy[0]["Z_fit"] and y is toy[0]["y_fit"]


def test_the_odds_correction_undoes_a_known_shift_of_the_prior():
    true_p = np.array([0.01, 0.05, 0.081, 0.3, 0.7])
    r_true = 0.088
    shifted_odds = true_p / (1 - true_p) / r_true
    shifted = shifted_odds / (1 + shifted_odds)
    assert imbalance.odds_correction(shifted, r_true, 1.0) == pytest.approx(true_p)
    assert imbalance.odds_correction(true_p, r_true, r_true) == pytest.approx(true_p)


def test_the_stopping_part_of_lightgbm_is_cut_before_the_rows_are_resampled(toy, monkeypatch):
    seen = []
    original = imbalance.resample

    def spy(method, Z, y, seed=imbalance.SEED):
        seen.append((len(y), int(y.sum())))
        return original(method, Z, y, seed)

    monkeypatch.setattr(imbalance, "resample", spy)
    part = toy[0]
    imbalance.fit_lightgbm(part["Z_fit"], part["y_fit"], "oversample", learning_rate=0.3)
    assert seen[0][0] == pytest.approx(0.9 * len(part["y_fit"]), abs=2)
    assert seen[0][1] / seen[0][0] == pytest.approx(part["y_fit"].mean(), abs=0.003)


@pytest.mark.parametrize("model_name", ["logistic", "lightgbm"])
def test_reweighting_raises_the_probabilities_and_the_odds_correction_repairs_the_log_loss(toy, model_name):
    none = imbalance.fold_row(toy[0], model_name, "none")
    weighted = imbalance.fold_row(toy[0], model_name, "class_weight")
    assert weighted["mean_probability"] > 2 * none["mean_probability"]
    assert weighted["share_above_half"] > 0.05 > none["share_above_half"]
    assert weighted["log_loss"] > none["log_loss"] + 0.03
    assert weighted["log_loss_corrected"] < weighted["log_loss"] - 0.03
    assert weighted["log_loss_corrected"] == pytest.approx(none["log_loss"], abs=0.03)


def test_the_ranking_barely_changes_with_class_weights_on_the_toy_table(toy):
    none = imbalance.fold_row(toy[0], "logistic", "none")
    for method in ("class_weight", "undersample", "oversample", "smote"):
        assert imbalance.fold_row(toy[0], "logistic", method)["roc_auc"] == pytest.approx(none["roc_auc"], abs=0.01)


def test_the_paired_difference_of_none_against_itself_is_zero(toy):
    table = imbalance.run(toy[:2], methods=["none", "class_weight"], models=["logistic"])
    difference = imbalance.paired_difference(table)
    assert difference.loc[("logistic", "none"), "difference"] == 0
    assert set(difference.index) == {("logistic", "none"), ("logistic", "class_weight")}
