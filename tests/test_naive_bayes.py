import numpy as np
import pandas as pd
import pytest

import forest
import knn_svm
import naive_bayes


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[*knn_svm.EXT_COLUMNS, "f3", "f4"])
    logit = -1.5 - 1.5 * X["EXT_SOURCE_1"] - 1.0 * X["EXT_SOURCE_2"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture
def parts(toy):
    X, y = toy
    return forest.prepare_folds(X, y, scale=True, clip=True)


def test_gaussian_bayes_posterior_matches_bayes_rule_worked_by_hand():
    Z = np.array([[0.0], [2.0], [1.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = naive_bayes.make_bayes(var_smoothing=0).fit(Z, y)
    x = 1.5
    mean = {0: 1.0, 1: 2.0}
    var = {0: 1.0, 1: 1.0}
    like = {c: np.exp(-((x - mean[c]) ** 2) / (2 * var[c])) / np.sqrt(2 * np.pi * var[c]) for c in (0, 1)}
    expected = like[1] * 0.5 / (like[1] * 0.5 + like[0] * 0.5)
    assert model.predict_proba(np.array([[x]]))[0, 1] == pytest.approx(expected)


def test_two_copies_of_a_column_count_the_evidence_twice():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.3).astype(int)
    a = rng.normal(loc=y * 1.0, size=4000)
    one = naive_bayes.make_bayes().fit(a.reshape(-1, 1), y)
    two = naive_bayes.make_bayes().fit(np.c_[a, a], y)
    logit = lambda p: np.log(p / (1 - p))
    prior = logit(y.mean())
    point = np.array([[1.8]])
    evidence_one = logit(one.predict_proba(point)[0, 1]) - prior
    evidence_two = logit(two.predict_proba(np.c_[point, point])[0, 1]) - prior
    assert evidence_two == pytest.approx(2 * evidence_one, rel=1e-6)


def test_each_fold_and_setting_gets_one_row_with_the_prediction_summary(parts):
    rows = pd.DataFrame(naive_bayes.sweep_and_calibrate(parts, "all", grid=[1e-9, 1e-2]))
    assert len(rows) == 5 * 2 + 5
    assert set(rows["experiment"]) == {"smoothing", "calibrated"}
    assert {"mean_p", "share_p_above_half", "fit_seconds", "pr_auc", "log_loss", "n_columns"} <= set(rows.columns)
    assert rows["mean_p"].between(0, 1).all()


def test_continuous_columns_drop_the_flags_and_the_one_hot_columns(parts):
    Z = parts[0]["Z_fit"]
    flags = np.c_[(np.arange(len(Z)) % 2).astype(np.float32), (np.arange(len(Z)) % 7 == 0).astype(np.float32)]
    mask = naive_bayes.continuous_mask(np.c_[Z, flags])
    assert mask[: Z.shape[1]].all() and not mask[Z.shape[1] :].any()
    kept = naive_bayes.continuous_parts(parts)
    assert kept[0]["Z_fit"].shape[1] == kept[0]["Z_val"].shape[1] == Z.shape[1]


def test_best_smoothing_is_the_setting_with_the_highest_mean_pr_auc():
    rows = [
        {"setting": 1e-9, "pr_auc": 0.1},
        {"setting": 1e-9, "pr_auc": 0.2},
        {"setting": 1e-2, "pr_auc": 0.3},
    ]
    assert naive_bayes.best_smoothing(rows) == 1e-2


def test_calibration_repairs_the_overconfidence_that_copied_columns_cause(toy):
    X, y = toy
    copies = pd.concat([X[["EXT_SOURCE_1", "EXT_SOURCE_2"]].add_suffix(f"_copy{i}") for i in range(6)], axis=1)
    part = forest.prepare_folds(pd.concat([X, copies], axis=1), y, scale=True, clip=True)[0]
    raw = naive_bayes.fold_row(part, "all", "e", 1e-9, naive_bayes.make_bayes())
    calibrated = naive_bayes.fold_row(part, "all", "e", 1e-9, naive_bayes.make_calibrated())
    assert raw["log_loss"] > calibrated["log_loss"] + 0.05
    assert raw["share_p_above_half"] > calibrated["share_p_above_half"]
    assert calibrated["roc_auc"] == pytest.approx(raw["roc_auc"], abs=0.02)
