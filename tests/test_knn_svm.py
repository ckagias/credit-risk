import numpy as np
import pandas as pd
import pytest

import forest
import knn_svm


@pytest.fixture
def parts():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    logit = -1.5 + 1.5 * X["f0"] - 1.0 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return forest.prepare_folds(X, y, scale=True, clip=True)


def test_knn_probability_is_the_share_of_defaulters_among_the_k_nearest():
    Z = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([1, 0, 0, 1, 1])
    model = knn_svm.make_knn(3).fit(Z, y)
    p = model.predict_proba(np.array([[1.2], [10.4]]))[:, 1]
    assert p == pytest.approx([1 / 3, 2 / 3])


def test_subsample_keeps_the_default_rate_and_returns_the_requested_size(parts):
    Z, y = knn_svm.subsample(parts[0], 400)
    assert len(Z) == len(y) == 400
    assert y.mean() == pytest.approx(parts[0]["y_fit"].mean(), abs=0.01)


def test_subsample_larger_than_the_fit_fold_returns_the_whole_fold(parts):
    Z, y = knn_svm.subsample(parts[0], 10**6)
    assert len(y) == len(parts[0]["y_fit"])


def test_subsample_is_the_same_for_the_same_seed_and_differs_for_another(parts):
    a, _ = knn_svm.subsample(parts[0], 300, seed=1)
    b, _ = knn_svm.subsample(parts[0], 300, seed=1)
    c, _ = knn_svm.subsample(parts[0], 300, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_timed_score_reports_scores_and_both_times(parts):
    part = parts[0]
    scored = knn_svm.timed_score(knn_svm.make_logistic(), part["Z_fit"], part["y_fit"], part["Z_val"], part["y_val"])
    assert {"pr_auc", "roc_auc", "log_loss", "fit_seconds", "predict_seconds"} <= set(scored)
    assert scored["roc_auc"] > 0.7 and scored["fit_seconds"] >= 0


def test_svm_probabilities_are_between_0_and_1_and_rank_better_than_chance(parts):
    part = parts[0]
    Z, y = knn_svm.subsample(part, 500)
    p = knn_svm.make_svm(Z, 1.0).fit(Z, y).predict_proba(part["Z_val"])[:, 1]
    assert p.min() >= 0 and p.max() <= 1
    assert knn_svm.evaluate.roc_auc(part["y_val"], p) > 0.65


def test_sweeps_give_one_row_per_fold_and_setting(parts):
    rows = knn_svm.knn_sweep(parts, k_grid=[5, 25], n=300)
    assert len(rows) == 5 * 2
    assert {r["n_fit"] for r in rows} == {300}
    rows = knn_svm.svm_sweep(parts[:2], c_grid=[0.1, 1], gamma_grid=[0.1, 1.0], n=300)
    assert len(rows) == 2 * 2 * 2
    assert {r["setting"] for r in rows} == {"C=0.1, gamma x0.1", "C=0.1, gamma x1", "C=1, gamma x0.1", "C=1, gamma x1"}


def test_size_curve_scores_three_models_at_every_size(parts):
    rows = pd.DataFrame(knn_svm.size_curve(parts[:2], k=15, C=1.0, gamma_multiplier=1.0, sizes=[200, 600]))
    assert len(rows) == 2 * 2 * 3
    assert set(rows["model"]) == {"knn", "svm", "logistic"}


def test_a_larger_training_sample_improves_the_logistic_score(parts):
    rows = pd.DataFrame(knn_svm.size_curve(parts, k=15, C=1.0, gamma_multiplier=1.0, sizes=[100, 1000]))
    pr = rows[rows["model"] == "logistic"].groupby("n_fit")["pr_auc"].mean()
    assert pr.loc[1000] > pr.loc[100]


def test_linear_svm_rows_include_the_logistic_reference_on_the_whole_fit_fold(parts):
    rows = pd.DataFrame(knn_svm.linear_svm_rows(parts[:1], c_grid=[0.01]))
    assert set(rows["model"]) == {"linear_svm", "logistic"}
    assert (rows["n_fit"] == len(parts[0]["y_fit"])).all()


def test_best_setting_picks_the_highest_mean_pr_auc():
    rows = [
        {"experiment": "e", "setting": 1, "pr_auc": 0.2},
        {"experiment": "e", "setting": 1, "pr_auc": 0.3},
        {"experiment": "e", "setting": 2, "pr_auc": 0.4},
        {"experiment": "other", "setting": 3, "pr_auc": 0.9},
    ]
    assert knn_svm.best_setting(rows, "e") == 2


def test_svm_setting_text_can_be_read_back_into_c_and_gamma_multiplier():
    assert knn_svm.parse_svm_setting(knn_svm.svm_setting(10, 0.01)) == (10.0, 0.01)
    assert knn_svm.parse_svm_setting(knn_svm.svm_setting(0.1, 1.0)) == (0.1, 1.0)


def test_gamma_multiplier_one_is_the_scale_rule_of_scikit_learn(parts):
    Z = parts[0]["Z_fit"]
    model = knn_svm.make_svm(Z, 1.0, 1.0)
    assert model.estimator.gamma == pytest.approx(1 / (Z.shape[1] * Z.var()))


def test_unweighted_svm_rows_use_the_given_setting_and_one_row_per_fold(parts):
    rows = knn_svm.svm_unweighted_rows(parts[:2], C=1.0, gamma_multiplier=0.1, n=300)
    assert len(rows) == 2
    assert {r["experiment"] for r in rows} == {"svm_unweighted"}
    assert {r["setting"] for r in rows} == {"C=1, gamma x0.1"}


def test_the_svm_uses_balanced_class_weights_unless_told_otherwise(parts):
    Z = parts[0]["Z_fit"]
    assert knn_svm.make_svm(Z).estimator.class_weight == "balanced"
    assert knn_svm.make_svm(Z, class_weight=None).estimator.class_weight is None


def test_ext_source_rows_use_only_those_columns_and_include_the_logistic_reference():
    rng = np.random.default_rng(1)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[*knn_svm.EXT_COLUMNS, "OTHER"])
    y = (rng.random(n) < 1 / (1 + np.exp(1.5 + X["EXT_SOURCE_1"]))).astype(int)
    rows = pd.DataFrame(knn_svm.ext_source_rows(X, y, k_grid=[5, 25], n=300))
    assert len(rows) == 5 * 3
    assert set(rows["model"]) == {"knn", "logistic"}
    assert set(rows["experiment"]) == {"ext_columns"}
