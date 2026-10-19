import numpy as np
import pandas as pd
import pytest

import forest


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = -2.0 + 1.5 * X["f0"] - 1.0 * X["f1"] + 0.8 * X["f0"] * X["f2"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture
def parts(toy):
    X, y = toy
    return forest.prepare_folds(X, y)


def test_there_is_one_prepared_part_per_fold_and_nothing_is_shared(parts):
    assert [p["fold"] for p in parts] == [0, 1, 2, 3, 4]
    for part in parts:
        assert len(part["Z_fit"]) + len(part["Z_val"]) == 3000
        assert part["y_fit"].mean() == pytest.approx(part["y_val"].mean(), abs=0.01)


def test_the_mean_of_the_first_k_trees_is_a_forest_of_k_trees(parts):
    part = parts[0]
    big = forest.make_forest(n_estimators=20, max_features=3, min_samples_leaf=5).fit(part["Z_fit"], part["y_fit"])
    means = forest.running_means(forest.tree_probabilities(big, part["Z_val"]), [1, 7, 20])
    for k in (1, 7, 20):
        small = forest.make_forest(n_estimators=k, max_features=3, min_samples_leaf=5).fit(part["Z_fit"], part["y_fit"])
        assert means[k] == pytest.approx(small.predict_proba(part["Z_val"])[:, 1], abs=1e-6)


def test_tree_count_curve_has_one_row_per_fold_count_and_split(parts):
    rows = forest.tree_count_curve(parts, counts=[1, 5, 20], max_features=3, min_samples_leaf=5)
    assert len(rows) == 5 * 3 * 2
    assert {r["experiment"] for r in rows} == {"trees"}


def test_more_trees_improve_validation_pr_auc_and_log_loss(parts):
    rows = pd.DataFrame(forest.tree_count_curve(parts, counts=[1, 50], max_features=3, min_samples_leaf=10))
    validation = rows[rows["split"] == "validation"].groupby("setting")[["pr_auc", "log_loss"]].mean()
    assert validation.loc[50, "pr_auc"] > validation.loc[1, "pr_auc"] + 0.03
    assert validation.loc[50, "log_loss"] < validation.loc[1, "log_loss"]


def test_a_bigger_leaf_lowers_the_training_score(parts):
    rows = pd.DataFrame(forest.setting_curve(parts[:2], "min_samples_leaf", [1, 200], n_estimators=20, max_features=3))
    train = rows[rows["split"] == "train"].groupby("setting")["pr_auc"].mean()
    assert train.loc[1] > train.loc[200] + 0.2


def test_the_feature_sweep_reports_each_setting_for_each_fold_and_split(parts):
    rows = forest.setting_curve(parts, "max_features", [1, 3, 6], n_estimators=10, min_samples_leaf=5)
    assert len(rows) == 5 * 3 * 2
    assert forest.best_setting(rows, "max_features") in (1, 3, 6)
