import numpy as np
import pandas as pd
import pytest

import tree


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = -2.0 + 1.5 * X["f0"] - 1.0 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_the_depth_curve_has_one_row_per_fold_depth_and_split(toy):
    X, y = toy
    curve = tree.depth_curve(X, y, depths=[2, 5, None], n_jobs=1)
    assert len(curve) == 5 * 3 * 2
    assert set(curve["split"]) == {"train", "validation"}


def test_a_limited_tree_stops_at_its_depth_and_a_full_one_goes_deeper(toy):
    X, y = toy
    curve = tree.depth_curve(X, y, depths=[3, None], n_jobs=1)
    assert curve.loc[curve["max_depth"] == 3, "depth"].max() == 3
    assert curve.loc[curve["max_depth"].isna(), "depth"].min() > 10


def test_training_score_rises_with_depth_and_a_full_tree_memorizes_the_training_folds(toy):
    X, y = toy
    curve = tree.depth_curve(X, y, depths=[2, 6, None], n_jobs=1)
    train = curve[curve["split"] == "train"].groupby("max_depth", dropna=False)["pr_auc"].mean().sort_index()
    assert (np.diff(train.to_numpy()) > 0).all()
    assert train.iloc[-1] == pytest.approx(1.0)


def test_a_full_tree_scores_worse_on_validation_than_a_shallow_one(toy):
    X, y = toy
    curve = tree.depth_curve(X, y, depths=[3, None], n_jobs=1)
    validation = curve[curve["split"] == "validation"].groupby("max_depth", dropna=False)["pr_auc"].mean().sort_index()
    assert validation.iloc[0] > validation.iloc[-1]
