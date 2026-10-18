import numpy as np
import pandas as pd
import pytest

import regularization as reg


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    logit = -2.0 + 1.2 * X["f0"] - 0.8 * X["f1"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_the_grid_is_cut_for_l1_only():
    assert max(reg.c_values("L2")) == 1.0
    assert max(reg.c_values("L1")) == reg.L1_MAX_C
    assert reg.c_values("L1") == [c for c in reg.c_values("L2") if c <= reg.L1_MAX_C]


def test_validation_curve_has_one_row_per_fold_setting_and_split(toy):
    X, y = toy
    curve = reg.validation_curve(X, y, n_jobs=1)
    settings = len(reg.c_values("L1")) + len(reg.c_values("L2"))
    assert len(curve) == 5 * settings * 2
    assert set(curve["split"]) == {"train", "validation"}
    assert sorted(curve["fold"].unique()) == [0, 1, 2, 3, 4]


def test_validation_curve_uses_the_same_folds_as_evaluate(toy):
    import evaluate
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from preprocess import make_preprocessor

    X, y = toy
    curve = reg.validation_curve(X, y, n_jobs=1)
    mine = curve[(curve["penalty"] == "L2") & (curve["C"] == 1.0) & (curve["split"] == "validation")].sort_values("fold")
    folds = evaluate.cross_validate(
        make_pipeline(make_preprocessor(), LogisticRegression(l1_ratio=0.0, max_iter=1000)), X, y, seed=reg.SEED, threshold=reg.THRESHOLD
    )
    assert mine["pr_auc"].tolist() == pytest.approx(folds["pr_auc"].tolist(), abs=1e-9)
    assert mine["roc_auc"].tolist() == pytest.approx(folds["roc_auc"].tolist(), abs=1e-9)


def test_a_stronger_penalty_shrinks_the_weights_and_l1_makes_them_zero(toy):
    X, y = toy
    path = reg.coefficient_path(X, y)
    size = path.assign(abs=path["coefficient"].abs()).groupby(["penalty", "C"])["abs"].sum()
    for penalty in ("L1", "L2"):
        sizes = size[penalty].sort_index().to_numpy()
        assert (np.diff(sizes) >= -1e-9).all()
    zeros = path[path["penalty"] == "L1"].assign(zero=lambda d: d["coefficient"] == 0).groupby("C")["zero"].sum()
    assert zeros.loc[min(reg.c_values("L1"))] > zeros.loc[max(reg.c_values("L1"))]
    assert (path[path["penalty"] == "L2"]["coefficient"] != 0).all()


def test_only_the_informative_columns_survive_a_strong_l1_penalty(toy):
    X, y = toy
    path = reg.coefficient_path(X, y)
    strong = path[(path["penalty"] == "L1") & (path["C"] == 1e-2)]
    assert set(strong.loc[strong["coefficient"] != 0, "feature"]) == {"num__f0", "num__f1"}
    weak = path[(path["penalty"] == "L1") & (path["C"] == 1e-1)]
    assert {"num__f0", "num__f1"} < set(weak.loc[weak["coefficient"] != 0, "feature"])
