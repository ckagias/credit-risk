import lightgbm as lgb
import numpy as np
import pytest

import explain


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 6000
    Z = rng.normal(size=(n, 4))
    logit = -2.0 + 1.5 * Z[:, 0] + 0.8 * Z[:, 1] * Z[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, num_leaves=8, random_state=0, verbose=-1).fit(Z[:4000], y[:4000])
    return model, Z[4000:], y[4000:], ["signal", "left", "right", "noise"]


def test_shap_values_plus_the_base_value_are_the_log_odds_of_the_model(fitted):
    model, Z, _, _ = fitted
    values, base = explain.shap_values(model, Z)
    assert values.shape == Z.shape
    assert values.sum(axis=1) + base == pytest.approx(model.predict(Z, raw_score=True), abs=1e-6)


def test_a_feature_the_model_never_uses_has_no_shap_value():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(3000, 3))
    y = (Z[:, 0] + rng.normal(scale=0.5, size=3000) > 0).astype(int)
    Z[:, 2] = 0.0
    model = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(Z, y)
    values, _ = explain.shap_values(model, Z)
    assert np.abs(values[:, 2]).max() == 0.0


def test_permutation_importance_ranks_the_signal_first_and_gives_the_noise_feature_nothing(fitted):
    model, Z, y, names = fitted
    table = explain.permutation_table(model, Z, y, names, repeats=5)
    assert table["feature"].iloc[0] == "signal"
    importance = table.set_index("feature")["importance_mean"]
    assert importance["signal"] > 5 * abs(importance["noise"])
    assert abs(importance["noise"]) < 0.02


def test_shap_table_sorts_by_mean_absolute_value_and_the_shares_add_to_one():
    values = np.array([[1.0, -3.0, 0.0], [-1.0, 3.0, 0.5]])
    table = explain.shap_table(values, ["a", "b", "c"])
    assert list(table["feature"]) == ["b", "a", "c"]
    assert table["mean_abs_shap"].tolist() == [3.0, 1.0, 0.25]
    assert table["share"].sum() == pytest.approx(1.0)


def test_the_picked_applicants_are_the_extremes_and_the_worst_misses():
    y = np.array([0, 1, 1, 0, 0, 1])
    p = np.array([0.30, 0.02, 0.60, 0.90, 0.01, 0.20])
    picked = explain.pick_applicants(y, p)
    assert picked == {"highest risk score": 3, "lowest risk score": 4, "defaulter scored lowest": 1, "repaid, scored highest": 3}


def test_each_applicant_explanation_adds_up_to_the_log_odds_when_every_feature_is_listed(fitted):
    model, Z, y, names = fitted
    p = model.predict_proba(Z)[:, 1]
    table = explain.applicant_rows(model, Z, y, p, names, top=4)
    assert len(table) == 4 * 4
    for _, rows in table.groupby("applicant"):
        assert rows["base_log_odds"].iloc[0] + rows["shap"].sum() == pytest.approx(rows["log_odds"].iloc[0])
        assert 1 / (1 + np.exp(-rows["log_odds"].iloc[0])) == pytest.approx(rows["probability"].iloc[0], abs=1e-6)


def test_agreement_is_one_for_the_same_ranking_and_counts_the_shared_top_features():
    import pandas as pd

    names = list("abcdef")
    permutation = pd.DataFrame({"feature": names, "importance_mean": [6, 5, 4, 3, 2, 1]})
    same = pd.DataFrame({"feature": names, "mean_abs_shap": [60, 50, 40, 30, 20, 10]})
    flipped = pd.DataFrame({"feature": names[::-1], "mean_abs_shap": [60, 50, 40, 30, 20, 10]})
    assert explain.agreement(permutation, same, top=3) == {"spearman": pytest.approx(1.0), "top_overlap": 3}
    assert explain.agreement(permutation, flipped, top=3) == {"spearman": pytest.approx(-1.0), "top_overlap": 0}
