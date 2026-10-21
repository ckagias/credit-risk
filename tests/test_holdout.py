import numpy as np
import pandas as pd
import pytest

import holdout
import threshold


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 9, size=8000)
    y = (rng.random(8000) < p).astype(int)
    return y, p, rng.lognormal(0, 0.5, 8000)


def test_the_cost_row_matches_the_cost_functions_and_its_interval_contains_the_estimate(scored):
    y, p, size = scored
    row = holdout.cost_row(y, p, size, boots=100)
    direct = 1000 * threshold.cost_per_applicant(y, p, np.array([holdout.CUT]), threshold.AMOUNT, size)[0]
    assert row["cost_per_1000"] == pytest.approx(direct)
    assert row["saving"] == pytest.approx(1 - row["cost_per_1000"] / row["cost_refuse_nobody"])
    for name in ("cost_per_1000", "saving", "share_refused", "defaulters_refused"):
        assert row[f"{name}_low"] <= row[name] <= row[f"{name}_high"]


def test_the_holdout_threshold_is_the_cost_formula_fixed_in_advance():
    assert holdout.CUT == pytest.approx(1 / 5.5)


def test_run_scores_every_model_once_with_an_interval_on_each_metric_and_costs_the_boosting_model(scored):
    y, p, size = scored
    X = pd.DataFrame({"AMT_CREDIT": size * 500000})

    def fit_constant(X_train, y_train):
        rate = y_train.mean()
        return lambda X_new: np.full(len(X_new), rate)

    def fit_oracle_like(X_train, y_train):
        return lambda X_new: p

    table, cost = holdout.run(X, y, X, y, 500000.0, models={"base rate": fit_constant, "LightGBM": fit_oracle_like}, boots=50)
    assert set(table["model"]) == {"base rate", "LightGBM"}
    assert len(table) == 2 * len(holdout.SCORES)
    assert (table["low"] <= table["high"]).all()
    pr = table[table["metric"] == "pr_auc"].set_index("model")["estimate"]
    assert pr["LightGBM"] > pr["base rate"]
    assert list(cost["model"]) == ["LightGBM"]


def test_the_holdout_is_refused_a_second_time(tmp_path, monkeypatch):
    already = tmp_path / "holdout.csv"
    already.write_text("model,metric\n")
    monkeypatch.setattr(holdout, "RESULTS", already)
    with pytest.raises(SystemExit, match="not scored twice"):
        holdout.main()
