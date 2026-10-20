import numpy as np
import pandas as pd
import pytest

import feature_gain as fg


def results(model="lightgbm", gains=(0.0, 0.01, 0.02), base=0.2):
    rows = []
    for config, gain in zip((fg.BASE, "bureau", fg.ALL), gains):
        for fold in range(5):
            rows.append(
                {
                    "model": model,
                    "config": config,
                    "fold": fold,
                    "n_features": 300 + 10 * len(rows) % 3,
                    "pr_auc": base + gain + 0.001 * fold,
                    "roc_auc": 0.7 + gain,
                    "log_loss": 0.25 - gain,
                }
            )
    return pd.DataFrame(rows)


def test_the_gain_is_the_paired_fold_difference_against_the_base():
    table = fg.gain_table(results()).set_index("config")
    assert table.loc[fg.BASE, "pr_auc_gain"] == pytest.approx(0.0)
    assert table.loc["bureau", "pr_auc_gain"] == pytest.approx(0.01)
    assert table.loc[fg.ALL, "pr_auc_gain"] == pytest.approx(0.02)
    assert table.loc["bureau", "log_loss_gain"] == pytest.approx(-0.01)
    assert table.loc["bureau", "pr_auc_gain_std"] == pytest.approx(0.0, abs=1e-12)


def test_paired_differences_ignore_the_fold_to_fold_level_but_the_plain_std_does_not():
    table = fg.gain_table(results()).set_index("config")
    assert table.loc["bureau", "pr_auc_std"] == pytest.approx(np.std([0.001 * k for k in range(5)], ddof=1))
    assert table.loc["bureau", "pr_auc_gain_std"] < table.loc["bureau", "pr_auc_std"]


def test_folds_better_counts_the_folds_where_pr_auc_rose():
    frame = results(gains=(0.0, 0.01, -0.01))
    frame.loc[(frame["config"] == "bureau") & (frame["fold"] == 0), "pr_auc"] -= 0.05
    table = fg.gain_table(frame).set_index("config")
    assert table.loc["bureau", "folds_better"] == 4
    assert table.loc[fg.ALL, "folds_better"] == 0


def test_each_model_is_compared_with_its_own_base():
    both = pd.concat([results("lightgbm", base=0.2), results("logistic", gains=(0.0, 0.03, 0.04), base=0.1)])
    table = fg.gain_table(both).set_index(["model", "config"])
    assert table.loc[("logistic", "bureau"), "pr_auc_gain"] == pytest.approx(0.03)
    assert table.loc[("lightgbm", "bureau"), "pr_auc_gain"] == pytest.approx(0.01)


def test_configs_are_the_base_each_group_alone_and_all_groups():
    listed = fg.configs(["a", "b"])
    assert listed == [(fg.BASE, []), ("a", ["a"]), ("b", ["b"]), (fg.ALL, ["a", "b"])]


def test_the_tuned_settings_come_from_the_best_completed_optuna_trial():
    params = fg.tuning_best_params()
    assert set(params) == set(fg.tuning.DEFAULTS)
    assert isinstance(params["num_leaves"], int) and params["num_leaves"] == 9
    assert params["reg_lambda"] == pytest.approx(7.5, abs=0.1)


def test_lightgbm_rows_and_logistic_rows_score_five_folds_on_a_toy_table():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.5 * X["a"] - X["b"])))).astype(int)
    boosted = fg.lightgbm_rows(X, y, fg.tuning.DEFAULTS, rate=0.3)
    assert boosted["fold"].tolist() == [0, 1, 2, 3, 4] and (boosted["pr_auc"] > y.mean()).all()
    linear = fg.logistic_rows(X, y)
    assert len(linear) == 5 and (linear["roc_auc"] > 0.7).all()
