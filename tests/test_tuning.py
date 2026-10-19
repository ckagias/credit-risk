import numpy as np
import optuna
import pandas as pd
import pytest

import tuning
from forest import prepare_folds


@pytest.fixture(scope="module")
def parts():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = -2.0 + 1.5 * X["f0"] - 1.0 * X["f1"] + 0.8 * X["f0"] * X["f2"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return prepare_folds(X, y), X, y


def test_the_first_trial_is_the_default_setting(parts):
    study = tuning.tune(parts[0], n_trials=3)
    assert study.trials[0].params == tuning.DEFAULTS


def test_a_study_runs_the_requested_number_of_trials_and_each_value_is_the_mean_pr_auc(parts):
    study = tuning.tune(parts[0], n_trials=4, n_startup_trials=100)
    assert len(study.trials) == 4
    first = study.trials[0]
    per_fold = [tuning.fold_scores(part, first.params)["pr_auc"] for part in parts[0]]
    assert first.value == pytest.approx(np.mean(per_fold))
    assert first.user_attrs["pr_auc_std"] == pytest.approx(np.std(per_fold, ddof=1))


def test_sampled_settings_stay_inside_their_ranges(parts):
    study = tuning.tune(parts[0], n_trials=6, n_startup_trials=100)
    for trial in study.trials[1:]:
        assert 8 <= trial.params["num_leaves"] <= 256
        assert 10 <= trial.params["min_child_samples"] <= 300
        assert 0.5 <= trial.params["subsample"] <= 1.0
        assert 0.3 <= trial.params["colsample_bytree"] <= 1.0
        assert 1e-3 <= trial.params["reg_alpha"] <= 10.0


def test_the_same_seed_gives_the_same_settings(parts):
    a = tuning.tune(parts[0], n_trials=4, n_startup_trials=100)
    b = tuning.tune(parts[0], n_trials=4, n_startup_trials=100)
    assert [t.params for t in a.trials] == [t.params for t in b.trials]


def test_pruned_trials_are_marked_and_only_ran_some_folds(parts):
    study = tuning.tune(parts[0], n_trials=14, n_startup_trials=2)
    table = tuning.trials_table(study)
    pruned = table[table["state"] == "PRUNED"]
    complete = table[table["state"] == "COMPLETE"]
    assert (complete["folds_run"] == 5).all()
    assert (pruned["folds_run"] < 5).all()


def test_the_trials_table_has_one_row_per_trial_with_settings_and_scores(parts):
    study = tuning.tune(parts[0], n_trials=3, n_startup_trials=100)
    table = tuning.trials_table(study)
    assert len(table) == 3
    assert {"number", "value", "state", "num_leaves", "reg_lambda", "pr_auc_std", "best_round", "duration"} <= set(table.columns)


def test_subsampling_turns_on_only_when_the_fraction_is_below_one():
    assert tuning.lightgbm_params({**tuning.DEFAULTS, "subsample": 1.0})["subsample_freq"] == 0
    assert tuning.lightgbm_params({**tuning.DEFAULTS, "subsample": 0.7})["subsample_freq"] == 1


def test_the_fresh_fold_check_scores_default_and_tuned_on_other_folds(parts):
    _, X, y = parts
    check = tuning.check_on_fresh_folds(X, y, tuning.DEFAULTS, rates=[0.3])
    assert set(check["model"]) == {"default", "tuned"}
    assert len(check) == 2 * 5
    same = check.pivot(index="fold", columns="model", values="pr_auc")
    assert same["default"].tolist() == pytest.approx(same["tuned"].tolist())


def test_a_stored_study_resumes_up_to_the_requested_total(parts, tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    tuning.tune(parts[0], n_trials=2, storage=storage, n_startup_trials=100)
    study = tuning.tune(parts[0], n_trials=4, storage=storage, n_startup_trials=100)
    assert len(study.trials) == 4
    assert [t.number for t in study.trials] == [0, 1, 2, 3]
