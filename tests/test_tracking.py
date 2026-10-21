from datetime import datetime, timedelta

import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import holdout
import tracking

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return uri, tmp_path / "artifacts"


def tiny_study(n=5):
    study = optuna.create_study(study_name="lightgbm", direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0))

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        n_leaves = trial.suggest_int("num_leaves", 8, 64)
        for step in range(3):
            trial.report(x * (step + 1) / 3, step)
            if trial.should_prune():
                trial.set_user_attr("folds_run", step + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("folds_run", 3)
        trial.set_user_attr("roc_auc", 0.7 + x / 10)
        trial.set_user_attr("log_loss", 0.25)
        trial.set_user_attr("best_round", 100.0 + n_leaves)
        trial.set_user_attr("pr_auc_std", 0.004)
        return x

    study.optimize(objective, n_trials=n)
    return study


def test_a_trial_record_carries_the_parameters_the_metrics_the_folds_and_the_state():
    start = datetime(2026, 10, 1, 12, 0, 0)
    trial = optuna.trial.create_trial(
        params={"num_leaves": 31, "subsample": 0.9},
        distributions={"num_leaves": optuna.distributions.IntDistribution(8, 256), "subsample": optuna.distributions.FloatDistribution(0.5, 1.0)},
        value=0.243,
        user_attrs={"pr_auc_std": 0.005, "roc_auc": 0.757, "log_loss": 0.246, "best_round": 400.0, "folds_run": 5},
        intermediate_values={0: 0.23, 1: 0.24, 2: 0.243},
    )
    record = tracking.trial_record(trial, "test")
    assert record["params"] == {"num_leaves": "31", "subsample": "0.9"}
    metrics = {(key, step): value for key, value, step, _ in record["metrics"]}
    assert metrics[("pr_auc_mean", 0)] == pytest.approx(0.243)
    assert metrics[("roc_auc", 0)] == pytest.approx(0.757)
    assert metrics[("pr_auc_running_mean", 1)] == pytest.approx(0.24)
    assert record["tags"]["optuna_state"] == "COMPLETE" and record["tags"]["source"] == "test"
    del start


def test_backfilling_a_study_creates_one_parent_run_and_one_child_run_per_trial_with_the_same_numbers(store):
    uri, artifacts = store
    study = tiny_study(6)
    tracker = tracking.backfill(study, uri, artifacts)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(tracking.TUNING_EXPERIMENT).experiment_id
    runs = client.search_runs([experiment], max_results=100)
    children = [r for r in runs if r.data.tags.get("mlflow.parentRunId") == tracker.parent]
    assert len(runs) == 1 + len(study.trials) and len(children) == len(study.trials)
    by_trial = {int(r.data.tags["optuna_trial_number"]): r for r in children}
    for trial in study.trials:
        run = by_trial[trial.number]
        assert run.data.params == {name: str(value) for name, value in trial.params.items()}
        assert run.data.tags["optuna_state"] == trial.state.name
        assert run.data.metrics["pr_auc_mean"] == pytest.approx(trial.value)
        assert run.info.status == "FINISHED"
        history = client.get_metric_history(run.info.run_id, "pr_auc_running_mean")
        assert [m.step for m in history] == sorted(trial.intermediate_values)
    parent = client.get_run(tracker.parent)
    assert parent.data.metrics["trials"] == len(study.trials)
    assert parent.data.metrics["best_pr_auc_mean"] == pytest.approx(study.best_value)
    assert parent.data.tags["source"] == tracking.SOURCE_BACKFILL


def test_the_run_keeps_the_start_time_of_the_trial(store):
    uri, artifacts = store
    study = tiny_study(2)
    tracker = tracking.backfill(study, uri, artifacts)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(tracking.TUNING_EXPERIMENT).experiment_id
    runs = {int(r.data.tags["optuna_trial_number"]): r for r in client.search_runs([experiment]) if "optuna_trial_number" in r.data.tags}
    for trial in study.trials:
        assert runs[trial.number].info.start_time == tracking.milliseconds(trial.datetime_start)
    assert tracker.parent


def test_a_study_cannot_be_tracked_twice(store):
    uri, artifacts = store
    study = tiny_study(2)
    tracking.backfill(study, uri, artifacts)
    with pytest.raises(RuntimeError, match="already tracked"):
        tracking.backfill(study, uri, artifacts)


def test_live_tracking_logs_each_trial_as_it_finishes(store):
    uri, artifacts = store
    tracker = tracking.StudyTracker(uri, artifacts)
    tracker.start("live")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3, callbacks=[tracker.callback])
    tracker.finish(study)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(tracking.TUNING_EXPERIMENT).experiment_id
    children = [r for r in client.search_runs([experiment]) if r.data.tags.get("mlflow.parentRunId") == tracker.parent]
    assert len(children) == 3
    assert all(r.data.tags["source"] == tracking.SOURCE_LIVE for r in children)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "kind": rng.choice(["x", "y", "XNA"], n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.5 * X["a"])))).astype(int)
    return X, y


def test_the_final_pipeline_predicts_the_same_probabilities_as_the_model_scored_on_the_holdout(frame):
    X, y = frame
    pipeline = tracking.fit_final_pipeline(X, y)
    predict = holdout.fit_lightgbm(X, y)
    assert pipeline.predict_proba(X)[:, 1] == pytest.approx(predict(X), abs=1e-12)


def test_the_logged_model_is_registered_with_an_alias_and_gives_the_same_probabilities_when_loaded_back(store, frame):
    uri, artifacts = store
    X, y = frame
    pipeline = tracking.fit_final_pipeline(X, y)
    figure = artifacts.parent / "note.txt"
    figure.write_text("figure stand-in")
    run_id, difference = tracking.log_final_model(
        pipeline,
        X.iloc[:200],
        {"num_leaves": 9},
        {"holdout_pr_auc": 0.3},
        {"holdout": "copied"},
        [figure],
        uri=uri,
        artifacts=artifacts,
        pip_requirements=["scikit-learn", "lightgbm", "pandas"],
    )
    assert difference < 1e-9
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.params == {"num_leaves": "9"}
    assert run.data.metrics["holdout_pr_auc"] == pytest.approx(0.3)
    assert run.data.tags["holdout"] == "copied"
    assert "note.txt" in [a.path for a in client.list_artifacts(run_id)]
    assert client.get_model_version_by_alias(tracking.MODEL_NAME, tracking.ALIAS).run_id == run_id
    mlflow.set_tracking_uri(uri)
    loaded = mlflow.pyfunc.load_model(f"models:/{tracking.MODEL_NAME}@{tracking.ALIAS}")
    assert np.asarray(loaded.predict(X.iloc[:50])) == pytest.approx(pipeline.predict_proba(X.iloc[:50]))


def test_the_reported_metrics_are_copied_from_the_result_files_and_not_recomputed():
    gain = pd.DataFrame({"model": ["lightgbm"] * 3, "config": ["all groups, tuned"] * 3, "pr_auc": [0.27, 0.28, 0.29], "roc_auc": [0.78, 0.79, 0.80], "log_loss": [0.24] * 3})
    holdout_table = pd.DataFrame(
        {"model": ["LightGBM"] * 3, "metric": ["pr_auc", "roc_auc", "log_loss"], "estimate": [0.3, 0.79, 0.23], "low": [0.29, 0.78, 0.22], "high": [0.31, 0.80, 0.24]}
    )
    cost = pd.DataFrame([{"model": "LightGBM", "cost_per_1000": 277.9, "cost_refuse_nobody": 338.8, "saving": 0.18, "share_refused": 0.11, "defaulters_refused": 0.41}])
    metrics = tracking.reported_metrics(gain, holdout_table, cost)
    assert metrics["cv_pr_auc_mean"] == pytest.approx(0.28)
    assert metrics["cv_pr_auc_std"] == pytest.approx(0.01)
    assert metrics["holdout_pr_auc"] == 0.3 and metrics["holdout_pr_auc_low"] == 0.29 and metrics["holdout_pr_auc_high"] == 0.31
    assert metrics["holdout_saving"] == 0.18
