import sqlite3
import subprocess
import sys
from pathlib import Path

import lightgbm
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from mlflow.entities import Metric, Param, RunStatus, RunTag
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

import boosting
import features
import threshold
import tuning
from feature_gain import tuning_best_params
from preprocess import load_features, make_preprocessor
from split import DB

TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACTS = Path("mlartifacts")
TUNING_EXPERIMENT = "credit-risk-lightgbm-tuning"
FINAL_EXPERIMENT = "credit-risk-final-model"
MODEL_NAME = "credit-risk-lightgbm"
ALIAS = "champion"
STUDY_NAME = "lightgbm"
SOURCE_BACKFILL = "backfilled from the stored Optuna study results/optuna_lightgbm.db"
SOURCE_LIVE = "logged live while the study ran"
FIGURES = ["importance", "shap_applicants", "segments", "fairness", "holdout"]
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "numpy.dtype",
    "preprocess.clean",
    "preprocess.to_float32",
    "sklearn.compose._column_transformer.make_column_selector",
]
SAMPLE_ROWS = 2000
ROUNDTRIP_ROWS = 1000


def milliseconds(moment):
    return int(moment.timestamp() * 1000) if moment is not None else None


def trial_record(trial, source):
    start, end = milliseconds(trial.datetime_start), milliseconds(trial.datetime_complete)
    attrs = trial.user_attrs
    metrics = []
    if trial.value is not None:
        metrics.append(("pr_auc_mean", trial.value, 0, end))
    for key in ("pr_auc_std", "roc_auc", "log_loss", "best_round", "folds_run"):
        if key in attrs:
            metrics.append((key, float(attrs[key]), 0, end))
    if trial.duration is not None:
        metrics.append(("duration_seconds", trial.duration.total_seconds(), 0, end))
    for step, value in sorted(trial.intermediate_values.items()):
        metrics.append(("pr_auc_running_mean", float(value), int(step), end))
    return {
        "name": f"trial {trial.number:02d}",
        "params": {name: str(value) for name, value in trial.params.items()},
        "metrics": metrics,
        "tags": {"optuna_state": trial.state.name, "optuna_trial_number": str(trial.number), "source": source},
        "start": start,
        "end": end,
    }


def experiment_id(client, name, artifacts=ARTIFACTS):
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    return client.create_experiment(name, artifact_location=(Path(artifacts) / name).resolve().as_uri())


class StudyTracker:
    def __init__(self, uri=TRACKING_URI, artifacts=ARTIFACTS, source=SOURCE_LIVE):
        mlflow.set_tracking_uri(uri)
        self.client = MlflowClient(tracking_uri=uri)
        self.experiment = experiment_id(self.client, TUNING_EXPERIMENT, artifacts)
        self.source = source
        self.parent = None

    def start(self, study_name=STUDY_NAME):
        existing = self.client.search_runs([self.experiment], filter_string=f"tags.optuna_study = '{study_name}'")
        if existing:
            raise RuntimeError(f"the study '{study_name}' is already tracked in {TUNING_EXPERIMENT}")
        tags = {
            "optuna_study": study_name,
            "source": self.source,
            "sampler": f"TPE, seed {tuning.SEED}",
            "pruner": "MedianPruner, n_startup_trials 10, n_warmup_steps 1",
            "objective": "mean PR-AUC over 5 stratified folds of the training part, base application columns, learning rate 0.05, early stopping",
        }
        self.parent = self.client.create_run(self.experiment, run_name=f"optuna study '{study_name}'", tags=tags).info.run_id
        return self.parent

    def log_trial(self, trial):
        record = trial_record(trial, self.source)
        tags = {**record["tags"], "mlflow.parentRunId": self.parent}
        run = self.client.create_run(self.experiment, start_time=record["start"], tags=tags, run_name=record["name"])
        run_id = run.info.run_id
        self.client.log_batch(
            run_id,
            metrics=[Metric(key, value, timestamp or 0, step) for key, value, step, timestamp in record["metrics"]],
            params=[Param(name, value) for name, value in record["params"].items()],
        )
        self.client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED), end_time=record["end"])
        return run_id

    def callback(self, study, trial):
        self.log_trial(trial)

    def finish(self, study):
        trials = study.trials
        states = pd.Series([t.state.name for t in trials]).value_counts()
        summary = {"trials": len(trials), "trials_complete": int(states.get("COMPLETE", 0)), "trials_pruned": int(states.get("PRUNED", 0)), "best_pr_auc_mean": study.best_value}
        self.client.log_batch(
            self.parent,
            metrics=[Metric(key, float(value), 0, 0) for key, value in summary.items()],
            params=[Param(f"best_{name}", str(value)) for name, value in study.best_params.items()] + [Param("best_trial", str(study.best_trial.number))],
        )
        self.client.set_terminated(self.parent, RunStatus.to_string(RunStatus.FINISHED))


def backfill(study, uri=TRACKING_URI, artifacts=ARTIFACTS):
    tracker = StudyTracker(uri, artifacts, source=SOURCE_BACKFILL)
    tracker.start(study.study_name)
    for trial in study.trials:
        tracker.log_trial(trial)
    tracker.finish(study)
    return tracker


def fit_final_pipeline(X, y):
    pre = make_preprocessor(scale=False).fit(X)
    booster = boosting.fit_with_early_stopping(pre.transform(X), y, tuning.TUNE_RATE, **tuning.lightgbm_params(tuning_best_params()))
    return Pipeline([("preprocess", pre), ("model", booster)])


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    return commit + (" (uncommitted changes)" if dirty else "")


def reported_metrics(gain, holdout, cost, config="all groups, tuned"):
    tuned = gain[(gain["model"] == "lightgbm") & (gain["config"] == config)]
    metrics = {}
    for key in ("pr_auc", "roc_auc", "log_loss"):
        metrics[f"cv_{key}_mean"] = float(tuned[key].mean())
        metrics[f"cv_{key}_std"] = float(tuned[key].std())
    scored = holdout[holdout["model"] == "LightGBM"].set_index("metric")
    for key in ("pr_auc", "roc_auc", "log_loss"):
        metrics[f"holdout_{key}"] = float(scored.loc[key, "estimate"])
        metrics[f"holdout_{key}_low"] = float(scored.loc[key, "low"])
        metrics[f"holdout_{key}_high"] = float(scored.loc[key, "high"])
    row = cost[cost["model"] == "LightGBM"].iloc[0]
    for key in ("cost_per_1000", "cost_refuse_nobody", "saving", "share_refused", "defaulters_refused"):
        metrics[f"holdout_{key}"] = float(row[key])
    return metrics


def log_final_model(pipeline, X_sample, params, metrics, tags, files, uri=TRACKING_URI, artifacts=ARTIFACTS, register=True, code_paths=None, pip_requirements=None):
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = experiment_id(client, FINAL_EXPERIMENT, artifacts)
    probabilities = pipeline.predict_proba(X_sample)
    signature = infer_signature(X_sample, probabilities)
    with mlflow.start_run(experiment_id=experiment, run_name="final LightGBM") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=TRUSTED_TYPES,
            registered_model_name=MODEL_NAME if register else None,
            code_paths=code_paths,
            pip_requirements=pip_requirements,
        )
        for file in files:
            mlflow.log_artifact(str(file))
        reloaded = mlflow.pyfunc.load_model(info.model_uri).predict(X_sample.iloc[:ROUNDTRIP_ROWS])
        difference = float(np.abs(np.asarray(reloaded) - probabilities[:ROUNDTRIP_ROWS]).max())
        mlflow.log_metric("roundtrip_max_abs_difference", difference)
    if register:
        version = client.search_model_versions(f"name = '{MODEL_NAME}'")[0].version
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    return run.info.run_id, difference


def final_run():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train", features.GROUPS)
    con.close()
    pipeline = fit_final_pipeline(X, y)
    booster = pipeline.named_steps["model"]
    params = {
        **{name: value for name, value in tuning_best_params().items()},
        "learning_rate": tuning.TUNE_RATE,
        "early_stopping_patience": boosting.PATIENCE,
        "early_stopping_share": boosting.STOP_SHARE,
        "best_iteration": booster.best_iteration_,
        "feature_groups": ",".join(features.GROUPS),
        "raw_columns": X.shape[1],
        "model_inputs": booster.n_features_in_,
        "training_rows": len(y),
        "seed": tuning.SEED,
        "decision_threshold": round(threshold.bayes_threshold(threshold.AMOUNT), 4),
        "loss_given_default": threshold.LOSS_GIVEN_DEFAULT,
        "margin": threshold.MARGIN,
    }
    metrics = reported_metrics(pd.read_csv("results/feature_gain.csv"), pd.read_csv("results/holdout.csv"), pd.read_csv("results/holdout_cost.csv"))
    tags = {
        "git_commit": git_state(),
        "holdout": "scored once by src/holdout.py; the holdout metrics are copied from results/holdout.csv and were not recomputed here",
        "model_object": "refitted here with the settings of src/holdout.py, the holdout predictions were not kept so the two fits were not compared on the holdout",
        "tuned_on": "base application columns (300 model inputs), 60 Optuna trials; the settings are reused with the six SQL feature groups",
        "cost_assumption": "a missed default loses 45% of the credit amount and a refused good loan a 10% margin, both assumed",
    }
    files = [Path("docs/img") / f"{name}.png" for name in FIGURES] + [Path("results/holdout.csv"), Path("results/holdout_cost.csv"), Path("requirements.txt")]
    requirements = [f"scikit-learn=={sklearn.__version__}", f"lightgbm=={lightgbm.__version__}", f"pandas=={pd.__version__}", f"numpy=={np.__version__}"]
    sample = X.sample(SAMPLE_ROWS, random_state=tuning.SEED)
    run_id, difference = log_final_model(pipeline, sample, params, metrics, tags, files, code_paths=["src/preprocess.py"], pip_requirements=requirements)
    print(f"run {run_id}: {len(files)} files, round-trip difference {difference:.2e}")


def backfill_run():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{tuning.STUDY_DB}")
    tracker = backfill(study)
    print(f"{len(study.trials)} trials logged under run {tracker.parent}")


if __name__ == "__main__":
    {"backfill": backfill_run, "final": final_run}[sys.argv[1] if len(sys.argv) > 1 else "backfill"]()
