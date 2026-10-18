import sqlite3

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import evaluate
from split import DB, load_part

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
THRESHOLD = 0.10


def ext_source_mean(X):
    return X[EXT_SOURCES].mean(axis=1).to_numpy().reshape(-1, 1)


def base_rate_model():
    return DummyClassifier(strategy="prior")


def ext_source_model():
    return make_pipeline(
        FunctionTransformer(ext_source_mean),
        SimpleImputer(strategy="median"),
        LogisticRegression(),
    )


def run_baselines(train):
    X = train[EXT_SOURCES]
    y = train["TARGET"].to_numpy()
    models = {"base rate for everyone": base_rate_model(), "EXT_SOURCE mean, logistic": ext_source_model()}
    results = {}
    for name, model in models.items():
        results[name] = evaluate.cross_validate(model, X, y, n_splits=5, seed=42, threshold=THRESHOLD)
    return results


def main():
    con = sqlite3.connect(DB)
    train = load_part(con, "train", EXT_SOURCES)
    con.close()
    print(f"train part: {len(train):,} applicants, default rate {train['TARGET'].mean():.4%}")
    for name, folds in run_baselines(train).items():
        print(f"\n{name} (5-fold CV, threshold {THRESHOLD} for precision and recall)")
        table = evaluate.summarize_folds(folds)
        table["min"] = folds[evaluate.METRICS].min()
        table["max"] = folds[evaluate.METRICS].max()
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()
