import sqlite3

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import load_group
from split import load_part

ID_AND_LABEL = ["SK_ID_CURR", "TARGET"]
DAYS_EMPLOYED_PLACEHOLDER = 365243
NO_VALUE_CODES = ["XNA", "XAP"]
MIN_CATEGORY_SHARE = 0.001
CLIP_LOWER = 0.001
CLIP_UPPER = 0.999


def clean(X):
    X = X.copy()
    if "DAYS_EMPLOYED" in X:
        X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].where(X["DAYS_EMPLOYED"] != DAYS_EMPLOYED_PLACEHOLDER)
    text = X.select_dtypes(exclude="number").columns
    for column in text:
        X[column] = X[column].astype(object).replace(NO_VALUE_CODES, np.nan)
    return X


def to_float32(X):
    return X.astype(np.float32)


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, lower=CLIP_LOWER, upper=CLIP_UPPER):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        binary = np.array([len(np.unique(column[~np.isnan(column)])) <= 2 for column in X.T])
        self.low_[binary] = -np.inf
        self.high_[binary] = np.inf
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def make_preprocessor(scale=True, clip=False):
    numeric_steps = [Winsorizer()] if clip else []
    numeric_steps.append(SimpleImputer(strategy="median", add_indicator=True))
    if scale:
        numeric_steps.append(StandardScaler())
    columns = ColumnTransformer(
        [
            ("num", make_pipeline(*numeric_steps), make_column_selector(dtype_include="number")),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=MIN_CATEGORY_SHARE,
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ],
        remainder="drop",
    )
    return make_pipeline(
        FunctionTransformer(clean, feature_names_out="one-to-one"),
        columns,
        FunctionTransformer(to_float32, feature_names_out="one-to-one"),
    )


def feature_columns(con):
    return [row[1] for row in con.execute("PRAGMA table_info(application_train)") if row[1] not in ID_AND_LABEL]


def load_features(con, part, groups=()):
    frame = load_part(con, part, feature_columns(con))
    for group in groups:
        frame = frame.merge(load_group(con, group, part), on="SK_ID_CURR", how="left", validate="one_to_one")
    return frame.drop(columns=ID_AND_LABEL), frame["TARGET"].to_numpy()
