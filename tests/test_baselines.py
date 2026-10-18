import sqlite3

import numpy as np
import pandas as pd
import pytest

import baselines
from split import DB, load_part, load_split

needs_db = pytest.mark.skipif(not DB.exists(), reason="data/home_credit.db is not built")


def test_ext_source_mean_skips_missing_scores():
    X = pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.2, np.nan, np.nan],
            "EXT_SOURCE_2": [np.nan, 0.5, np.nan],
            "EXT_SOURCE_3": [0.6, 0.7, np.nan],
        }
    )
    mean = baselines.ext_source_mean(X)
    assert mean.shape == (3, 1)
    assert mean[0, 0] == pytest.approx(0.4)
    assert mean[1, 0] == pytest.approx(0.6)
    assert np.isnan(mean[2, 0])


def test_ext_source_model_gives_a_lower_risk_to_a_higher_score():
    rng = np.random.default_rng(0)
    score = rng.random(4000)
    y = (rng.random(4000) < 0.25 - 0.2 * score).astype(int)
    X = pd.DataFrame({"EXT_SOURCE_1": score, "EXT_SOURCE_2": np.nan, "EXT_SOURCE_3": np.nan})
    model = baselines.ext_source_model().fit(X, y)
    low, high = model.predict_proba(pd.DataFrame({"EXT_SOURCE_1": [0.1, 0.9], "EXT_SOURCE_2": np.nan, "EXT_SOURCE_3": np.nan}))[:, 1]
    assert high < low


def test_ext_source_model_can_score_an_applicant_with_no_scores():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((500, 3)), columns=baselines.EXT_SOURCES)
    y = (rng.random(500) < 0.1).astype(int)
    model = baselines.ext_source_model().fit(X, y)
    blank = pd.DataFrame({c: [np.nan] for c in baselines.EXT_SOURCES})
    assert 0 < model.predict_proba(blank)[0, 1] < 1


@needs_db
def test_load_part_returns_exactly_the_applicants_of_that_part():
    con = sqlite3.connect(DB)
    split = load_split(con)
    train = load_part(con, "train", baselines.EXT_SOURCES)
    holdout = load_part(con, "holdout", baselines.EXT_SOURCES)
    con.close()
    assert set(train["SK_ID_CURR"]) == set(split.loc[split["part"] == "train", "SK_ID_CURR"])
    assert set(holdout["SK_ID_CURR"]) == set(split.loc[split["part"] == "holdout", "SK_ID_CURR"])
    assert set(train["SK_ID_CURR"]).isdisjoint(holdout["SK_ID_CURR"])
    assert list(train.columns) == ["SK_ID_CURR", "TARGET", *baselines.EXT_SOURCES]
