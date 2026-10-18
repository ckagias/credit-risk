import sqlite3

import pandas as pd
import pytest

from split import DB, HOLDOUT_SHARE, load_split, make_split

needs_db = pytest.mark.skipif(not DB.exists(), reason="data/home_credit.db is not built")


def ids_in_both_parts(split):
    parts = split.groupby("SK_ID_CURR")["part"].nunique()
    return parts[parts > 1].index.tolist()


def default_rate(split, applicants, part):
    ids = split.loc[split["part"] == part, "SK_ID_CURR"]
    return applicants.loc[applicants["SK_ID_CURR"].isin(ids), "TARGET"].mean()


@pytest.fixture
def toy_applicants():
    return pd.DataFrame({"SK_ID_CURR": range(100_000, 101_000), "TARGET": [1] * 80 + [0] * 920})


@pytest.fixture
def real():
    con = sqlite3.connect(DB)
    applicants = pd.read_sql("SELECT SK_ID_CURR, TARGET FROM application_train", con)
    split = load_split(con)
    con.close()
    return applicants, split


def test_toy_split_has_no_overlap_and_covers_everyone(toy_applicants):
    split = make_split(toy_applicants)
    assert ids_in_both_parts(split) == []
    assert len(split) == len(toy_applicants)
    assert set(split["SK_ID_CURR"]) == set(toy_applicants["SK_ID_CURR"])


def test_toy_split_sizes_and_default_rate_worked_out_by_hand(toy_applicants):
    split = make_split(toy_applicants)
    assert (split["part"] == "holdout").sum() == 200
    assert (split["part"] == "train").sum() == 800
    assert default_rate(split, toy_applicants, "holdout") == pytest.approx(0.08)
    assert default_rate(split, toy_applicants, "train") == pytest.approx(0.08)


def test_toy_split_is_the_same_on_every_run(toy_applicants):
    first = make_split(toy_applicants).reset_index(drop=True)
    second = make_split(toy_applicants.sample(frac=1, random_state=1).sort_values("SK_ID_CURR")).reset_index(drop=True)
    assert first.equals(second)


def test_overlap_check_catches_an_applicant_in_both_parts(toy_applicants):
    split = make_split(toy_applicants)
    first_id = split["SK_ID_CURR"].iloc[0]
    other_part = "holdout" if split.iloc[0]["part"] == "train" else "train"
    leaked = pd.concat([split, pd.DataFrame({"SK_ID_CURR": [first_id], "part": [other_part]})])
    assert ids_in_both_parts(leaked) == [first_id]


@needs_db
def test_saved_split_has_no_applicant_in_both_parts(real):
    applicants, split = real
    assert ids_in_both_parts(split) == []
    assert split["SK_ID_CURR"].is_unique
    assert set(split["SK_ID_CURR"]) == set(applicants["SK_ID_CURR"])


@needs_db
def test_saved_split_has_the_same_default_rate_in_both_parts(real):
    applicants, split = real
    train = default_rate(split, applicants, "train")
    holdout = default_rate(split, applicants, "holdout")
    assert abs(train - holdout) < 1e-4
    assert abs(train - applicants["TARGET"].mean()) < 1e-4


@needs_db
def test_saved_split_holdout_share(real):
    applicants, split = real
    share = (split["part"] == "holdout").mean()
    assert share == pytest.approx(HOLDOUT_SHARE, abs=1e-4)


@needs_db
def test_saved_split_matches_a_fresh_run(real):
    applicants, split = real
    fresh = make_split(applicants.sort_values("SK_ID_CURR")).reset_index(drop=True)
    saved = split.sort_values("SK_ID_CURR").reset_index(drop=True)
    assert fresh.equals(saved)
