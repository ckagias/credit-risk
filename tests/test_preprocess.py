import sqlite3

import numpy as np
import pandas as pd
import pytest

import preprocess as pp
from split import DB

needs_db = pytest.mark.skipif(not DB.exists(), reason="data/home_credit.db is not built")


def frame(x, days, color, kind):
    return pd.DataFrame({"x": x, "DAYS_EMPLOYED": days, "color": color, "kind": kind})


@pytest.fixture
def toy_train():
    return frame(
        x=[1.0, 2.0, 3.0, np.nan],
        days=[-100, 365243, -300, 365243],
        color=["red", "blue", "red", "XNA"],
        kind=["a", "b", "XAP", "a"],
    )


def column(pre, Z, name):
    return Z[:, list(pre.get_feature_names_out()).index(name)]


def test_clean_turns_the_placeholder_and_the_no_value_codes_into_missing(toy_train):
    cleaned = pp.clean(toy_train)
    assert cleaned["DAYS_EMPLOYED"].isna().tolist() == [False, True, False, True]
    assert cleaned["color"].isna().tolist() == [False, False, False, True]
    assert cleaned["kind"].isna().tolist() == [False, False, True, False]
    assert cleaned["x"].isna().tolist() == [False, False, False, True]


def test_clean_leaves_the_input_untouched(toy_train):
    before = toy_train.copy()
    pp.clean(toy_train)
    assert toy_train.equals(before)


def test_numeric_column_is_filled_with_the_training_median_and_scaled_by_hand(toy_train):
    pre = pp.make_preprocessor().fit(toy_train)
    Z = pre.transform(toy_train)
    scaled = column(pre, Z, "num__x")
    spread = np.sqrt(2 / 4)
    assert scaled.tolist() == pytest.approx([-1 / spread, 0, 1 / spread, 0], abs=1e-6)


def test_validation_rows_use_training_statistics_only(toy_train):
    pre = pp.make_preprocessor().fit(toy_train)
    validation = frame(x=[np.nan, 3.0, 1000.0], days=[-50, -50, -50], color=["red", "red", "red"], kind=["a", "a", "a"])
    Z = pre.transform(validation)
    spread = np.sqrt(2 / 4)
    assert column(pre, Z, "num__x").tolist() == pytest.approx([0, 1 / spread, 998 / spread], rel=1e-5)
    again = pp.make_preprocessor().fit(toy_train)
    assert column(again, again.transform(toy_train), "num__x").tolist() == pytest.approx(column(pre, pre.transform(toy_train), "num__x").tolist())


def test_the_placeholder_becomes_a_missing_flag_and_a_median(toy_train):
    pre = pp.make_preprocessor(scale=False).fit(toy_train)
    Z = pre.transform(toy_train)
    assert column(pre, Z, "num__missingindicator_DAYS_EMPLOYED").tolist() == [0, 1, 0, 1]
    assert column(pre, Z, "num__DAYS_EMPLOYED").tolist() == pytest.approx([-100, -200, -300, -200])


def test_no_value_codes_become_a_missing_category(toy_train):
    pre = pp.make_preprocessor(scale=False).fit(toy_train)
    Z = pre.transform(toy_train)
    assert column(pre, Z, "cat__color_missing").tolist() == [0, 0, 0, 1]
    assert column(pre, Z, "cat__kind_missing").tolist() == [0, 0, 1, 0]
    assert "cat__color_XNA" not in pre.get_feature_names_out()


def test_one_hot_columns_have_exactly_one_one_per_variable(toy_train):
    pre = pp.make_preprocessor(scale=False).fit(toy_train)
    Z = pre.transform(toy_train)
    names = list(pre.get_feature_names_out())
    for variable in ("color", "kind"):
        block = Z[:, [i for i, n in enumerate(names) if n.startswith(f"cat__{variable}_")]]
        assert block.sum(axis=1).tolist() == [1, 1, 1, 1]


def test_a_category_not_seen_in_training_does_not_break_transform(toy_train):
    pre = pp.make_preprocessor(scale=False).fit(toy_train)
    new = frame(x=[2.0], days=[-10], color=["green"], kind=["a"])
    Z = pre.transform(new)
    names = list(pre.get_feature_names_out())
    color_block = Z[:, [i for i, n in enumerate(names) if n.startswith("cat__color_")]]
    assert color_block.sum() == 0
    assert not np.isnan(Z).any()


def test_output_is_float32_without_gaps(toy_train):
    Z = pp.make_preprocessor().fit_transform(toy_train)
    assert Z.dtype == np.float32
    assert not np.isnan(Z).any() and not np.isinf(Z).any()


@needs_db
def test_real_training_part_has_no_gaps_and_the_expected_columns():
    con = sqlite3.connect(DB)
    X, y = pp.load_features(con, "train")
    con.close()
    pre = pp.make_preprocessor().fit(X, y)
    Z = pre.transform(X)
    names = pre.get_feature_names_out()
    assert Z.shape[0] == len(X) == 246008
    assert Z.dtype == np.float32
    assert not np.isnan(Z).any() and not np.isinf(Z).any()
    assert "num__missingindicator_DAYS_EMPLOYED" in names
    assert "cat__ORGANIZATION_TYPE_missing" in names
    assert not any(n.endswith("_XNA") for n in names)
    assert "SK_ID_CURR" not in " ".join(names) and "TARGET" not in " ".join(names)


def test_winsorizer_caps_values_at_the_training_percentiles_worked_out_by_hand():
    train = np.arange(1000, dtype=float).reshape(-1, 1)
    clipper = pp.Winsorizer().fit(train)
    assert clipper.low_[0] == pytest.approx(0.999)
    assert clipper.high_[0] == pytest.approx(998.001)
    out = clipper.transform(np.array([[-50.0], [500.0], [1e6]]))
    assert out.ravel().tolist() == pytest.approx([0.999, 500.0, 998.001])


def test_winsorizer_leaves_binary_columns_and_missing_values_alone():
    rare_flag = np.array([0.0] * 3000 + [1.0, 1.0])
    train = np.column_stack([rare_flag, np.arange(3002, dtype=float)])
    train[5, 1] = np.nan
    clipper = pp.Winsorizer().fit(train)
    assert np.quantile(rare_flag, pp.CLIP_UPPER) == 0.0
    out = clipper.transform(np.array([[1.0, np.nan], [0.0, 50000.0]]))
    assert out[0, 0] == 1.0 and out[1, 0] == 0.0
    assert np.isnan(out[0, 1])
    assert out[1, 1] == pytest.approx(clipper.high_[1])


def test_winsorizer_learns_its_bounds_from_the_training_rows_only():
    train = np.arange(1000, dtype=float).reshape(-1, 1)
    clipper = pp.Winsorizer().fit(train)
    before = (clipper.low_.copy(), clipper.high_.copy())
    clipper.transform(np.array([[1e9]]))
    assert clipper.low_ == before[0] and clipper.high_ == before[1]


def test_clip_option_caps_a_validation_outlier_in_the_pipeline():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"income": rng.normal(100, 10, 2000), "kind": ["a", "b"] * 1000})
    validation = pd.DataFrame({"income": [100.0, 1e8], "kind": ["a", "b"]})
    plain = pp.make_preprocessor().fit(train)
    clipped = pp.make_preprocessor(clip=True).fit(train)
    names = list(plain.get_feature_names_out())
    column = names.index("num__income")
    assert plain.transform(validation)[1, column] > 1e5
    assert clipped.transform(validation)[1, column] < 5
