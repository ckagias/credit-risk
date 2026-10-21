import numpy as np
import pandas as pd
import pytest

import segments
import threshold


def brute_force(y, p, cut):
    refused = p >= cut
    return {
        "fnr": ((y == 1) & ~refused).sum() / (y == 1).sum(),
        "fpr": ((y == 0) & refused).sum() / (y == 0).sum(),
        "share_refused": refused.mean(),
    }


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 9, size=20000)
    y = (rng.random(20000) < p).astype(int)
    return y, p, rng.lognormal(0, 0.5, 20000)


def test_the_wilson_interval_of_five_in_ten_is_the_textbook_one():
    low, high = segments.interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-3)


def test_an_interval_on_no_trials_is_empty():
    assert all(np.isnan(segments.interval(0, 0)))


def test_the_segment_rates_match_a_direct_count(scored):
    y, p, size = scored
    row = segments.segment_row(y, p, size, cut=0.1, boots=0)
    expected = brute_force(y, p, 0.1)
    for key, value in expected.items():
        assert row[key] == pytest.approx(value)
    assert row["fnr_low"] < row["fnr"] < row["fnr_high"]
    assert row["fpr_low"] < row["fpr"] < row["fpr_high"]


def test_the_segment_cost_is_the_amount_weighted_cost_and_refusing_nobody_costs_the_missed_defaults(scored):
    y, p, size = scored
    row = segments.segment_row(y, p, size, cut=0.1, boots=0)
    flagged = p >= 0.1
    manual = 1000 * (4.5 * size[(y == 1) & ~flagged].sum() + size[(y == 0) & flagged].sum()) / len(y)
    assert row["cost_per_1000"] == pytest.approx(manual)
    assert row["cost_refuse_nobody"] == pytest.approx(1000 * 4.5 * size[y == 1].sum() / len(y))


def test_the_default_cut_is_the_bayes_threshold_of_the_amount_scheme():
    assert segments.CUT == pytest.approx(threshold.bayes_threshold(threshold.AMOUNT))
    assert segments.CUT == pytest.approx(1 / 5.5)


def test_the_bootstrap_interval_of_the_auc_contains_the_estimate(scored):
    y, p, size = scored
    row = segments.segment_row(y, p, size, boots=50)
    assert row["roc_auc_low"] < row["roc_auc"] < row["roc_auc_high"]
    assert 0.5 < row["roc_auc"] < 1.0


def test_age_bands_follow_the_edges_and_cover_every_applicant():
    X = pd.DataFrame(
        {
            "DAYS_BIRTH": [-20 * 365.25, -29.99 * 365.25, -30 * 365.25, -59.9 * 365.25, -60 * 365.25, -69 * 365.25],
            "AMT_INCOME_TOTAL": [60000.0, 90000, 120000, 150000, 200000, 400000],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 6,
            "bur_n_credits": [0, 1, 2, 0, 3, 4],
            "app_ext_n_known": [3, 2, 1, 0, 3, 3],
        }
    )
    labels = segments.segment_labels(X)
    assert list(labels["age"]) == ["under 30", "under 30", "30 to 39", "50 to 59", "60 and over", "60 and over"]
    assert list(labels["bureau record"]).count("no bureau record") == 2
    for family, label in labels.items():
        assert len(label) == len(X)
        assert pd.Series(np.asarray(label)).notna().all()


def test_income_bands_have_nearly_equal_sizes():
    income = pd.Series(np.random.default_rng(1).lognormal(11, 0.5, 5000))
    sizes = segments.income_band(income).value_counts()
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 5


def test_the_table_has_one_row_per_segment_and_a_row_for_everybody(scored):
    y, p, size = scored
    frame = pd.DataFrame({"y": y, "p": p})
    labels = {"half": np.where(np.arange(len(y)) % 2 == 0, "even", "odd")}
    table = segments.segment_table(frame, labels, size)
    assert list(table["segment"]) == ["even", "odd", "all applicants"]
    assert table["n"].tolist() == [10000, 10000, 20000]


def test_ordered_segments_keep_their_order_and_text_segments_are_sorted(scored):
    y, p, size = scored
    frame = pd.DataFrame({"y": y, "p": p})
    n = len(y)
    ordered = pd.Series(pd.Categorical(np.where(np.arange(n) % 2 == 0, "b low", "a high"), categories=["b low", "a high"], ordered=True))
    text = np.where(np.arange(n) % 2 == 0, "b low", "a high")
    table = segments.segment_table(frame, {"ordered": ordered, "text": text}, size)
    assert list(table["segment"]) == ["b low", "a high", "a high", "b low", "all applicants"]


def test_the_income_bands_run_from_low_to_high_and_the_top_band_is_open():
    income = pd.Series(np.random.default_rng(2).lognormal(11, 0.5, 5000))
    labels = segments.income_band(income)
    names = list(labels.cat.categories)
    assert len(names) == 5 and names[-1].endswith("and over")
    assert [int(n.split()[1].rstrip("k")) for n in names] == sorted(int(n.split()[1].rstrip("k")) for n in names)
