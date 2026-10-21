import numpy as np
import pandas as pd
import pytest

import fairness


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 9, size=30000)
    y = (rng.random(30000) < p).astype(int)
    return y, p


def test_the_group_rates_match_a_direct_count(scored):
    y, p = scored
    row = fairness.group_row(y, p, cut=0.1)
    refused = p >= 0.1
    assert row["approval_rate"] == pytest.approx(1 - refused.mean())
    assert row["tpr"] == pytest.approx(refused[y == 1].mean())
    assert row["fpr"] == pytest.approx(refused[y == 0].mean())
    assert row["precision"] == pytest.approx(y[refused].mean())
    assert row["approval_low"] < row["approval_rate"] < row["approval_high"]
    assert 0.5 < row["roc_auc"] < 1.0


def test_two_groups_scored_by_the_same_rule_with_the_same_scores_have_no_gap(scored):
    y, p = scored
    half = len(y) // 2
    row = fairness.gap_row(y[:half], p[:half], y[half:], p[half:], cut=0.1, boots=300)
    for name in ("approval_rate", "tpr", "fpr"):
        assert row[f"{name}_gap_low"] < 0 < row[f"{name}_gap_high"]
    assert row["approval_ratio_low"] < 1 < row["approval_ratio_high"]


def test_a_group_with_scores_pushed_up_is_approved_less_and_the_interval_excludes_zero(scored):
    y, p = scored
    shifted = np.clip(p * 2, 0, 1)
    row = fairness.gap_row(y, shifted, y, p, cut=0.1, boots=300)
    assert row["approval_rate_gap"] < 0 and row["approval_rate_gap_high"] < 0
    assert row["approval_ratio"] < 1 and row["approval_ratio_high"] < 1
    assert row["fpr_gap"] > 0 and row["fpr_gap_low"] > 0


def test_the_gap_point_estimate_is_the_difference_of_the_two_groups_rates(scored):
    y, p = scored
    a, b = slice(0, 10000), slice(10000, 30000)
    row = fairness.gap_row(y[a], p[a], y[b], p[b], cut=0.1, boots=10)
    ra, rb = fairness.rates(y[a], p[a], 0.1), fairness.rates(y[b], p[b], 0.1)
    assert row["approval_rate_gap"] == pytest.approx(ra[0] - rb[0])
    assert row["tpr_gap"] == pytest.approx(ra[1] - rb[1])
    assert row["fpr_gap"] == pytest.approx(ra[2] - rb[2])
    assert row["approval_ratio"] == pytest.approx(ra[0] / rb[0])


def test_the_gender_labels_leave_out_the_unknown_code_and_the_tables_skip_it():
    X = pd.DataFrame(
        {
            "CODE_GENDER": ["F", "M", "XNA", "F"],
            "DAYS_BIRTH": [-12000.0, -15000, -20000, -9000],
            "AMT_INCOME_TOTAL": [1e5, 2e5, 1.5e5, 1.2e5],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 4,
            "bur_n_credits": [1, 0, 2, 3],
            "app_ext_n_known": [3, 2, 1, 3],
        }
    )
    label_map = fairness.labels(X)
    assert label_map["gender"].isna().tolist() == [False, False, True, False]
    frame = pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.1, 0.5, 0.2, 0.3]})
    table = fairness.group_table(frame, {"gender": label_map["gender"]}, "m")
    assert list(table["group"]) == ["men", "women"]
    assert table["n"].tolist() == [1, 2]
