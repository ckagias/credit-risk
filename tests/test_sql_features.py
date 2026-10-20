import sqlite3

import pandas as pd
import pytest

import features
from split import DB

needs_db = pytest.mark.skipif(not DB.exists(), reason="data/home_credit.db is not built")


def toy_connection(**tables):
    con = sqlite3.connect(":memory:")
    for name, rows in tables.items():
        pd.DataFrame(rows).to_sql(name, con, index=False)
    return con


def built(con, group):
    features.build_group(con, group)
    return pd.read_sql(f"SELECT * FROM {features.table_of(group)} ORDER BY SK_ID_CURR", con).set_index("SK_ID_CURR")


def credit(applicant, bureau_id, status, days, total, debt, overdue, max_overdue, day_overdue, end, prolong, kind):
    return {
        "SK_ID_CURR": applicant,
        "SK_ID_BUREAU": bureau_id,
        "CREDIT_ACTIVE": status,
        "DAYS_CREDIT": days,
        "AMT_CREDIT_SUM": total,
        "AMT_CREDIT_SUM_DEBT": debt,
        "AMT_CREDIT_SUM_OVERDUE": overdue,
        "AMT_CREDIT_MAX_OVERDUE": max_overdue,
        "CREDIT_DAY_OVERDUE": day_overdue,
        "DAYS_CREDIT_ENDDATE": end,
        "CNT_CREDIT_PROLONG": prolong,
        "CREDIT_TYPE": kind,
    }


@pytest.fixture
def bureau_db():
    return toy_connection(
        application_train=[{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}, {"SK_ID_CURR": 3}],
        bureau=[
            credit(1, 10, "Active", -100, 1000, 400, 0, 50, 0, 200, 0, "Consumer credit"),
            credit(1, 11, "Closed", -500, 2000, 0, 0, None, 0, -300, 1, "Credit card"),
            credit(1, 12, "Active", -30, 3000, 1500, 100, 700, 15, 60, 2, "Consumer credit"),
            credit(1, 13, "Bad debt", -800, 500, 500, 500, 500, 90, -700, 0, "Car loan"),
            credit(1, 14, "Closed", -2000, 100, 0, 0, 0, 0, -1900, 0, "Consumer credit"),
            credit(2, 20, "Active", -10, 0, 0, 0, 0, 0, 5, 0, "Credit card"),
        ],
        bureau_balance=[
            {"SK_ID_BUREAU": 10, "MONTHS_BALANCE": 0, "STATUS": "0"},
            {"SK_ID_BUREAU": 10, "MONTHS_BALANCE": -1, "STATUS": "1"},
            {"SK_ID_BUREAU": 10, "MONTHS_BALANCE": -2, "STATUS": "0"},
            {"SK_ID_BUREAU": 10, "MONTHS_BALANCE": -14, "STATUS": "2"},
            {"SK_ID_BUREAU": 10, "MONTHS_BALANCE": -15, "STATUS": "X"},
            {"SK_ID_BUREAU": 12, "MONTHS_BALANCE": -1, "STATUS": "0"},
            {"SK_ID_BUREAU": 12, "MONTHS_BALANCE": -2, "STATUS": "5"},
            {"SK_ID_BUREAU": 20, "MONTHS_BALANCE": -1, "STATUS": "C"},
            {"SK_ID_BUREAU": 99, "MONTHS_BALANCE": -1, "STATUS": "3"},
        ],
    )


def test_bureau_has_one_row_per_applicant_including_those_without_credits(bureau_db):
    table = built(bureau_db, "bureau")
    assert table.index.tolist() == [1, 2, 3]
    assert features.row_count_problems(bureau_db, "bureau") == []


def test_bureau_credit_counts_and_sums_for_an_applicant_with_five_credits(bureau_db):
    row = built(bureau_db, "bureau").loc[1]
    assert (row["bur_n_credits"], row["bur_n_active"], row["bur_n_closed"], row["bur_n_bad_or_sold"]) == (5, 2, 2, 1)
    assert row["bur_n_types"] == 3
    assert row["bur_n_last_year"] == 2
    assert (row["bur_days_oldest"], row["bur_days_newest"]) == (-2000, -30)
    assert (row["bur_credit_sum"], row["bur_debt_sum"], row["bur_overdue_sum"], row["bur_max_overdue"]) == (6600, 2400, 600, 700)
    assert (row["bur_active_credit_sum"], row["bur_active_debt_sum"]) == (4000, 1900)
    assert row["bur_active_debt_ratio"] == pytest.approx(0.475)
    assert row["bur_active_end_mean"] == pytest.approx(130)
    assert (row["bur_day_overdue_max"], row["bur_n_overdue_now"], row["bur_prolong_sum"]) == (90, 2, 3)


def test_the_three_most_recent_credits_are_picked_with_a_window_function(bureau_db):
    row = built(bureau_db, "bureau").loc[1]
    assert row["bur_last3_credit_mean"] == pytest.approx((3000 + 1000 + 2000) / 3)
    assert row["bur_last3_n_active"] == 2


def test_monthly_status_counts_skip_closed_and_unknown_months_and_orphan_credits(bureau_db):
    row = built(bureau_db, "bureau").loc[1]
    assert row["bur_bb_months"] == 7
    assert row["bur_bb_late_share"] == pytest.approx(3 / 6)
    assert row["bur_bb_late_last_year"] == 2
    assert row["bur_bb_worst_status"] == 5


def test_an_applicant_whose_only_months_are_closed_has_no_late_share(bureau_db):
    row = built(bureau_db, "bureau").loc[2]
    assert row["bur_bb_months"] == 1
    assert pd.isna(row["bur_bb_late_share"]) and pd.isna(row["bur_bb_worst_status"])
    assert row["bur_bb_late_last_year"] == 0


def test_a_zero_credit_sum_gives_a_missing_ratio_and_not_a_division_error(bureau_db):
    assert pd.isna(built(bureau_db, "bureau").loc[2, "bur_active_debt_ratio"])


def test_an_applicant_without_credits_has_zero_counts_and_missing_amounts(bureau_db):
    row = built(bureau_db, "bureau").loc[3]
    assert (row["bur_n_credits"], row["bur_n_active"], row["bur_bb_months"], row["bur_last3_n_active"]) == (0, 0, 0, 0)
    assert pd.isna(row["bur_credit_sum"]) and pd.isna(row["bur_last3_credit_mean"]) and pd.isna(row["bur_days_newest"])


def test_building_twice_gives_the_same_table(bureau_db):
    first = built(bureau_db, "bureau")
    second = built(bureau_db, "bureau")
    pd.testing.assert_frame_equal(first, second)


def previous(applicant, prev_id, kind, status, days, applied, granted, annuity, rate, term, termination):
    return {
        "SK_ID_PREV": prev_id,
        "SK_ID_CURR": applicant,
        "NAME_CONTRACT_TYPE": kind,
        "NAME_CONTRACT_STATUS": status,
        "DAYS_DECISION": days,
        "AMT_APPLICATION": applied,
        "AMT_CREDIT": granted,
        "AMT_ANNUITY": annuity,
        "RATE_DOWN_PAYMENT": rate,
        "CNT_PAYMENT": term,
        "DAYS_TERMINATION": termination,
    }


@pytest.fixture
def previous_db():
    return toy_connection(
        application_train=[{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}, {"SK_ID_CURR": 3}],
        previous_application=[
            previous(1, 1, "Cash loans", "Approved", -100, 1000, 900, 90, 0.0, 12, 365243),
            previous(1, 2, "Consumer loans", "Refused", -50, 2000, 0, None, None, None, None),
            previous(1, 3, "Consumer loans", "Approved", -400, 500, 500, 50, 0.2, 6, -300),
            previous(1, 4, "Revolving loans", "Canceled", -800, 300, None, None, None, None, None),
            previous(1, 5, "Cash loans", "Approved", -10, 4000, 3000, 300, None, 24, 365243),
            previous(2, 6, "Consumer loans", "Refused", -20, 700, 0, None, None, None, None),
        ],
    )


def test_previous_application_counts_by_status_and_type(previous_db):
    table = built(previous_db, "previous_application")
    assert table.index.tolist() == [1, 2, 3]
    row = table.loc[1]
    assert (row["prev_n"], row["prev_n_approved"], row["prev_n_refused"], row["prev_n_canceled"], row["prev_n_unused"]) == (5, 3, 1, 1, 0)
    assert row["prev_refused_share"] == pytest.approx(0.2)
    assert (row["prev_n_cash"], row["prev_n_consumer"], row["prev_n_revolving"]) == (2, 2, 1)
    assert row["prev_n_last_year"] == 3
    assert (row["prev_days_oldest"], row["prev_days_newest"]) == (-800, -10)


def test_previous_application_amounts_only_use_approved_applications(previous_db):
    row = built(previous_db, "previous_application").loc[1]
    assert row["prev_credit_sum"] == 4400
    assert row["prev_credit_mean"] == pytest.approx(4400 / 3)
    assert row["prev_credit_max"] == 3000
    assert row["prev_granted_ratio"] == pytest.approx((0.9 + 1.0 + 0.75) / 3)
    assert row["prev_annuity_mean"] == pytest.approx(440 / 3)
    assert row["prev_down_rate_mean"] == pytest.approx(0.1)
    assert row["prev_term_mean"] == pytest.approx(14)


def test_the_termination_placeholder_counts_loans_still_running(previous_db):
    assert built(previous_db, "previous_application").loc[1, "prev_n_still_open"] == 2


def test_the_latest_application_and_the_last_three_are_picked_by_window(previous_db):
    table = built(previous_db, "previous_application")
    assert (table.loc[1, "prev_last_refused"], table.loc[1, "prev_last_credit"], table.loc[1, "prev_last3_n_refused"]) == (0, 3000, 1)
    assert (table.loc[2, "prev_last_refused"], table.loc[2, "prev_last3_n_refused"]) == (1, 1)


def test_an_applicant_with_only_a_refusal_has_no_amounts_and_one_with_nothing_has_zero_counts(previous_db):
    table = built(previous_db, "previous_application")
    assert table.loc[2, "prev_n_approved"] == 0 and pd.isna(table.loc[2, "prev_credit_mean"])
    assert table.loc[3, "prev_n"] == 0 and pd.isna(table.loc[3, "prev_last_refused"])
    assert features.row_count_problems(previous_db, "previous_application") == []


def pos(applicant, loan, month, status, future, dpd, dpd_def):
    return {
        "SK_ID_PREV": loan,
        "SK_ID_CURR": applicant,
        "MONTHS_BALANCE": month,
        "NAME_CONTRACT_STATUS": status,
        "CNT_INSTALMENT_FUTURE": future,
        "SK_DPD": dpd,
        "SK_DPD_DEF": dpd_def,
    }


@pytest.fixture
def pos_db():
    return toy_connection(
        application_train=[{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}, {"SK_ID_CURR": 3}],
        pos_cash_balance=[
            pos(1, 1, -3, "Active", 2, 0, 0),
            pos(1, 1, -2, "Active", 1, 5, 0),
            pos(1, 1, -1, "Completed", 0, 0, 0),
            pos(1, 2, -14, "Active", 8, 0, 0),
            pos(1, 2, -2, "Active", 6, 10, 10),
            pos(1, 2, -1, "Active", 5, 20, 0),
            pos(2, 3, -1, "Active", 4, 0, 0),
        ],
    )


def test_pos_cash_counts_and_late_shares(pos_db):
    table = built(pos_db, "pos_cash")
    assert table.index.tolist() == [1, 2, 3]
    row = table.loc[1]
    assert (row["pos_n_loans"], row["pos_n_months"], row["pos_months_oldest"]) == (2, 6, -14)
    assert (row["pos_dpd_max"], row["pos_dpd_mean"]) == (20, pytest.approx(35 / 6))
    assert row["pos_late_share"] == pytest.approx(3 / 6)
    assert row["pos_late_def_share"] == pytest.approx(1 / 6)


def test_pos_cash_last_twelve_months_leave_out_the_old_month(pos_db):
    row = built(pos_db, "pos_cash").loc[1]
    assert row["pos_last12_months"] == 5
    assert row["pos_last12_late_share"] == pytest.approx(3 / 5)


def test_pos_cash_late_days_slope_is_the_least_squares_slope_over_months(pos_db):
    row = built(pos_db, "pos_cash").loc[1]
    months = [-3, -2, -1, -14, -2, -1]
    days = [0, 5, 0, 0, 10, 20]
    expected = ((6 * sum(m * d for m, d in zip(months, days))) - sum(months) * sum(days)) / (6 * sum(m * m for m in months) - sum(months) ** 2)
    assert expected == pytest.approx(505 / 761)
    assert row["pos_dpd_slope"] == pytest.approx(expected)
    assert row["pos_dpd_slope"] > 0


def test_pos_cash_latest_row_of_each_loan_decides_active_and_completed(pos_db):
    row = built(pos_db, "pos_cash").loc[1]
    assert (row["pos_n_active_loans"], row["pos_n_completed_loans"]) == (1, 1)
    assert row["pos_future_installments"] == 5


def test_pos_cash_a_single_month_has_no_slope_and_no_history_has_zero_counts(pos_db):
    table = built(pos_db, "pos_cash")
    assert pd.isna(table.loc[2, "pos_dpd_slope"]) and table.loc[2, "pos_late_share"] == 0
    assert table.loc[3, "pos_n_loans"] == 0 and pd.isna(table.loc[3, "pos_dpd_max"])
    assert features.row_count_problems(pos_db, "pos_cash") == []


def payment(applicant, loan, number, due, paid_on, due_amount, paid, version=1):
    return {
        "SK_ID_CURR": applicant,
        "SK_ID_PREV": loan,
        "NUM_INSTALMENT_VERSION": version,
        "NUM_INSTALMENT_NUMBER": number,
        "DAYS_INSTALMENT": due,
        "DAYS_ENTRY_PAYMENT": paid_on,
        "AMT_INSTALMENT": due_amount,
        "AMT_PAYMENT": paid,
    }


@pytest.fixture
def installments_db():
    first = [
        payment(1, 1, 1, -100, -100, 1000, 1000),
        payment(1, 1, 2, -80, -75, 1000, 1000),
        payment(1, 1, 3, -60, -58, 1000, 500),
        payment(1, 1, 3, -60, -50, 1000, 500),
        payment(1, 1, 4, -40, -45, 1000, 800),
        payment(1, 1, 5, -20, None, 1000, None),
        payment(1, 2, 1, -30, -30, 500, 500),
    ]
    second = [
        payment(2, 3, k, -10 * k, -10 * k + (20 if k >= 11 else 5 if k <= 4 else 0), 100, 100) for k in range(1, 13)
    ]
    return toy_connection(application_train=[{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}, {"SK_ID_CURR": 3}], installments_payments=first + second)


def test_installments_collapse_split_payments_into_one_instalment(installments_db):
    row = built(installments_db, "installments").loc[1]
    assert (row["ins_n_instalments"], row["ins_n_loans"], row["ins_n_unpaid"]) == (6, 2, 1)
    assert row["ins_days_oldest"] == -100


def test_installments_days_late_uses_the_last_part_of_a_split_payment(installments_db):
    row = built(installments_db, "installments").loc[1]
    assert row["ins_late_share"] == pytest.approx(2 / 5)
    assert row["ins_days_late_mean"] == pytest.approx(2.0)
    assert row["ins_days_late_max"] == 10
    assert row["ins_days_late_pos_mean"] == pytest.approx(3.0)


def test_installments_a_split_payment_that_adds_up_is_not_underpaid(installments_db):
    row = built(installments_db, "installments").loc[1]
    assert row["ins_underpaid_sum"] == 200
    assert row["ins_underpaid_share"] == pytest.approx(1 / 5)
    assert row["ins_underpaid_ratio"] == pytest.approx(200 / 4500)


def test_installments_last_ten_picks_the_most_recent_by_due_date(installments_db):
    row = built(installments_db, "installments").loc[2]
    assert row["ins_n_instalments"] == 12
    assert row["ins_last10_late_share"] == pytest.approx(4 / 10)
    assert row["ins_last10_days_late_mean"] == pytest.approx(2.0)
    assert row["ins_late_share"] == pytest.approx(6 / 12)
    assert row["ins_days_late_mean"] == pytest.approx(5.0)


def test_installments_late_slope_is_the_change_in_days_late_per_year(installments_db):
    import numpy as np

    due = np.array([-10 * k for k in range(1, 13)])
    late = np.array([20 if k >= 11 else 5 if k <= 4 else 0 for k in range(1, 13)])
    expected = np.polyfit(due, late, 1)[0] * 365
    assert built(installments_db, "installments").loc[2, "ins_late_slope_year"] == pytest.approx(expected)


def test_installments_applicant_without_payments_has_zero_counts(installments_db):
    row = built(installments_db, "installments").loc[3]
    assert row["ins_n_instalments"] == 0 and pd.isna(row["ins_late_share"]) and pd.isna(row["ins_late_slope_year"])
    assert features.row_count_problems(installments_db, "installments") == []


def card(applicant, card_id, month, status, balance, limit, drawings, atm, minimum, paid, dpd, dpd_def):
    return {
        "SK_ID_PREV": card_id,
        "SK_ID_CURR": applicant,
        "MONTHS_BALANCE": month,
        "NAME_CONTRACT_STATUS": status,
        "AMT_BALANCE": balance,
        "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_DRAWINGS_CURRENT": drawings,
        "AMT_DRAWINGS_ATM_CURRENT": atm,
        "AMT_INST_MIN_REGULARITY": minimum,
        "AMT_PAYMENT_CURRENT": paid,
        "SK_DPD": dpd,
        "SK_DPD_DEF": dpd_def,
    }


@pytest.fixture
def card_db():
    return toy_connection(
        application_train=[{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}, {"SK_ID_CURR": 3}],
        credit_card_balance=[
            card(1, 1, -3, "Active", 100, 1000, 50, 50, 20, 20, 0, 0),
            card(1, 1, -2, "Active", 200, 1000, 0, 0, 20, 10, 0, 0),
            card(1, 1, -1, "Active", 300, 1000, 100, 50, 20, 20, 5, 5),
            card(1, 2, -8, "Active", 500, 500, 200, 0, 50, 100, 0, 0),
            card(1, 2, -1, "Completed", 250, 500, 0, 0, None, None, 0, 0),
            card(2, 3, -1, "Active", 0, 0, 0, 0, 0, 0, 0, 0),
        ],
    )


def test_credit_card_counts_and_utilization(card_db):
    table = built(card_db, "credit_card")
    assert table.index.tolist() == [1, 2, 3]
    row = table.loc[1]
    assert (row["cc_n_cards"], row["cc_n_months"], row["cc_months_oldest"]) == (2, 5, -8)
    assert row["cc_utilization_mean"] == pytest.approx(0.42)
    assert row["cc_utilization_max"] == pytest.approx(1.0)
    assert row["cc_last6_utilization_mean"] == pytest.approx(0.275)
    assert (row["cc_balance_mean"], row["cc_balance_max"]) == (pytest.approx(270), 500)


def test_credit_card_cash_advance_share_and_missed_minimum_payments(card_db):
    row = built(card_db, "credit_card").loc[1]
    assert row["cc_drawings_mean"] == pytest.approx(70)
    assert row["cc_atm_share"] == pytest.approx(100 / 350)
    assert row["cc_min_payment_missed_share"] == pytest.approx(1 / 4)


def test_credit_card_days_past_due(card_db):
    row = built(card_db, "credit_card").loc[1]
    assert (row["cc_dpd_max"], row["cc_late_share"], row["cc_late_def_share"]) == (5, pytest.approx(0.2), pytest.approx(0.2))


def test_credit_card_current_position_comes_from_the_latest_month_of_each_active_card(card_db):
    row = built(card_db, "credit_card").loc[1]
    assert row["cc_n_active_cards"] == 1
    assert row["cc_balance_now"] == 300
    assert row["cc_utilization_now"] == pytest.approx(0.3)


def test_credit_card_zero_limit_gives_missing_utilization_and_no_card_gives_zero_counts(card_db):
    table = built(card_db, "credit_card")
    assert pd.isna(table.loc[2, "cc_utilization_mean"]) and pd.isna(table.loc[2, "cc_utilization_now"])
    assert table.loc[3, "cc_n_cards"] == 0 and pd.isna(table.loc[3, "cc_balance_mean"])
    assert features.row_count_problems(card_db, "credit_card") == []


def applicant(sk, income, credit, annuity, goods, birth, employed, family, children, e1, e2, e3):
    return {
        "SK_ID_CURR": sk,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": birth,
        "DAYS_EMPLOYED": employed,
        "CNT_FAM_MEMBERS": family,
        "CNT_CHILDREN": children,
        "EXT_SOURCE_1": e1,
        "EXT_SOURCE_2": e2,
        "EXT_SOURCE_3": e3,
    }


@pytest.fixture
def application_db():
    return toy_connection(
        application_train=[
            applicant(1, 100000, 500000, 25000, 450000, -14610, -3652.5, 4, 2, 0.5, 0.6, None),
            applicant(2, 0, 300000, 15000, 300000, -23376, 365243, 1, 0, None, None, None),
            applicant(3, 80000, 200000, None, 180000, -10957.5, -365.25, 2, 1, 0.9, None, 0.1),
        ]
    )


def test_application_ratios_for_a_regular_applicant(application_db):
    row = built(application_db, "application").loc[1]
    assert row["app_credit_to_income"] == pytest.approx(5.0)
    assert row["app_annuity_to_income"] == pytest.approx(0.25)
    assert row["app_credit_term"] == pytest.approx(20.0)
    assert row["app_credit_to_goods"] == pytest.approx(500000 / 450000)
    assert row["app_income_per_person"] == pytest.approx(25000)
    assert row["app_children_share"] == pytest.approx(0.5)


def test_application_age_and_employment_use_years_and_the_age_at_the_start_of_the_job(application_db):
    row = built(application_db, "application").loc[1]
    assert row["app_age_years"] == pytest.approx(40.0)
    assert row["app_years_employed"] == pytest.approx(10.0)
    assert row["app_employed_to_age"] == pytest.approx(0.25)
    assert row["app_age_at_job_start"] == pytest.approx(30.0)


def test_application_the_employment_placeholder_becomes_missing_in_every_derived_column(application_db):
    row = built(application_db, "application").loc[2]
    assert pd.isna(row["app_years_employed"]) and pd.isna(row["app_employed_to_age"]) and pd.isna(row["app_age_at_job_start"])
    assert row["app_age_years"] == pytest.approx(23376 / 365.25)


def test_application_zero_income_or_missing_annuity_gives_missing_ratios(application_db):
    table = built(application_db, "application")
    assert pd.isna(table.loc[2, "app_credit_to_income"]) and pd.isna(table.loc[2, "app_annuity_to_income"])
    assert pd.isna(table.loc[3, "app_credit_term"]) and pd.isna(table.loc[3, "app_annuity_to_income"])


def test_application_ext_source_summary_skips_missing_scores(application_db):
    table = built(application_db, "application")
    assert (table.loc[1, "app_ext_n_known"], table.loc[1, "app_ext_mean"], table.loc[1, "app_ext_min"], table.loc[1, "app_ext_max"]) == (2, pytest.approx(0.55), 0.5, 0.6)
    assert (table.loc[3, "app_ext_mean"], table.loc[3, "app_ext_min"], table.loc[3, "app_ext_max"]) == (pytest.approx(0.5), 0.1, 0.9)
    assert table.loc[2, "app_ext_n_known"] == 0
    assert pd.isna(table.loc[2, "app_ext_mean"]) and pd.isna(table.loc[2, "app_ext_min"]) and pd.isna(table.loc[2, "app_ext_max"])
    assert features.row_count_problems(application_db, "application") == []


def test_load_features_adds_the_group_columns_and_keeps_rows_and_targets_in_order(bureau_db):
    import preprocess

    bureau_db.execute("DROP TABLE application_train")
    pd.DataFrame({"SK_ID_CURR": [3, 1, 2], "TARGET": [0, 1, 0], "AMT_INCOME_TOTAL": [10.0, 20.0, 30.0]}).to_sql("application_train", bureau_db, index=False)
    pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "part": ["train", "train", "holdout"]}).to_sql("split", bureau_db, index=False)
    features.build_group(bureau_db, "bureau")
    X, y = preprocess.load_features(bureau_db, "train", groups=["bureau"])
    plain, plain_y = preprocess.load_features(bureau_db, "train")
    assert plain.columns.tolist() == ["AMT_INCOME_TOTAL"]
    assert len(X) == 2 and y.tolist() == plain_y.tolist() == [1, 0]
    assert X["AMT_INCOME_TOTAL"].tolist() == [20.0, 30.0]
    assert X["bur_n_credits"].tolist() == [5, 1]
    assert "SK_ID_CURR" not in X.columns and "TARGET" not in X.columns
    assert "bur_n_credits" not in plain.columns


@needs_db
def test_real_load_features_with_groups_has_the_same_rows_as_without():
    import preprocess

    con = sqlite3.connect(DB)
    present = [g for g in features.GROUPS if con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (features.table_of(g),)).fetchone()]
    if not present:
        pytest.skip("no feature table is built")
    plain, plain_y = preprocess.load_features(con, "train")
    X, y = preprocess.load_features(con, "train", groups=present[:1])
    assert len(X) == len(plain) and (y == plain_y).all()
    assert X.iloc[:, : plain.shape[1]].equals(plain)
    con.close()


@pytest.mark.parametrize("group", features.GROUPS)
def test_the_sql_never_reads_the_target_or_the_split(group):
    text = (features.SQL_DIR / f"{group}.sql").read_text().upper()
    assert "TARGET" not in text
    assert "SPLIT" not in text


@needs_db
@pytest.mark.parametrize("group", features.GROUPS)
def test_each_built_table_has_one_row_per_applicant(group):
    con = sqlite3.connect(DB)
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (features.table_of(group),)).fetchone()
    if not exists:
        pytest.skip(f"{features.table_of(group)} is not built")
    assert features.row_count_problems(con, group) == []
    con.close()
