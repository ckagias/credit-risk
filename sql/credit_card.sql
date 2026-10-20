DROP TABLE IF EXISTS feat_credit_card;

CREATE TABLE feat_credit_card AS
WITH summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(DISTINCT SK_ID_PREV) AS cc_n_cards,
        COUNT(*) AS cc_n_months,
        MIN(MONTHS_BALANCE) AS cc_months_oldest,
        AVG(AMT_BALANCE) AS cc_balance_mean,
        MAX(AMT_BALANCE) AS cc_balance_max,
        AVG(AMT_BALANCE * 1.0 / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS cc_utilization_mean,
        MAX(AMT_BALANCE * 1.0 / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) AS cc_utilization_max,
        AVG(AMT_BALANCE * 1.0 / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)) FILTER (WHERE MONTHS_BALANCE >= -6) AS cc_last6_utilization_mean,
        AVG(AMT_DRAWINGS_CURRENT) AS cc_drawings_mean,
        SUM(AMT_DRAWINGS_ATM_CURRENT) * 1.0 / NULLIF(SUM(AMT_DRAWINGS_CURRENT), 0) AS cc_atm_share,
        AVG(AMT_PAYMENT_CURRENT < AMT_INST_MIN_REGULARITY) AS cc_min_payment_missed_share,
        MAX(SK_DPD) AS cc_dpd_max,
        AVG(SK_DPD > 0) AS cc_late_share,
        AVG(SK_DPD_DEF > 0) AS cc_late_def_share
    FROM credit_card_balance
    GROUP BY SK_ID_CURR
),
ranked AS (
    SELECT
        SK_ID_CURR,
        NAME_CONTRACT_STATUS,
        AMT_BALANCE,
        AMT_CREDIT_LIMIT_ACTUAL,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE DESC) AS recency
    FROM credit_card_balance
),
cards AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS cc_n_active_cards,
        SUM(AMT_BALANCE) AS cc_balance_now,
        SUM(AMT_CREDIT_LIMIT_ACTUAL) AS cc_limit_now
    FROM ranked
    WHERE recency = 1 AND NAME_CONTRACT_STATUS = 'Active'
    GROUP BY SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    COALESCE(s.cc_n_cards, 0) AS cc_n_cards,
    COALESCE(s.cc_n_months, 0) AS cc_n_months,
    s.cc_months_oldest,
    s.cc_balance_mean,
    s.cc_balance_max,
    s.cc_utilization_mean,
    s.cc_utilization_max,
    s.cc_last6_utilization_mean,
    s.cc_drawings_mean,
    s.cc_atm_share,
    s.cc_min_payment_missed_share,
    s.cc_dpd_max,
    s.cc_late_share,
    s.cc_late_def_share,
    COALESCE(c.cc_n_active_cards, 0) AS cc_n_active_cards,
    c.cc_balance_now,
    c.cc_balance_now * 1.0 / NULLIF(c.cc_limit_now, 0) AS cc_utilization_now
FROM application_train a
LEFT JOIN summary s ON s.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN cards c ON c.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_feat_credit_card_SK_ID_CURR ON feat_credit_card (SK_ID_CURR);
