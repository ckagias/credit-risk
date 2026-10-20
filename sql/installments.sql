DROP TABLE IF EXISTS feat_installments;

CREATE TABLE feat_installments AS
WITH instalment AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MAX(DAYS_INSTALMENT) AS due_day,
        MAX(DAYS_ENTRY_PAYMENT) - MAX(DAYS_INSTALMENT) AS days_late,
        MAX(AMT_INSTALMENT) AS due_amount,
        MAX(MAX(AMT_INSTALMENT) - SUM(AMT_PAYMENT), 0) AS underpaid
    FROM installments_payments
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY due_day DESC, SK_ID_PREV DESC) AS recency
    FROM instalment
),
summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS ins_n_instalments,
        COUNT(DISTINCT SK_ID_PREV) AS ins_n_loans,
        COUNT(*) FILTER (WHERE days_late IS NULL) AS ins_n_unpaid,
        MIN(due_day) AS ins_days_oldest,
        AVG(days_late > 0) AS ins_late_share,
        AVG(days_late) AS ins_days_late_mean,
        MAX(days_late) AS ins_days_late_max,
        AVG(MAX(days_late, 0)) AS ins_days_late_pos_mean,
        AVG(underpaid > 0.01) AS ins_underpaid_share,
        SUM(underpaid) AS ins_underpaid_sum,
        SUM(underpaid) * 1.0 / NULLIF(SUM(due_amount) FILTER (WHERE underpaid IS NOT NULL), 0) AS ins_underpaid_ratio,
        AVG(days_late > 0) FILTER (WHERE recency <= 10) AS ins_last10_late_share,
        AVG(days_late) FILTER (WHERE recency <= 10) AS ins_last10_days_late_mean,
        365.0 * (COUNT(days_late) * SUM(due_day * days_late * 1.0) - SUM(due_day * 1.0) FILTER (WHERE days_late IS NOT NULL) * SUM(days_late * 1.0))
            / NULLIF(
                COUNT(days_late) * SUM(due_day * due_day * 1.0) FILTER (WHERE days_late IS NOT NULL)
                - SUM(due_day * 1.0) FILTER (WHERE days_late IS NOT NULL) * SUM(due_day * 1.0) FILTER (WHERE days_late IS NOT NULL),
                0
            ) AS ins_late_slope_year
    FROM ranked
    GROUP BY SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    COALESCE(s.ins_n_instalments, 0) AS ins_n_instalments,
    COALESCE(s.ins_n_loans, 0) AS ins_n_loans,
    COALESCE(s.ins_n_unpaid, 0) AS ins_n_unpaid,
    s.ins_days_oldest,
    s.ins_late_share,
    s.ins_days_late_mean,
    s.ins_days_late_max,
    s.ins_days_late_pos_mean,
    s.ins_underpaid_share,
    s.ins_underpaid_sum,
    s.ins_underpaid_ratio,
    s.ins_last10_late_share,
    s.ins_last10_days_late_mean,
    s.ins_late_slope_year
FROM application_train a
LEFT JOIN summary s ON s.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_feat_installments_SK_ID_CURR ON feat_installments (SK_ID_CURR);
