DROP TABLE IF EXISTS feat_pos_cash;

CREATE TABLE feat_pos_cash AS
WITH summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(DISTINCT SK_ID_PREV) AS pos_n_loans,
        COUNT(*) AS pos_n_months,
        MIN(MONTHS_BALANCE) AS pos_months_oldest,
        MAX(SK_DPD) AS pos_dpd_max,
        AVG(SK_DPD) AS pos_dpd_mean,
        AVG(SK_DPD > 0) AS pos_late_share,
        AVG(SK_DPD_DEF > 0) AS pos_late_def_share,
        COUNT(*) FILTER (WHERE MONTHS_BALANCE >= -12) AS pos_last12_months,
        AVG(SK_DPD > 0) FILTER (WHERE MONTHS_BALANCE >= -12) AS pos_last12_late_share,
        (COUNT(*) * SUM(MONTHS_BALANCE * SK_DPD * 1.0) - SUM(MONTHS_BALANCE * 1.0) * SUM(SK_DPD * 1.0))
            / NULLIF(COUNT(*) * SUM(MONTHS_BALANCE * MONTHS_BALANCE * 1.0) - SUM(MONTHS_BALANCE * 1.0) * SUM(MONTHS_BALANCE * 1.0), 0)
            AS pos_dpd_slope
    FROM pos_cash_balance
    GROUP BY SK_ID_CURR
),
ranked AS (
    SELECT
        SK_ID_CURR,
        NAME_CONTRACT_STATUS,
        CNT_INSTALMENT_FUTURE,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE DESC) AS recency
    FROM pos_cash_balance
),
loans AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Active') AS pos_n_active_loans,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Completed') AS pos_n_completed_loans,
        SUM(CNT_INSTALMENT_FUTURE) FILTER (WHERE NAME_CONTRACT_STATUS = 'Active') AS pos_future_installments
    FROM ranked
    WHERE recency = 1
    GROUP BY SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    COALESCE(s.pos_n_loans, 0) AS pos_n_loans,
    COALESCE(s.pos_n_months, 0) AS pos_n_months,
    s.pos_months_oldest,
    s.pos_dpd_max,
    s.pos_dpd_mean,
    s.pos_late_share,
    s.pos_late_def_share,
    COALESCE(s.pos_last12_months, 0) AS pos_last12_months,
    s.pos_last12_late_share,
    s.pos_dpd_slope,
    COALESCE(l.pos_n_active_loans, 0) AS pos_n_active_loans,
    COALESCE(l.pos_n_completed_loans, 0) AS pos_n_completed_loans,
    l.pos_future_installments
FROM application_train a
LEFT JOIN summary s ON s.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN loans l ON l.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_feat_pos_cash_SK_ID_CURR ON feat_pos_cash (SK_ID_CURR);
