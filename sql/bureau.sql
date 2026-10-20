DROP TABLE IF EXISTS feat_bureau;

CREATE TABLE feat_bureau AS
WITH credit AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS bur_n_credits,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bur_n_active,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Closed') AS bur_n_closed,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE IN ('Bad debt', 'Sold')) AS bur_n_bad_or_sold,
        COUNT(DISTINCT CREDIT_TYPE) AS bur_n_types,
        COUNT(*) FILTER (WHERE DAYS_CREDIT >= -365) AS bur_n_last_year,
        MIN(DAYS_CREDIT) AS bur_days_oldest,
        MAX(DAYS_CREDIT) AS bur_days_newest,
        SUM(AMT_CREDIT_SUM) AS bur_credit_sum,
        SUM(AMT_CREDIT_SUM_DEBT) AS bur_debt_sum,
        SUM(AMT_CREDIT_SUM_OVERDUE) AS bur_overdue_sum,
        MAX(AMT_CREDIT_MAX_OVERDUE) AS bur_max_overdue,
        SUM(AMT_CREDIT_SUM) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bur_active_credit_sum,
        SUM(AMT_CREDIT_SUM_DEBT) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bur_active_debt_sum,
        AVG(DAYS_CREDIT_ENDDATE) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bur_active_end_mean,
        MAX(CREDIT_DAY_OVERDUE) AS bur_day_overdue_max,
        COUNT(*) FILTER (WHERE CREDIT_DAY_OVERDUE > 0) AS bur_n_overdue_now,
        SUM(CNT_CREDIT_PROLONG) AS bur_prolong_sum
    FROM bureau
    GROUP BY SK_ID_CURR
),
ranked AS (
    SELECT
        SK_ID_CURR,
        CREDIT_ACTIVE,
        AMT_CREDIT_SUM,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_CREDIT DESC, SK_ID_BUREAU DESC) AS recency
    FROM bureau
),
recent AS (
    SELECT
        SK_ID_CURR,
        AVG(AMT_CREDIT_SUM) AS bur_last3_credit_mean,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bur_last3_n_active
    FROM ranked
    WHERE recency <= 3
    GROUP BY SK_ID_CURR
),
monthly AS (
    SELECT
        b.SK_ID_CURR,
        COUNT(*) AS bur_bb_months,
        COUNT(*) FILTER (WHERE bb.STATUS IN ('0', '1', '2', '3', '4', '5')) AS bur_bb_known_months,
        COUNT(*) FILTER (WHERE bb.STATUS IN ('1', '2', '3', '4', '5')) AS bur_bb_late_months,
        COUNT(*) FILTER (WHERE bb.MONTHS_BALANCE >= -12 AND bb.STATUS IN ('1', '2', '3', '4', '5')) AS bur_bb_late_last_year,
        MAX(CASE WHEN bb.STATUS IN ('0', '1', '2', '3', '4', '5') THEN CAST(bb.STATUS AS INTEGER) END) AS bur_bb_worst_status
    FROM bureau_balance bb
    JOIN bureau b ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
    GROUP BY b.SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    COALESCE(c.bur_n_credits, 0) AS bur_n_credits,
    COALESCE(c.bur_n_active, 0) AS bur_n_active,
    COALESCE(c.bur_n_closed, 0) AS bur_n_closed,
    COALESCE(c.bur_n_bad_or_sold, 0) AS bur_n_bad_or_sold,
    COALESCE(c.bur_n_types, 0) AS bur_n_types,
    COALESCE(c.bur_n_last_year, 0) AS bur_n_last_year,
    c.bur_days_oldest,
    c.bur_days_newest,
    c.bur_credit_sum,
    c.bur_debt_sum,
    c.bur_overdue_sum,
    c.bur_max_overdue,
    c.bur_active_credit_sum,
    c.bur_active_debt_sum,
    c.bur_active_debt_sum * 1.0 / NULLIF(c.bur_active_credit_sum, 0) AS bur_active_debt_ratio,
    c.bur_active_end_mean,
    c.bur_day_overdue_max,
    COALESCE(c.bur_n_overdue_now, 0) AS bur_n_overdue_now,
    c.bur_prolong_sum,
    r.bur_last3_credit_mean,
    COALESCE(r.bur_last3_n_active, 0) AS bur_last3_n_active,
    COALESCE(m.bur_bb_months, 0) AS bur_bb_months,
    m.bur_bb_late_months * 1.0 / NULLIF(m.bur_bb_known_months, 0) AS bur_bb_late_share,
    COALESCE(m.bur_bb_late_last_year, 0) AS bur_bb_late_last_year,
    m.bur_bb_worst_status
FROM application_train a
LEFT JOIN credit c ON c.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN recent r ON r.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN monthly m ON m.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_feat_bureau_SK_ID_CURR ON feat_bureau (SK_ID_CURR);
