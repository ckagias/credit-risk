DROP TABLE IF EXISTS feat_previous_application;

CREATE TABLE feat_previous_application AS
WITH summary AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS prev_n,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_n_approved,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused') AS prev_n_refused,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Canceled') AS prev_n_canceled,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Unused offer') AS prev_n_unused,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Cash loans') AS prev_n_cash,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Consumer loans') AS prev_n_consumer,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Revolving loans') AS prev_n_revolving,
        COUNT(*) FILTER (WHERE DAYS_DECISION >= -365) AS prev_n_last_year,
        MIN(DAYS_DECISION) AS prev_days_oldest,
        MAX(DAYS_DECISION) AS prev_days_newest,
        SUM(AMT_CREDIT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_credit_sum,
        AVG(AMT_CREDIT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_credit_mean,
        MAX(AMT_CREDIT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_credit_max,
        AVG(AMT_CREDIT * 1.0 / NULLIF(AMT_APPLICATION, 0)) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_granted_ratio,
        AVG(AMT_ANNUITY) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_annuity_mean,
        AVG(RATE_DOWN_PAYMENT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_down_rate_mean,
        AVG(CNT_PAYMENT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_term_mean,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved' AND DAYS_TERMINATION = 365243) AS prev_n_still_open
    FROM previous_application
    GROUP BY SK_ID_CURR
),
ranked AS (
    SELECT
        SK_ID_CURR,
        NAME_CONTRACT_STATUS,
        AMT_CREDIT,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC, SK_ID_PREV DESC) AS recency
    FROM previous_application
),
last_one AS (
    SELECT
        SK_ID_CURR,
        CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END AS prev_last_refused,
        AMT_CREDIT AS prev_last_credit
    FROM ranked
    WHERE recency = 1
),
last_three AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused') AS prev_last3_n_refused
    FROM ranked
    WHERE recency <= 3
    GROUP BY SK_ID_CURR
)
SELECT
    a.SK_ID_CURR,
    COALESCE(s.prev_n, 0) AS prev_n,
    COALESCE(s.prev_n_approved, 0) AS prev_n_approved,
    COALESCE(s.prev_n_refused, 0) AS prev_n_refused,
    COALESCE(s.prev_n_canceled, 0) AS prev_n_canceled,
    COALESCE(s.prev_n_unused, 0) AS prev_n_unused,
    s.prev_n_refused * 1.0 / NULLIF(s.prev_n, 0) AS prev_refused_share,
    COALESCE(s.prev_n_cash, 0) AS prev_n_cash,
    COALESCE(s.prev_n_consumer, 0) AS prev_n_consumer,
    COALESCE(s.prev_n_revolving, 0) AS prev_n_revolving,
    COALESCE(s.prev_n_last_year, 0) AS prev_n_last_year,
    s.prev_days_oldest,
    s.prev_days_newest,
    s.prev_credit_sum,
    s.prev_credit_mean,
    s.prev_credit_max,
    s.prev_granted_ratio,
    s.prev_annuity_mean,
    s.prev_down_rate_mean,
    s.prev_term_mean,
    COALESCE(s.prev_n_still_open, 0) AS prev_n_still_open,
    l.prev_last_refused,
    l.prev_last_credit,
    COALESCE(t.prev_last3_n_refused, 0) AS prev_last3_n_refused
FROM application_train a
LEFT JOIN summary s ON s.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN last_one l ON l.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN last_three t ON t.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_feat_previous_application_SK_ID_CURR ON feat_previous_application (SK_ID_CURR);
