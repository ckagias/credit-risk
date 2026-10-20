DROP TABLE IF EXISTS feat_application;

CREATE TABLE feat_application AS
WITH base AS (
    SELECT
        SK_ID_CURR,
        AMT_CREDIT * 1.0 / NULLIF(AMT_INCOME_TOTAL, 0) AS app_credit_to_income,
        AMT_ANNUITY * 1.0 / NULLIF(AMT_INCOME_TOTAL, 0) AS app_annuity_to_income,
        AMT_CREDIT * 1.0 / NULLIF(AMT_ANNUITY, 0) AS app_credit_term,
        AMT_CREDIT * 1.0 / NULLIF(AMT_GOODS_PRICE, 0) AS app_credit_to_goods,
        AMT_INCOME_TOTAL * 1.0 / NULLIF(CNT_FAM_MEMBERS, 0) AS app_income_per_person,
        CNT_CHILDREN * 1.0 / NULLIF(CNT_FAM_MEMBERS, 0) AS app_children_share,
        -DAYS_BIRTH / 365.25 AS app_age_years,
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE -DAYS_EMPLOYED / 365.25 END AS app_years_employed,
        (CASE WHEN EXT_SOURCE_1 IS NULL THEN 0 ELSE 1 END
            + CASE WHEN EXT_SOURCE_2 IS NULL THEN 0 ELSE 1 END
            + CASE WHEN EXT_SOURCE_3 IS NULL THEN 0 ELSE 1 END) AS app_ext_n_known,
        COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0) AS ext_total,
        MIN(COALESCE(EXT_SOURCE_1, 9), COALESCE(EXT_SOURCE_2, 9), COALESCE(EXT_SOURCE_3, 9)) AS ext_lowest,
        MAX(COALESCE(EXT_SOURCE_1, -9), COALESCE(EXT_SOURCE_2, -9), COALESCE(EXT_SOURCE_3, -9)) AS ext_highest
    FROM application_train
)
SELECT
    SK_ID_CURR,
    app_credit_to_income,
    app_annuity_to_income,
    app_credit_term,
    app_credit_to_goods,
    app_income_per_person,
    app_children_share,
    app_age_years,
    app_years_employed,
    app_years_employed / app_age_years AS app_employed_to_age,
    app_age_years - app_years_employed AS app_age_at_job_start,
    app_ext_n_known,
    CASE WHEN app_ext_n_known > 0 THEN ext_total / app_ext_n_known END AS app_ext_mean,
    CASE WHEN app_ext_n_known > 0 THEN ext_lowest END AS app_ext_min,
    CASE WHEN app_ext_n_known > 0 THEN ext_highest END AS app_ext_max
FROM base;

CREATE UNIQUE INDEX idx_feat_application_SK_ID_CURR ON feat_application (SK_ID_CURR);
