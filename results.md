# Results

Loan default prediction on the Home Credit Default Risk data. Every number below comes from a script in `src/` and a file in `results/` (named in `README.md` and `MATH.md`). Intervals are 95% unless a standard deviation over folds is given.

## Summary

- The best model is LightGBM with tuned settings on the application columns plus 106 columns built in SQL from the other six tables. On the 61,503 holdout applicants, scored once, it has **PR-AUC 0.2957 (0.2835 to 0.3089)**, ROC-AUC 0.7918 (0.7852 to 0.7986) and log loss 0.2347 (0.2303 to 0.2394). Predicting the 8.1% base rate for everyone gives PR-AUC 0.0807, so the model is 3.7 times the base rate.
- The ranking of the families is the same on the holdout and in cross-validation: LightGBM, then logistic regression, then random forest, then the one-feature baseline.
- **What moved the score most was the features, not the model.** Adding the SQL feature groups raised LightGBM from 0.2385 to 0.2756 PR-AUC in cross-validation (higher in 5 of 5 folds), and the choice of model between logistic regression and LightGBM was worth 0.014 to 0.021 on the same columns. Tuning added 0.006 on top, and class weights, resampling and calibration added nothing to the ranking.
- Under an assumed cost (a missed default loses 45% of the loan, a refused good loan a 10% margin, both assumptions) the model at threshold 0.182 costs 18.0% (16.3% to 19.6%) less on the holdout than approving everybody.
- It is well calibrated overall and in every segment of 1,000 applicants or more, but it ranks worse for applicants aged 60 and over and under 30, and its error rates differ between men and women and between age groups.
- No result here is a statement about a live lending business: the data has no dates, the costs are assumed, and the model has not been served or monitored.

## Setup

- **Data.** `application_train`, 307,511 applicants, 8.07% default, plus six tables of credit history (bureau, previous applications, POS cash, instalments, credit card). `application_test` has no public labels and is not used.
- **Split.** Stratified 80/20 of `application_train`, fixed seed: 246,008 training and 61,503 holdout applicants. Settings were chosen by 5-fold stratified cross-validation on the training part. The holdout was scored once, after the scoring code and the settings were committed.
- **Features.** 120 application columns (300 model inputs after imputation, missing indicators and encoding), plus 106 columns from six SQL feature groups (478 model inputs in the final model). Every fit step (imputation, clipping, scaling, encoding) is fitted inside the training fold only.
- **Metrics.** PR-AUC (the main one, because only 8% default), ROC-AUC and log loss, from one evaluation function checked against scikit-learn. Accuracy is not used.

## Cross-validation: the families on the same folds

5-fold stratified cross-validation on the 246,008 training applicants, mean (standard deviation over folds). Application columns only unless stated.

| Model | PR-AUC | ROC-AUC | Log loss |
|---|---|---|---|
| Base rate for everyone | 0.0807 | 0.5000 | 0.2805 |
| `EXT_SOURCE` mean, logistic | 0.1889 (0.0021) | 0.7164 (0.0025) | 0.2577 (0.0004) |
| Decision tree, depth 7 | 0.1866 (0.0053) | 0.7138 | 0.2659 |
| Naive Bayes, 3 `EXT_SOURCE` columns, calibrated | 0.1971 | 0.7207 | 0.2567 |
| Naive Bayes, all columns, calibrated | 0.1461 | 0.6238 | 0.2784 |
| kNN, k = 500, 20,000 training rows | 0.1698 | | |
| RBF SVM, 20,000 training rows | 0.2123 | | |
| Logistic regression, L2, `C` = 0.03 | 0.2249 (0.0045) | 0.7475 (0.0012) | 0.2495 (0.0003) |
| Random forest, 500 trees, 30 columns per split | 0.2274 (0.0026) | 0.7462 (0.0012) | 0.2513 (0.0003) |
| LightGBM, learning rate 0.02, default settings | 0.2401 (0.0052) | 0.7563 (0.0017) | 0.2468 (0.0005) |
| XGBoost, learning rate 0.02, default settings | 0.2401 (0.0050) | 0.7563 (0.0014) | 0.2467 (0.0004) |
| LightGBM, tuned, new folds (seed 7), learning rate 0.05 | 0.2431 (0.0030) | 0.7587 (0.0037) | 0.2461 (0.0007) |
| **With the six SQL feature groups** | | | |
| Random forest | 0.2527 (0.0023) | 0.7664 (0.0013) | 0.2450 (0.0003) |
| Logistic regression | 0.2606 (0.0032) | 0.7743 (0.0014) | 0.2415 (0.0003) |
| LightGBM, default settings, learning rate 0.05 | 0.2756 (0.0039) | 0.7842 (0.0012) | 0.2381 (0.0004) |
| LightGBM, tuned settings | 0.2813 (0.0045) | 0.7891 (0.0011) | 0.2365 (0.0004) |

- Boosting beats the forest and logistic regression on the same columns, by 0.013 to 0.015 PR-AUC with application columns only, and by 0.021 (over logistic regression) and 0.029 (over the forest) with the SQL features, higher in 5 of 5 folds in the SQL case.
- LightGBM and XGBoost give the same scores (fold-by-fold difference in PR-AUC 0.0000, std 0.0012 at learning rate 0.02), and XGBoost takes 2.1 to 2.7 times as long to fit.
- The tuned settings beat the default ones in every one of 10 fold comparisons on new folds, by 0.003 to 0.004 PR-AUC (60 Optuna trials; the study took 24 minutes).
- The SQL features help every model and every group improves PR-AUC in 5 of 5 folds: all six together add 0.037 to LightGBM (0.2385 to 0.2756) and 0.036 to logistic regression (0.2250 to 0.2606).

## Holdout

Scored once on 61,503 applicants, all models on the six SQL groups and refitted on the 246,008 training applicants. Intervals from 1,000 bootstrap resamples of the holdout applicants.

| Model | PR-AUC | ROC-AUC | Log loss |
|---|---:|---:|---:|
| LightGBM, tuned | 0.2957 (0.2835 to 0.3089) | 0.7918 (0.7852 to 0.7986) | 0.2347 (0.2303 to 0.2394) |
| Logistic regression, `C` = 0.03 | 0.2731 (0.2622 to 0.2860) | 0.7775 (0.7707 to 0.7845) | 0.2398 (0.2354 to 0.2446) |
| Random forest, 200 trees | 0.2605 (0.2491 to 0.2735) | 0.7690 (0.7617 to 0.7762) | 0.2440 (0.2398 to 0.2486) |
| One-feature logistic (`EXT_SOURCE` mean) | 0.1921 (0.1825 to 0.2014) | 0.7165 (0.7083 to 0.7241) | 0.2574 (0.2528 to 0.2623) |
| Base rate for everyone | 0.0807 (0.0787 to 0.0828) | 0.5000 | 0.2805 (0.2755 to 0.2856) |

- Every model scores higher on the holdout than its cross-validation mean (LightGBM by 0.014 PR-AUC, logistic regression 0.013, forest 0.008, one-feature model 0.003). The final models see 25% more training rows than a cross-validation fit, and the holdout is a different sample; the two were not separated.
- The PR-AUC intervals of LightGBM and logistic regression overlap (0.2835 to 0.2860), so the holdout alone does not show that LightGBM is better. In cross-validation it was better in 5 of 5 folds (by 0.021 on average); on the holdout its ROC-AUC interval is above that of logistic regression (0.7852 against 0.7845).
- The intervals reflect the sampling of the holdout applicants for one fit of each model, not the variation from refitting.

## Calibration, imbalance and the decision threshold

- **Imbalance.** Class weights, undersampling, oversampling and SMOTE did not improve PR-AUC beyond the fold-to-fold noise for logistic regression or LightGBM (largest gain +0.0007, std 0.0031; seven of eight changes zero or negative). They raise the mean predicted probability from 0.08 to about 0.40 and the log loss from 0.25 to 0.56 to 0.60, until the odds are multiplied back by 1/w.
- **Calibration.** Logistic regression and LightGBM trained on log loss are already calibrated (ECE 0.0025 and 0.0035) and neither Platt scaling nor isotonic regression improves them. The forest is not (ECE 0.0129) and Platt scaling brings it to 0.0027. The best model's out-of-fold ECE is 0.0025.
- **Threshold.** A missed default is assumed to lose 45% of the credit amount and a refused good loan a 10% margin (ratio 4.5, loan-size weighted), so the best threshold on calibrated scores is 1/(1 + 4.5) = 0.182. On the training part, out of fold, it costs 281.2 per 1,000 applicants against 337.9 for refusing nobody (16.8% less); the grid optimum is 0.19 at the same cost (281.1). On the holdout it costs 277.9 (269.0 to 287.1) against 338.8 (328.9 to 350.2), 18.0% less (16.3% to 19.6%), refusing 11.0% of the applicants and 41.0% of the defaulters.
- **The cost assumption decides how much the model is worth.** With a flat ratio of 10 an earlier model saved 33.7% over refusing nobody; at 4.5 the same kind of model saves 11.2% (application columns only). On the loan-size weighted cost, thresholds chosen for a flat ratio of 2, 5, 10 or 20 cost 6.8%, 0.6%, 16.5% or 64.7% more than the best one (application-column model).

## What the model uses

- **One column dominates.** The mean of the three `EXT_SOURCE` scores costs 0.044 PR-AUC when shuffled, six times the next column, and has 3.3 times the mean absolute SHAP value of the next. The six SQL groups hold 62.6% of the SHAP importance and the raw application columns 37.3%.
- Permutation importance and SHAP agree on the top (15 of the top 20 columns are the same) and only partly below it (rank correlation 0.59 over 477 columns).
- Higher annuity, annuity-to-income, credit-to-goods ratio, share of late instalments and number of future instalments raise the predicted risk; a higher `EXT_SOURCE` mean, previous annuity and number of POS-cash monthly records in the last 12 months lower it. The credit term has no single direction.

## Where the model fails

Out of fold on the 246,008 training applicants, at the threshold 0.182.

- It is calibrated in all 17 segments of 1,000 applicants or more (mean predicted probability within 0.002 of the default rate).
- Most of the difference in error counts between segments follows the default rate: income bands 1 to 5 all have ROC-AUC between 0.786 and 0.790, yet the top band (6.5% default) has 68.3% of its defaulters missed against 57.7% to 60.8% in the others.
- Ranking is worse at the age extremes: ROC-AUC 0.745 (0.733 to 0.759) for 60 and over and 0.764 (0.756 to 0.772) for under 30, against 0.789 overall. At 60 and over the model misses 82.3% of defaulters and saves 7.5% of the cost; under 30 it refuses 19.7% of applicants and 15.8% of the good loans.
- Revolving loans gain little (6.7% saved, 79.0% of defaulters missed), and applicants without a bureau record rank worse (ROC-AUC 0.777 against 0.790). Applicants with no `EXT_SOURCE` score at all are 139 (12 defaulters), too few to conclude anything.

## Fairness

Calibrated within every gender and age group, yet the error rates differ (out of fold, threshold 0.182).

| Group | Default rate | Approved | Defaulters refused | Good loans refused |
|---|---:|---:|---:|---:|
| Women | 7.0% | 91.5% | 35.1% | 6.5% |
| Men | 10.1% | 84.3% | 46.5% | 12.2% |
| Under 30 | 11.5% | 80.3% | 49.6% | 15.8% |
| 60 and over | 4.9% | 96.8% | 17.7% | 2.4% |

- Approval ratio, men against women: 0.922 (0.919 to 0.925). Under 30 against 60 and over: 0.830 (0.825 to 0.834). Both are above 0.8, the four-fifths rule of thumb.
- The gender column adds nothing measurable to the ranking (PR-AUC 0.2809 without it and 0.2809 with it), but without it the model is no longer calibrated by gender (it predicts 9.3% for men whose rate is 10.1%) and the gaps shrink without closing (approval ratio 0.945, defaulters refused +6.7 points instead of +11.3). Age is an input and removing it was not tried.
- Not examined: gender and age together, other attributes, whether the difference in default rates comes from the sample or from past lending, and whether using these inputs is legal in a given market.

## What did not work, and what is not established

- **Unclipped linear regression** (side study on the loan amount): one applicant with an income of 117,000,000 breaks the fit (RMSE 419,414, R² -0.16 against 347,500 and 0.255 clipped).
- **Deep decision trees** overfit: at depth 30 the training PR-AUC is 0.915 and the validation PR-AUC 0.101, and from depth 8 the log loss is worse than predicting the base rate.
- **kNN and SVM on this data**: kNN is behind logistic regression at every size (0.1698 against 0.2139 at 20,000 rows); the SVM is level with it from 5,000 rows while its fit time grows 119 times for 10 times the rows.
- **Naive Bayes** with its default settings gives 94.7% of applicants a probability above 0.5 and a log loss of 29.7, and more columns make the ranking worse (0.0830 PR-AUC with all 300 columns). A larger variance floor and calibration repair the probabilities but not the ranking.
- **PCA** hurt both logistic regression (0.1988 on 30 components against 0.2061 on the raw 70 columns) and LightGBM (0.2010 against 0.2303), and helped only Naive Bayes.
- **k-means** finds clusters whose default rates differ (3.9% to 13.0% at k = 20) but the applicants have no natural groups (silhouette 0.13 at k = 5).
- **Class weights, resampling and calibration** did not improve the ranking (above). The class-weighted LightGBM's odds are shifted by 2.34 on the log scale against the exact 2.43; the cause is that the weighted trees stretch the log-odds (slope 1.03 to 1.08 against 1.01 for logistic regression).
- **The tuned settings were found on the application columns and reused with the SQL features.** They were not tuned again on the 478-input model, and some of them sit at the edge of the range searched.
- **The calibration and threshold study used the application-column LightGBM**, not the best model. The threshold was re-checked on the best model's out-of-fold predictions (0.182 against a grid optimum of 0.19, same cost).
- **The holdout is higher than cross-validation for every model** and the reason was not isolated.
- **Limits.** The data has no application dates, so the split is random and says nothing about drift over time. The holdout is one sample of 61,503 applicants. The 45% and 10% cost figures are assumptions. The Kaggle leaderboard was not used. The model is not served, monitored or validated outside this dataset.
- **Process.** Commit `55dad14` was committed with 44 failing tests (a preprocessing change broke the clipped preprocessor) and fixed in the next commit (`50c6ebc`).

## Reproduce

See `README.md` for the order of the scripts, and `demo.ipynb` for the story in one notebook, which runs from the committed result files. The tests (`python -m pytest tests`) check every metric, the SQL feature tables, the leakage rules, the maths in `MATH.md` and the holdout guard.
