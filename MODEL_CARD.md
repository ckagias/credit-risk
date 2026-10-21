# Model card: credit default risk, LightGBM

## Model

- **What it is.** A LightGBM gradient-boosted tree classifier that outputs the probability that a loan applicant defaults, behind a preprocessing pipeline (median imputation with missing indicators, one-hot encoding, replacement of placeholder codes). Logged in MLflow as `credit-risk-lightgbm` (alias `champion`) with a signature of 226 input columns and 2 output probabilities.
- **Settings.** 9 leaves per tree, at least 206 applicants per leaf, row sampling 0.85, column sampling 0.56, L1 penalty 0.024, L2 penalty 7.5, learning rate 0.05, stopped by early stopping on a 10% part of the training data (1,525 rounds in the final fit). Found by 60 Optuna trials on the 300 application inputs and reused with the SQL features.
- **Inputs.** The 120 application columns plus 106 columns built in SQL from the credit bureau, previous applications, POS cash, instalments and credit card tables, 478 model inputs in all.
- **Code.** `src/` in this repository; the final fit is `src/tracking.py final` and the scoring is `src/holdout.py`.

## Intended use

A study of how well loan default can be ranked from application and credit-history data, and of what such a model does and does not do. It was built to compare algorithms and to practise evaluating, explaining and checking a model.

## Not intended for

- Deciding real loan applications. The model has not been validated on any lender's applicants, on any period of time, or outside this dataset, and the cost used for its threshold is assumed.
- Any use of gender or age as an input to a decision about a person. The model uses both directly (`CODE_GENDER`, `DAYS_BIRTH`). The rules on using gender and age in lending differ between markets and gender is restricted or not allowed as an input in many of them; this project does not assess the legal position.
- Use without monitoring: nothing here detects drift in the inputs or the default rate.

## Data

Home Credit Default Risk (Kaggle): 307,511 applicants with a default label (8.07% default) and six tables of credit history. The applicants were split 80/20, stratified, with a fixed seed, into 246,008 for training and tuning and 61,503 for a holdout that was scored once. The data has no application dates, so the split is random and not by time. Whether the applicants in the data were selected by an earlier approval decision was not examined.

## Performance

On the holdout (95% bootstrap intervals): PR-AUC 0.2957 (0.2835 to 0.3089), ROC-AUC 0.7918 (0.7852 to 0.7986), log loss 0.2347 (0.2303 to 0.2394). The base rate gives PR-AUC 0.0807, logistic regression 0.2731 (0.2622 to 0.2860) and a random forest 0.2605 (0.2491 to 0.2735). In 5-fold cross-validation on the training part LightGBM scored 0.2813 (std 0.0045); the holdout score is higher for every model and the reason was not isolated.

The probabilities are calibrated: expected calibration error 0.0025 out of fold, and the mean predicted probability is within 0.002 of the default rate in every segment of 1,000 applicants or more.

At the decision threshold 0.182, chosen from an assumed cost (a missed default loses 45% of the credit amount, a refused good loan a 10% margin), the model refuses 11.0% of applicants and 41.0% of defaulters on the holdout, and costs 18.0% (16.3% to 19.6%) less than approving everybody. The cost figures are assumptions: the threshold and this saving change with them.

## What drives the predictions

The mean of the three `EXT_SOURCE` scores (external credit scores) dominates: shuffling it costs 0.044 PR-AUC, six times the next column. The SQL feature groups hold 62.6% of the SHAP importance. Higher annuity, annuity-to-income ratio, credit-to-goods ratio, share of late instalments and number of future instalments raise the predicted risk; a higher `EXT_SOURCE` mean lowers it.

## Where it is weaker

- **Age.** ROC-AUC 0.745 (0.733 to 0.759) for applicants aged 60 and over and 0.764 (0.756 to 0.772) under 30, against 0.789 overall. At 60 and over it misses 82.3% of defaulters; under 30 it refuses 15.8% of good loans.
- **Revolving loans** gain little from the model (6.7% of cost saved against 17.1% for cash loans), and applicants **without a bureau record** rank worse (ROC-AUC 0.777 against 0.790).
- Applicants with none of the three `EXT_SOURCE` scores (0.06% of applicants, 139 out of 246,008) are too few to judge.

## Fairness findings

Out of fold at the threshold 0.182, the scores are calibrated within every gender and age group, but the error rates differ:

| Group | Default rate | Approved | Defaulters refused | Good loans refused |
|---|---:|---:|---:|---:|
| Women | 7.0% | 91.5% | 35.1% | 6.5% |
| Men | 10.1% | 84.3% | 46.5% | 12.2% |
| Under 30 | 11.5% | 80.3% | 49.6% | 15.8% |
| 60 and over | 4.9% | 96.8% | 17.7% | 2.4% |

The approval ratio is 0.922 for men against women and 0.830 for under 30 against 60 and over, both above the four-fifths rule of thumb of 0.8. Removing the gender column leaves PR-AUC unchanged (0.2809) and narrows the gap between men and women (approval ratio 0.945) without closing it, and breaks calibration by gender. Age was not removed. Gender and age together, other attributes, and the legal position were not examined. These results describe this model on this sample; they do not show that it is fair or unfair.

## Limits

- One holdout sample of 61,503 applicants and one fit of each model.
- No time dimension, no validation on another lender or another period.
- The tuned settings were found on the application inputs, not the final 478.
- The registered model is a refit with the settings used for the holdout; the holdout predictions were not kept, so the two fits were not compared directly.
- The cost assumptions are not data. The Kaggle leaderboard was not used.

## Details

`results.md` has every result and what did not work, `MATH.md` the maths and `README.md` how to reproduce it.
