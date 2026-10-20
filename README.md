# Credit Risk Prediction

Classic machine learning algorithms (linear models, trees, boosting and more) compared with the same metrics and the same cross-validation, with features built in SQL. The aim is to understand each algorithm: its loss, how it learns and where it fails. Data is Home Credit Default Risk (307,511 loan applications, 8.1% default).

Status: in progress. Setup, data, evaluation, linear models, trees and ensembles, the SQL features, kNN and SVM, Naive Bayes, PCA, k-means, class imbalance, calibration, the decision threshold and the maths of Phases 6 and 7 are done.

## Data

Home Credit Default Risk, downloaded from Kaggle into `data/raw/` (not tracked).

| Table | Rows | Size |
|---|---:|---:|
| `application_train` | 307,511 | 166 MB |
| `application_test` | 48,744 | 27 MB |
| `bureau` | 1,716,428 | 170 MB |
| `bureau_balance` | 27,299,925 | 376 MB |
| `previous_application` | 1,670,214 | 405 MB |
| `POS_CASH_balance` | 10,001,358 | 393 MB |
| `installments_payments` | 13,605,401 | 723 MB |
| `credit_card_balance` | 3,840,312 | 425 MB |

## Features from SQL

`sql/` holds one file per feature group. `src/features.py` builds each into a `feat_*` table with one row for each of the 307,511 applicants and checks that count. `src/feature_gain.py` adds one group at a time to the application columns and scores it with the same 5-fold cross-validation (PR-AUC, results in `results/feature_gain.csv`).

| Group | Columns | LightGBM PR-AUC gain | Logistic PR-AUC gain |
|---|---:|---:|---:|
| bureau, with `bureau_balance` | 25 | +0.0110 | +0.0088 |
| previous_application | 23 | +0.0107 | +0.0120 |
| POS_CASH_balance | 13 | +0.0094 | +0.0069 |
| installments_payments | 14 | +0.0104 | +0.0086 |
| credit_card_balance | 17 | +0.0068 | +0.0074 |
| application ratios | 14 | +0.0083 | +0.0036 |
| all six | 106 | +0.0371 | +0.0356 |

Base PR-AUC is 0.2385 for LightGBM and 0.2250 for logistic regression. With all six groups it is 0.2756 and 0.2606, and 0.2813 for LightGBM with the tuned parameters. Every group improves PR-AUC in all 5 folds.

Build the tables from the project root with `PYTHONPATH=src python src/features.py`. SQLite needs a large page cache for the 10 and 13.6 million row tables (set in `build_all`): the POS_CASH build took 72 s with it and did not finish in 55 minutes without it.

## kNN and SVM

`src/knn_svm.py` fits kNN, an RBF SVM and logistic regression on stratified subsamples of each training fold (2,000 to 20,000 rows of about 197,000) and scores them on the whole validation fold, base application columns only. PR-AUC, mean of 5 folds:

| Training rows | kNN (k = 500) | SVM (balanced weights) | Logistic | SVM fit time | Logistic fit time |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.1321 | 0.1779 | 0.1706 | 0.7 s | 0.5 s |
| 5,000 | 0.1501 | 0.1978 | 0.1951 | 5.1 s | 1.1 s |
| 10,000 | 0.1593 | 0.2069 | 0.2061 | 20.4 s | 1.4 s |
| 20,000 | 0.1698 | 0.2123 | 0.2139 | 84.7 s | 2.1 s |

kNN is behind logistic regression in every fold at every size. The SVM is level with logistic regression from 5,000 rows up, with a fit time that grows 119 times for 10 times the rows. Without class weights the SVM scores 0.114 at 5,000 rows. kNN on only the three `EXT_SOURCE` columns scores 0.1915 against 0.1698 on all columns. Results in `results/knn_svm.csv` and `results/knn_ext.csv`.

## Naive Bayes

`src/naive_bayes.py` fits `GaussianNB` on the 246,008 training rows with the same 5 folds. `var_smoothing` is chosen by cross-validation and a sigmoid calibration follows. PR-AUC, mean of 5 folds (base rate 0.0807, one-feature logistic baseline 0.1889):

| Columns | Default `var_smoothing` | Best `var_smoothing`, calibrated |
|---|---:|---:|
| all 300 | 0.0830 | 0.1461 |
| 70 continuous | 0.1431 | 0.1672 |
| 3 `EXT_SOURCE` | 0.1942 | 0.1971 |

With the defaults Naive Bayes gives 94.7% of applicants a probability above 0.5 (log loss 29.7). Dropping the 0/1 columns, flattening the likelihoods with a larger `var_smoothing` and calibrating repairs the probabilities, and more columns still make the ranking worse. Results in `results/naive_bayes.csv`.

## PCA

`src/pca.py` fits PCA on the 70 continuous application columns of each training fold and scores models on the top components. Chart in `docs/img/pca.png`.

![PCA: scree plot, cumulative variance and PR-AUC against components](docs/img/pca.png)

The first component holds 26.0% of the variance and loads on the `_AVG`, `_MEDI` and `_MODE` copies of the housing columns. 18 components keep 80% of the variance, 26 keep 90% and 30 keep 95%. PR-AUC with 30 components against the raw 70 columns: logistic regression 0.1988 against 0.2061, LightGBM 0.2010 against 0.2303, Naive Bayes 0.1582 against 0.1431. PCA helps only Naive Bayes, whose columns it decorrelates. Results in `results/pca_variance.csv`, `results/pca_models.csv` and `results/pca_loadings.csv`.

## k-means

`src/kmeans.py` clusters the 70 continuous application columns of each training fold for k from 2 to 20, assigns the validation fold to the nearest centroid, and compares default rates per cluster. Chart in `docs/img/kmeans.png`.

![k-means: inertia and silhouette, default rate per cluster, cluster rate as a risk score](docs/img/kmeans.png)

The clusters do differ in default rate: at k = 5 the lowest and highest validation rates average 5.1% and 9.8% over the folds against 8.1% overall (chi-square p below 0.0001 at every k), and the gap widens with k, from 5.6% and 8.3% at k = 2 to 3.9% and 13.0% at k = 20. The applicants have no natural groups: the silhouette score is 0.49 at k = 2, where a small group of outliers splits off, 0.13 at k = 5 and 0.05 at k = 20, and inertia has no elbow. At k = 5 the clusters are described by loan size, by how much housing data the applicant has and by one cluster of 533 applicants with a bad `YEARS_BEGINEXPLUATATION` value. The cluster's default rate as a risk score gives ROC-AUC 0.549 at k = 5 and 0.607 at k = 20, against 0.5 for no ranking. Results in `results/kmeans_summary.csv`, `results/kmeans_clusters.csv` and `results/kmeans_profile.csv`.

## Class imbalance

`src/imbalance.py` compares doing nothing with balanced class weights, random undersampling, random oversampling and SMOTE (`imbalanced-learn`) for logistic regression and LightGBM (learning rate 0.05, defaults, early stopping on PR-AUC of a 10% part cut off before resampling). Resampling is applied to the training part of each fold only. Chart in `docs/img/imbalance.png`.

![PR-AUC difference from doing nothing, and log loss before and after undoing the odds shift](docs/img/imbalance.png)

| Model | Method | PR-AUC | Difference from nothing (folds better) | Log loss | Log loss after the odds correction | Mean probability |
|---|---|---:|---:|---:|---:|---:|
| Logistic | nothing | 0.2250 | | 0.2495 | 0.2495 | 0.081 |
| Logistic | class weights | 0.2231 | -0.0019 (0 of 5) | 0.5919 | 0.2496 | 0.420 |
| Logistic | undersample | 0.2213 | -0.0037 (0 of 5) | 0.5951 | 0.2501 | 0.423 |
| Logistic | oversample | 0.2226 | -0.0024 (0 of 5) | 0.5918 | 0.2497 | 0.420 |
| Logistic | SMOTE | 0.2197 | -0.0053 (0 of 5) | 0.5809 | 0.2526 | 0.402 |
| LightGBM | nothing | 0.2386 | | 0.2470 | 0.2470 | 0.080 |
| LightGBM | class weights | 0.2386 | -0.0000 (3 of 5) | 0.5584 | 0.2475 | 0.401 |
| LightGBM | undersample | 0.2308 | -0.0078 (0 of 5) | 0.5902 | 0.2483 | 0.422 |
| LightGBM | oversample | 0.2393 | +0.0007 (3 of 5) | 0.5581 | 0.2474 | 0.401 |
| LightGBM | SMOTE | 0.2355 | -0.0031 (1 of 5) | 0.2475 | 0.3594 | 0.085 |

No method improves PR-AUC or ROC-AUC beyond fold-to-fold noise: the largest gain is +0.0007 (std 0.0031), and seven of the eight changes are zero or negative. Class weights and resampling raise the average predicted probability from 0.08 to about 0.40, put about a third of the applicants above 0.5 (0.2% when doing nothing) and more than double the log loss (from 0.25 to between 0.56 and 0.60). Multiplying the odds by the known factor (the true ratio of defaults to repaid loans, 0.088) gives the log loss back to within 0.0013 of doing nothing for every method except SMOTE on logistic regression (0.0031 higher). LightGBM on SMOTE rows is the exception: 99.8% of the synthetic rows have a non-0/1 value in one of the 230 binary columns, which no real applicant has, and its probabilities stay near the base rate (0.085), so the correction does not apply (0.3594). Undersampling is also the fastest for LightGBM (1.4 s against 4.8 s) and the weakest on PR-AUC. Results in `results/imbalance.csv`.

## Calibration

`src/calibration.py` fits each model on 80% of every training fold, fits a calibrator on the other 20%, and scores the validation fold. Platt scaling is a logistic regression on the log-odds of the predicted probability. Isotonic regression is a non-decreasing step function. The calibration curve pools the 5 validation folds into 10 equal-count groups. Chart in `docs/img/calibration.png`.

![Calibration curves before and after calibration, calibration error and PR-AUC](docs/img/calibration.png)

| Model | Calibrator | Log loss | Brier | ECE | PR-AUC | Mean probability |
|---|---|---:|---:|---:|---:|---:|
| Logistic | none | 0.2496 | 0.0686 | 0.0025 | 0.2246 | 0.0808 |
| Logistic | Platt | 0.2496 | 0.0686 | 0.0026 | 0.2246 | 0.0805 |
| Logistic | isotonic | 0.2512 | 0.0687 | 0.0031 | 0.2175 | 0.0806 |
| LightGBM | none | 0.2477 | 0.0682 | 0.0035 | 0.2357 | 0.0802 |
| LightGBM | Platt | 0.2476 | 0.0682 | 0.0032 | 0.2357 | 0.0805 |
| LightGBM | isotonic | 0.2491 | 0.0682 | 0.0029 | 0.2278 | 0.0805 |
| LightGBM, class weights | none | 0.5584 | 0.1901 | 0.3205 | 0.2363 | 0.4013 |
| LightGBM, class weights | Platt | 0.2475 | 0.0682 | 0.0038 | 0.2363 | 0.0806 |
| LightGBM, class weights | isotonic | 0.2488 | 0.0682 | 0.0040 | 0.2269 | 0.0805 |
| Forest | none | 0.2518 | 0.0691 | 0.0129 | 0.2256 | 0.0812 |
| Forest | Platt | 0.2501 | 0.0687 | 0.0027 | 0.2256 | 0.0806 |
| Forest | isotonic | 0.2517 | 0.0687 | 0.0039 | 0.2177 | 0.0807 |

Logistic regression and LightGBM trained on log loss are already calibrated, and neither calibrator improves them (the log loss changes by at most 0.0001 with Platt scaling). The random forest, with 200 trees, 30 columns per split and `min_samples_leaf` = 30, is too high for low-risk applicants and too low for the highest decile (0.209 predicted against 0.256 observed), and Platt scaling brings its ECE from 0.0129 to 0.0027 and its log loss from 0.2518 to 0.2501 (lower in 5 of 5 folds). LightGBM with balanced class weights keeps its ranking and its log loss goes from 0.5584 to 0.2475 with Platt scaling. Platt scaling keeps the order of the applicants, so it never changes PR-AUC or ROC-AUC. Isotonic regression reaches a similar calibration but lowers PR-AUC by 0.0070 to 0.0094 in every fold, because the steps make many applicants tie. Results in `results/calibration.csv` and `results/calibration_curve.csv`.

## Decision threshold

`src/threshold.py` turns the probabilities into refuse or accept from an assumed cost: a missed default costs `r` times a refused good loan, with `r` = 10 as the working assumption (a default loses most of the loan, a refused good loan loses its margin) and 2, 5 and 20 for sensitivity. For calibrated probabilities the lowest expected cost is at the threshold $1 / (1 + r)$ (0.0909 for `r` = 10). Each fold's cost is scored with the threshold chosen on the other 4 folds (`cv_tuned`, the lowest cost on a grid of 0.005 steps) and with the fixed rules. LightGBM at its defaults, learning rate 0.05, trained on 80% of each training fold (the rest is the calibration part). Chart in `docs/img/threshold.png`.

![Cost against threshold for four cost ratios, and for LightGBM with and without calibration](docs/img/threshold.png)

Cost per 1,000 applicants in units of one refused good loan, `r` = 10, mean of 5 folds:

| Rule | Threshold | Cost | Applicants refused | Defaulters caught |
|---|---:|---:|---:|---:|
| Refuse nobody | | 807.3 | 0% | 0% |
| Fixed 0.5 | 0.5 | 796.4 | 0.2% | 1.5% |
| Threshold that maximises F1 | 0.143 | 577.3 | 14.9% | 42.7% |
| Fixed 0.1 | 0.1 | 538.9 | 25.2% | 58.6% |
| Tuned on the other 4 folds | 0.090 | 535.2 | 28.7% | 62.9% |
| 1 / (1 + r) | 0.0909 | 535.4 | 28.3% | 62.5% |

- The formula and the tuned threshold have the same cost: the fold-by-fold difference is -0.13 (std 0.53) per 1,000 applicants, and the formula needs no tuning. The best threshold rises as the cost ratio falls, 0.335, 0.16, 0.09 and 0.05 for `r` = 2, 5, 10 and 20, and with `r` = 2 the cost is almost the same for every rule (157 against 161 for refusing nobody), so the model hardly helps.
- The threshold that maximises F1 costs 8% more than the formula (577.3 against 535.4, higher in 5 of 5 folds) and refuses half as many applicants, and fixed 0.5 saves only 1.3% over refusing nobody (796.4 against 807.3).
- The cost ratio is an assumption and the threshold depends on it. A threshold chosen for `r` = 10 costs 88.1% more than the best one if the true ratio is 2, 12.2% more if it is 5 and 15.5% more if it is 20. Choosing for 5 when the truth is 10 costs 11.2% more, and for 20 when the truth is 10, 10.2% more.
- The formula needs calibrated probabilities. For LightGBM with balanced class weights the lowest cost is at 0.535 (534.7), the odds shift of 11.4 times applied to 0.0909, and the formula's 0.0909 refuses 97.3% of the applicants at a cost of 894.6. After Platt scaling the formula works again (535.9).
- A calibrated model predicts its own cost: with the probabilities of the LightGBM the expected cost at the formula's threshold is 538.7 against a realised 535.4 (0.6% apart), and for `r` = 20 742.4 against 722.4. For the class-weighted model the expected cost (592.5) is far from the realised one (894.6).
- Costs are per applicant and equal for all loans. Weighting by loan size, a different cost ratio per loan, or the holdout, are not used here. Results in `results/threshold_folds.csv` and `results/threshold_curve.csv`.
