# Credit Risk Prediction

Classic machine learning algorithms (linear models, trees, boosting and more) compared with the same metrics and the same cross-validation, with features built in SQL. The aim is to understand each algorithm: its loss, how it learns and where it fails. Data is Home Credit Default Risk (307,511 loan applications, 8.1% default).

Status: in progress. Setup, data, evaluation, linear models, trees and ensembles, and the SQL features are done.

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
