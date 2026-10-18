import sqlite3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_features, make_preprocessor
from split import DB

TARGET = "AMT_CREDIT"
DROPPED = ["AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"]
RESULTS = Path("results/linear_regression.csv")
RESIDUALS = Path("results/regression_residuals.csv")
SEED = 42
SAMPLE = 10_000
RIDGE_ALPHAS = [1, 100, 1e3, 1e4, 1e5, 1e6]
LASSO_ALPHAS = [100, 300, 1e3, 3e3, 1e4]


def normal_equation(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(X1.T @ X1, X1.T @ y)


def ridge_closed_form(X, y, alpha):
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc = X - x_mean
    weights = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    return y_mean - x_mean @ weights, weights


def models():
    found = {("OLS", 0.0): LinearRegression()}
    for alpha in RIDGE_ALPHAS:
        found[("Ridge", alpha)] = Ridge(alpha=alpha)
    for alpha in LASSO_ALPHAS:
        found[("Lasso", alpha)] = Lasso(alpha=alpha, max_iter=3000)
    return found


def load_regression_data(con):
    X, _ = load_features(con, "train")
    return X.drop(columns=DROPPED), X[TARGET].to_numpy()


def score_fold(fold, val_idx, X_fit, y_fit, X_val, y_val, clip):
    pre = make_preprocessor(clip=clip).fit(X_fit)
    Z_fit, Z_val = pre.transform(X_fit), pre.transform(X_val)
    rows, predictions = [], {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for (name, alpha), model in models().items():
            model.fit(Z_fit, y_fit)
            predicted = model.predict(Z_val)
            predictions[(name, alpha)] = predicted
            rows.append(
                {
                    "fold": fold,
                    "clip": clip,
                    "model": name,
                    "alpha": alpha,
                    "rmse": float(np.sqrt(mean_squared_error(y_val, predicted))),
                    "mae": float(mean_absolute_error(y_val, predicted)),
                    "r2": float(r2_score(y_val, predicted)),
                    "nonzero": int(np.count_nonzero(model.coef_)),
                }
            )
    unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return rows, val_idx, predictions, unconverged


def cross_validate(X, y, clip, n_jobs=5):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, pd.qcut(y, 10, labels=False))
    jobs = (
        delayed(score_fold)(k, val_idx, X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx], clip)
        for k, (fit_idx, val_idx) in enumerate(folds)
    )
    results = Parallel(n_jobs=n_jobs)(jobs)
    rows = pd.DataFrame([row for fold_rows, *_ in results for row in fold_rows])
    out_of_fold = {key: np.empty(len(y)) for key in models()}
    for _, val_idx, predictions, _ in results:
        for key, predicted in predictions.items():
            out_of_fold[key][val_idx] = predicted
    return rows, out_of_fold, sum(r[3] for r in results)


def summarize(rows):
    table = rows.groupby(["clip", "model", "alpha"])[["rmse", "mae", "r2", "nonzero"]].agg(["mean", "std"])
    table.columns = ["_".join(c) for c in table.columns]
    return table.reset_index()


def main():
    con = sqlite3.connect(DB)
    X, y = load_regression_data(con)
    con.close()
    print(f"{len(y):,} applicants, {X.shape[1]} columns, target {TARGET} mean {y.mean():,.0f}")
    tables, kept = [], {}
    for clip in (False, True):
        start = time.time()
        rows, out_of_fold, unconverged = cross_validate(X, y, clip)
        print(f"clip={clip}: {time.time() - start:.0f}s, Lasso fits that hit the iteration limit: {unconverged}")
        tables.append(summarize(rows))
        kept[clip] = out_of_fold
        print(rows[(rows["model"] == "OLS")][["fold", "rmse", "mae", "r2"]].round(0).to_string(index=False))
    table = pd.concat(tables, ignore_index=True)
    print(table.round(4).to_string(index=False))
    RESULTS.parent.mkdir(exist_ok=True)
    table.to_csv(RESULTS, index=False, float_format="%.6f")
    clipped = table[table["clip"]]
    best = clipped.loc[clipped["rmse_mean"].idxmin()]
    key = (best["model"], float(best["alpha"]))
    print(f"best clipped model by RMSE: {key}")
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(y), SAMPLE, replace=False))
    fitted = kept[True][key][pick]
    pd.DataFrame({"fitted": fitted, "actual": y[pick], "residual": y[pick] - fitted}).to_csv(RESIDUALS, index=False, float_format="%.2f")
    print("RMSE of predicting the mean for everyone (std of the target):", round(float(y.std())))


if __name__ == "__main__":
    main()
