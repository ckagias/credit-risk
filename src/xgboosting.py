from pathlib import Path

import xgboost as xgb

import boosting
import evaluate
from baselines import THRESHOLD
from boosting import MAX_ROUNDS, PATIENCE, SEED, stopping_split

CURVE = Path("results/xgboost_curve.csv")
STOPPED = Path("results/xgboost_stopped.csv")


def make_booster(learning_rate, n_estimators=MAX_ROUNDS, **params):
    return xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=SEED, **params)


def fit_with_early_stopping(Z, y, learning_rate, patience=PATIENCE, max_rounds=MAX_ROUNDS):
    Z_fit, Z_stop, y_fit, y_stop = stopping_split(Z, y)
    booster = make_booster(learning_rate, max_rounds, early_stopping_rounds=patience, eval_metric="logloss")
    return booster.fit(Z_fit, y_fit, eval_set=[(Z_stop, y_stop)], verbose=False)


def fit_fixed_rounds(Z, y, learning_rate, rounds):
    Z_fit, _, y_fit, _ = stopping_split(Z, y)
    return make_booster(learning_rate, rounds).fit(Z_fit, y_fit)


def score_at(booster, Z, y, rounds):
    return evaluate.score(y, booster.predict_proba(Z, iteration_range=(0, rounds))[:, 1], THRESHOLD)


BACKEND = boosting.Backend(fit_with_early_stopping, fit_fixed_rounds, score_at, lambda booster: booster.best_iteration + 1)


def main():
    boosting.run(BACKEND, CURVE, STOPPED)


if __name__ == "__main__":
    main()
