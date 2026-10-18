import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import logistic_math as lm
from logistic import SEED, make_model
from preprocess import load_features, make_preprocessor
from split import DB

BEST_C = 0.03
STEPS = 5000
RECORD_EVERY = 25
SAMPLE = 20_000
GRADIENT_DESCENT = Path("results/gradient_descent.csv")
LOGITS = Path("results/logit_sample.csv")


def fit_both(Z, y):
    lam = lm.penalty_weight(BEST_C, len(y))
    library = LogisticRegression(C=BEST_C, l1_ratio=0.0, max_iter=2000).fit(Z, y)
    w_lib, b_lib = library.coef_.ravel(), float(library.intercept_[0])
    optimum = lm.objective(Z, y, w_lib, b_lib, lam)
    w, b, history, learning_rate = lm.gradient_descent(Z, y, lam, STEPS, record_every=RECORD_EVERY)
    return library, (w_lib, b_lib), optimum, (w, b), history, learning_rate, lam


def out_of_fold_logits(X, y):
    logits = np.empty(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)
    for fit_idx, val_idx in folds:
        model = make_model(C=BEST_C).fit(X.iloc[fit_idx], y[fit_idx])
        logits[val_idx] = model.decision_function(X.iloc[val_idx])
    return logits


def main():
    con = sqlite3.connect(DB)
    X, y = load_features(con, "train")
    con.close()
    Z = make_preprocessor().fit(X).transform(X).astype(np.float64)
    start = time.time()
    library, (w_lib, b_lib), optimum, (w, b), history, learning_rate, lam = fit_both(Z, y)
    print(f"gradient descent {STEPS} steps, learning rate {learning_rate:.4f}, lambda {lam:.3e}: {time.time() - start:.0f}s")
    frame = pd.DataFrame(history, columns=["step", "log_loss", "objective"])
    frame["library_optimum"] = optimum
    GRADIENT_DESCENT.parent.mkdir(exist_ok=True)
    frame.to_csv(GRADIENT_DESCENT, index=False, float_format="%.8f")
    p_lib, p_gd = lm.sigmoid(Z @ w_lib + b_lib), lm.sigmoid(Z @ w + b)
    print(f"library optimum {optimum:.6f}, gradient descent after {STEPS} steps {frame['objective'].iloc[-1]:.6f}, gap {frame['objective'].iloc[-1] - optimum:.2e}")
    print(f"weights: correlation {np.corrcoef(w, w_lib)[0, 1]:.4f}, largest difference {np.abs(w - w_lib).max():.4f}, intercept {b:.4f} vs {b_lib:.4f}")
    print(f"predicted probabilities: largest difference {np.abs(p_gd - p_lib).max():.5f}, correlation {np.corrcoef(p_gd, p_lib)[0, 1]:.6f}")
    logits = out_of_fold_logits(X, y)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(y), SAMPLE, replace=False))
    pd.DataFrame({"logit": logits[pick], "default": y[pick]}).to_csv(LOGITS, index=False, float_format="%.5f")
    print("out-of-fold logits: range", round(logits.min(), 2), "to", round(logits.max(), 2), "| share of applicants with probability above 0.5:", round(float((logits > 0).mean()), 5))


if __name__ == "__main__":
    main()
