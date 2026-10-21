import numpy as np


def fit_gaussian(X, y, var_smoothing=0.0):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    epsilon = var_smoothing * X.var(axis=0).max()
    params = {"prior": float(y.mean()), "epsilon": float(epsilon)}
    for label in (0, 1):
        rows = X[y == label]
        params[f"mean_{label}"] = rows.mean(axis=0)
        params[f"var_{label}"] = rows.var(axis=0) + epsilon
    return params


def log_gaussian(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)


def evidence(params, X):
    X = np.asarray(X, dtype=float)
    return log_gaussian(X, params["mean_1"], params["var_1"]) - log_gaussian(X, params["mean_0"], params["var_0"])


def log_odds(params, X):
    prior = np.log(params["prior"] / (1 - params["prior"]))
    return prior + evidence(params, X).sum(axis=1)
