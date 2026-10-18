import numpy as np

EPS = 1e-15


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def penalty_weight(C, n):
    return 1.0 / (C * n)


def log_loss_of(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def objective(Z, y, w, b, lam):
    p = sigmoid(Z @ w + b)
    return log_loss_of(y, p) + 0.5 * lam * float(w @ w)


def gradient(Z, y, w, b, lam):
    residual = sigmoid(Z @ w + b) - y
    n = len(y)
    return Z.T @ residual / n + lam * w, float(residual.sum() / n)


def largest_curvature(Z, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=Z.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(steps):
        v = Z.T @ (Z @ v) / len(Z)
        value = np.linalg.norm(v)
        v /= value
    return float(value)


def gradient_descent(Z, y, lam, steps, learning_rate=None, record_every=1):
    n, d = Z.shape
    if learning_rate is None:
        learning_rate = 1.0 / (0.25 * (largest_curvature(Z) + 1.0) + lam)
    w, b = np.zeros(d), 0.0
    history = []
    for step in range(steps + 1):
        if step % record_every == 0:
            p = sigmoid(Z @ w + b)
            history.append((step, log_loss_of(y, p), log_loss_of(y, p) + 0.5 * lam * float(w @ w)))
        grad_w, grad_b = gradient(Z, y, w, b, lam)
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
    return w, b, np.array(history), learning_rate
