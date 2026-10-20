import numpy as np
import pandas as pd
import pytest

import forest
import naive_bayes
import pca


@pytest.fixture
def parts():
    rng = np.random.default_rng(0)
    n = 1500
    base = rng.normal(size=(n, 3))
    X = pd.DataFrame(
        {
            "a": base[:, 0],
            "b": base[:, 0] * 0.9 + rng.normal(scale=0.3, size=n),
            "c": base[:, 1],
            "d": base[:, 2],
            "e": base[:, 2] + rng.normal(scale=0.5, size=n),
            "f": rng.normal(size=n),
        }
    )
    logit = -1.5 + 1.2 * base[:, 0] - 0.8 * base[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return naive_bayes.continuous_parts(forest.prepare_folds(X, y, scale=True, clip=True))


def test_explained_ratios_of_a_fold_add_to_one_and_never_increase(parts):
    rows = pd.DataFrame(pca.variance_rows(parts))
    for _, fold in rows.groupby("fold"):
        assert fold["explained_ratio"].sum() == pytest.approx(1.0)
        assert (np.diff(fold["explained_ratio"]) <= 1e-12).all()
        assert fold["cumulative"].iloc[-1] == pytest.approx(1.0)


def test_one_fold_gives_one_row_per_component(parts):
    rows = pd.DataFrame(pca.variance_rows(parts))
    assert len(rows) == 5 * parts[0]["Z_fit"].shape[1]


def test_variance_ratios_match_the_eigenvalues_of_the_covariance_matrix_worked_by_hand(parts):
    Z = parts[0]["Z_fit"]
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(Z, rowvar=False)))[::-1]
    assert pca.fit_pca(Z).explained_variance_ratio_ == pytest.approx(eigenvalues / eigenvalues.sum(), abs=1e-6)


def test_component_scores_are_uncorrelated(parts):
    Z = parts[0]["Z_fit"]
    scores = pca.fit_pca(Z).transform(Z)
    correlation = np.corrcoef(scores, rowvar=False)
    assert np.abs(correlation - np.eye(correlation.shape[0])).max() < 1e-6


def test_components_needed_counts_from_the_mean_cumulative_curve():
    variance = [{"fold": f, "component": i, "cumulative": c} for f in range(2) for i, c in enumerate([0.5, 0.8, 0.95, 1.0], start=1)]
    assert pca.components_needed(variance, targets=[0.5, 0.8, 0.9, 1.0]) == {0.5: 1, 0.8: 2, 0.9: 3, 1.0: 4}


def test_every_model_is_scored_on_the_raw_columns_and_on_each_number_of_components(parts):
    n = parts[0]["Z_fit"].shape[1]
    rows = pd.DataFrame(pca.model_rows(parts[:2], components=[2, 3], models=("logistic", "naive_bayes")))
    assert len(rows) == 2 * 2 * (1 + 2 + 1)
    assert set(rows["representation"]) == {"raw", "pca"}
    assert set(rows.loc[rows["representation"] == "pca", "n_columns"]) == {2, 3, n}


def test_logistic_regression_on_all_components_equals_it_on_the_raw_columns(parts):
    rows = pd.DataFrame(pca.model_rows(parts[:1], components=[2], models=("logistic",)))
    n = parts[0]["Z_fit"].shape[1]
    raw = rows[rows["representation"] == "raw"]["pr_auc"].iloc[0]
    full = rows[(rows["representation"] == "pca") & (rows["n_columns"] == n)]["pr_auc"].iloc[0]
    assert full == pytest.approx(raw, abs=2e-4)


def test_two_components_hold_less_signal_than_all_of_them(parts):
    rows = pd.DataFrame(pca.model_rows(parts, components=[1], models=("logistic",)))
    mean = rows.groupby(["representation", "n_columns"])["roc_auc"].mean()
    n = parts[0]["Z_fit"].shape[1]
    assert mean[("pca", n)] > mean[("pca", 1)] + 0.02


def test_loadings_of_a_toy_table_name_the_correlated_pair_on_the_first_component():
    rng = np.random.default_rng(1)
    n = 800
    shared = rng.normal(size=n)
    X = pd.DataFrame({"x1": shared, "x2": shared + rng.normal(scale=0.1, size=n), "x3": rng.normal(size=n), "x4": rng.normal(size=n)})
    y = (rng.random(n) < 0.2).astype(int)
    table = pca.loadings(X, y)
    first = table[(table["component"] == 1) & (table["rank"] <= 2)]
    assert set(first["column"]) == {"x1", "x2"}
