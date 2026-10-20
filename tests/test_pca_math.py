import numpy as np
import pandas as pd
import pytest

import pca
import pca_math


@pytest.fixture
def Z():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2000, 5))
    return base @ rng.normal(size=(5, 8)) + 0.3 * rng.normal(size=(2000, 8))


def test_eigenvalues_are_sorted_and_match_scikit_learn(Z):
    values, _ = pca_math.eigen_decomposition(Z)
    assert (np.diff(values) <= 0).all()
    assert values == pytest.approx(pca.fit_pca(Z).explained_variance_, abs=1e-8)


def test_eigenvectors_are_orthonormal(Z):
    _, vectors = pca_math.eigen_decomposition(Z)
    assert vectors.T @ vectors == pytest.approx(np.eye(Z.shape[1]), abs=1e-10)


def test_reconstruction_error_on_the_fit_rows_is_the_sum_of_the_discarded_eigenvalues(Z):
    rows = pd.DataFrame(pca_math.reconstruction_errors(Z, Z[:500]))
    assert rows["fit_error"].to_numpy() == pytest.approx(rows["tail_eigenvalues"].to_numpy(), abs=1e-8)


def test_error_is_the_total_variance_with_no_components_and_zero_with_all(Z):
    rows = pd.DataFrame(pca_math.reconstruction_errors(Z, Z))
    assert rows["fit_error"].iloc[0] == pytest.approx(((Z - Z.mean(axis=0)) ** 2).sum(axis=1).mean())
    assert rows["fit_error"].iloc[-1] == pytest.approx(0, abs=1e-8)
    assert (np.diff(rows["fit_error"]) <= 1e-9).all()


def test_decile_rows_cover_every_row_once_per_component(Z):
    y = (np.random.default_rng(1).random(len(Z)) < 0.1).astype(int)
    rows = pd.DataFrame(pca_math.decile_rows(Z, Z, y, components=2, bins=10))
    assert len(rows) == 20
    assert (rows.groupby("component")["n"].sum() == len(Z)).all()
