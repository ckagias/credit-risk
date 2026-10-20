import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import knn_math


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 6))
    y = (rng.random(400) < 0.3).astype(int)
    return Z, y, rng.normal(size=(50, 6))


def test_squared_distance_by_the_expanded_formula_matches_the_direct_sum(data):
    Z, _, Q = data
    direct = ((Q[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    assert knn_math.squared_distances(Q, Z) == pytest.approx(direct, abs=1e-9)


@pytest.mark.parametrize("k", [1, 5, 25])
def test_probability_is_the_share_of_defaulters_among_the_k_nearest_and_matches_scikit_learn(data, k):
    Z, y, Q = data
    by_hand = knn_math.knn_probability(Z, y, Q, k)
    library = KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(Z, y).predict_proba(Q)[:, 1]
    assert by_hand == pytest.approx(library)


def test_the_nearest_neighbour_gets_relatively_farther_as_columns_are_added():
    rng = np.random.default_rng(1)
    part = {"Z_fit": rng.normal(size=(3000, 60)), "y_fit": (rng.random(3000) < 0.1).astype(int), "Z_val": rng.normal(size=(200, 60))}
    import pandas as pd

    table = pd.DataFrame(knn_math.concentration_rows(part, column_counts=[1, 5, 20, 60], draws=2, fit_rows=3000, query_rows=100))
    ratio = table.groupby("columns")["ratio"].mean()
    assert (np.diff(ratio.to_numpy()) > 0).all()
    assert ratio.iloc[0] < 0.01 and ratio.iloc[-1] > 0.5
