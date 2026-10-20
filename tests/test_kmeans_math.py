import numpy as np
import pandas as pd
import pytest

import kmeans_math


@pytest.fixture
def Z():
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, -3.0]])
    return centers[rng.integers(0, 4, size=1200)] + rng.normal(scale=1.0, size=(1200, 2))


def test_the_objective_never_rises_through_assign_and_update_steps(Z):
    for init in kmeans_math.random_starts(Z, k=4, starts=4):
        _, _, history = kmeans_math.lloyd(Z, init)
        table = pd.DataFrame(history)
        assert (table["after_update"] <= table["after_assign"] + 1e-9).all()
        assert (table["after_assign"].iloc[1:].to_numpy() <= table["after_update"].iloc[:-1].to_numpy() + 1e-9).all()


def test_the_hand_written_loop_reaches_the_same_objective_as_scikit_learn_from_the_same_start(Z):
    for init in kmeans_math.random_starts(Z, k=4, starts=3):
        centers, labels, _ = kmeans_math.lloyd(Z, init)
        assert kmeans_math.objective(Z, centers, labels) == pytest.approx(kmeans_math.fit_kmeans_from(Z, init), rel=1e-6)


def test_squared_distances_match_the_direct_sum(Z):
    centers = Z[:4]
    direct = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assert kmeans_math.squared_distances(Z, centers) == pytest.approx(direct, abs=1e-9)


def test_the_loop_stops_when_no_point_changes_cluster(Z):
    _, labels, history = kmeans_math.lloyd(Z, kmeans_math.random_starts(Z, k=4, starts=1)[0])
    assert len(history) < kmeans_math.MAX_ITERATIONS
    assert history[-1]["after_assign"] == pytest.approx(history[-1]["after_update"])
