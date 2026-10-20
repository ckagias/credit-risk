import numpy as np
import pandas as pd
import pytest

import forest
import kmeans
import naive_bayes


@pytest.fixture
def parts():
    rng = np.random.default_rng(0)
    n = 1500
    group = rng.integers(0, 3, size=n)
    centers = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    points = centers[group] + rng.normal(scale=0.5, size=(n, 2))
    X = pd.DataFrame({"a": points[:, 0], "b": points[:, 1], "c": rng.normal(size=n)})
    y = (rng.random(n) < np.array([0.03, 0.10, 0.25])[group]).astype(int)
    return naive_bayes.continuous_parts(forest.prepare_folds(X, y, scale=True, clip=True))


def test_inertia_is_the_sum_of_squared_distances_to_the_assigned_centroid_worked_by_hand(parts):
    Z = parts[0]["Z_fit"]
    km = kmeans.fit_kmeans(Z, 3)
    by_hand = ((Z - km.cluster_centers_[km.labels_]) ** 2).sum()
    assert km.inertia_ == pytest.approx(by_hand, rel=1e-6)


def test_each_centroid_is_the_mean_of_its_points(parts):
    Z = parts[0]["Z_fit"]
    km = kmeans.fit_kmeans(Z, 3)
    for c in range(3):
        assert km.cluster_centers_[c] == pytest.approx(Z[km.labels_ == c].mean(axis=0), abs=1e-4)


def test_inertia_never_rises_with_more_clusters(parts):
    summary, _ = kmeans.sweep(parts[:1], grid=[2, 3, 5, 8])
    assert (np.diff(summary["inertia_per_row"]) < 0).all()


def test_cluster_table_rates_average_back_to_the_overall_rate(parts):
    part = parts[0]
    km = kmeans.fit_kmeans(part["Z_fit"], 4)
    table = kmeans.cluster_table(km.labels_, part["y_fit"], km.predict(part["Z_val"]), part["y_val"], 4)
    assert table["n_fit"].sum() == len(part["y_fit"])
    assert table["n_val"].sum() == len(part["y_val"])
    assert (table["n_fit"] * table["rate_fit"]).sum() / table["n_fit"].sum() == pytest.approx(part["y_fit"].mean())
    assert (table["n_val"] * table["rate_val"]).sum() / table["n_val"].sum() == pytest.approx(part["y_val"].mean())


def test_one_fold_and_one_k_give_one_summary_row_and_k_cluster_rows(parts):
    summary, table = kmeans.fold_rows(parts[0], 3)
    assert summary["k"] == 3
    assert len(table) == 3
    assert set(table["fold"]) == {parts[0]["fold"]}


def test_clusters_with_different_planted_default_rates_are_found_and_scored(parts):
    summary, _ = kmeans.sweep(parts, grid=[3])
    assert summary["rate_max_val"].mean() > 2 * summary["rate_min_val"].mean()
    assert (summary["chi_square_p_val"] < 0.01).all()
    assert summary["roc_auc"].mean() > 0.6


def test_chi_square_p_is_not_small_when_the_clusters_share_one_rate():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 4, size=4000)
    y = (rng.random(4000) < 0.1).astype(int)
    assert kmeans.chi_square_p(labels, y, 4) > 0.01


def test_silhouette_is_high_for_well_separated_blobs_and_low_for_noise():
    rng = np.random.default_rng(4)
    blobs = np.vstack([rng.normal(loc=-5, size=(300, 2)), rng.normal(loc=5, size=(300, 2))])
    noise = rng.normal(size=(600, 2))
    assert kmeans.silhouette(blobs, kmeans.fit_kmeans(blobs, 2).labels_) > 0.8
    assert kmeans.silhouette(noise, kmeans.fit_kmeans(noise, 2).labels_) < 0.5


def test_centroid_profile_names_the_column_that_separates_the_groups():
    rng = np.random.default_rng(5)
    n = 1200
    group = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"split": group * 6.0 + rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    y = (rng.random(n) < 0.1).astype(int)
    profile = kmeans.centroid_profile(X, y, k=2, per_cluster=1)
    assert set(profile["column"]) == {"split"}
    assert profile["size"].sum() == n
