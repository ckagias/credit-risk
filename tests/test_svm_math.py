import numpy as np
import pytest

import svm_math


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    n = 400
    Z = rng.normal(size=(n, 4))
    y = (Z[:, 0] + 0.7 * rng.normal(size=n) > 1.0).astype(int)
    return Z, y


def test_hinge_loss_is_zero_beyond_the_margin_and_linear_inside():
    f = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
    assert svm_math.hinge(np.ones(5), f) == pytest.approx([0, 0, 0.5, 1, 2])
    assert svm_math.hinge(np.zeros(5), -f) == pytest.approx([0, 0, 0.5, 1, 2])


def test_balanced_weights_give_both_classes_the_same_total():
    y = np.array([1] * 10 + [0] * 90)
    w = svm_math.balanced_weights(y)
    assert w[y == 1].sum() == pytest.approx(w[y == 0].sum())
    assert w.sum() == pytest.approx(len(y))


def test_decision_function_equals_the_kernel_sum_over_the_support_vectors(small):
    Z, y = small
    model = svm_math.make_svc(Z, 1.0).fit(Z, y)
    assert svm_math.decision_by_hand(model, Z, Z) == pytest.approx(model.decision_function(Z), abs=1e-8)


@pytest.mark.parametrize("C", [0.1, 1, 10])
def test_primal_and_dual_objectives_meet_at_the_optimum(small, C):
    Z, y = small
    model = svm_math.make_svc(Z, C).fit(Z, y)
    primal, dual, w2 = svm_math.primal_and_dual(model, Z, y, C)
    assert w2 > 0
    assert primal >= dual - 1e-6
    assert primal == pytest.approx(dual, rel=5e-3)


def test_a_larger_c_gives_fewer_points_at_the_upper_bound_and_a_narrower_margin(small):
    Z, y = small
    widths, bounded = [], []
    for C in (0.1, 10):
        model = svm_math.make_svc(Z, C).fit(Z, y)
        widths.append(2 / np.sqrt(svm_math.squared_norm_of_w(model, Z)))
        weights = svm_math.balanced_weights(y)
        bounded.append((np.abs(model.dual_coef_[0]) >= C * weights[model.support_] - 1e-8).sum())
    assert widths[0] > widths[1]
    assert bounded[0] > bounded[1]
