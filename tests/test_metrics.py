import numpy as np
import pytest

from dlquanttime.metrics import icc, pearson_r


def test_pearson_r_identical_arrays_is_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    assert pearson_r(a, a) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation():
    a = np.arange(10, dtype=float)
    b = -a
    assert pearson_r(a, b) == pytest.approx(-1.0)


def test_pearson_r_uncorrelated_constant_returns_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([5.0, 5.0, 5.0])
    assert pearson_r(a, b) == 0.0


def test_pearson_r_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pearson_r([1, 2, 3], [1, 2])


def test_icc_identical_arrays_is_one():
    rng = np.random.default_rng(1)
    a = rng.normal(size=50)
    assert icc(a, a) == pytest.approx(1.0, abs=1e-6)


def test_icc_systematic_scale_difference_is_low():
    rng = np.random.default_rng(2)
    a = rng.normal(size=50)
    b = a * 2 + 5  # perfectly correlated but poor absolute agreement
    assert pearson_r(a, b) == pytest.approx(1.0)
    assert icc(a, b) < 0.5


def test_icc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        icc([1, 2, 3], [1, 2])
