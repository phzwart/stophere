import pytest

from experiments.phase3_random_baseline_audit import (
    success_probability_without_replacement,
)


def test_exact_random_probability_boundaries():
    assert success_probability_without_replacement(100, 10, 0) == 0.0
    assert success_probability_without_replacement(100, 0, 20) == 0.0
    assert success_probability_without_replacement(100, 10, 91) == 1.0


def test_exact_random_probability_matches_small_enumeration():
    # One minus C(8, 2) / C(10, 2).
    expected = 1.0 - 28.0 / 45.0
    assert success_probability_without_replacement(10, 2, 2) == pytest.approx(expected)


def test_exact_random_probability_is_monotone():
    values = [success_probability_without_replacement(100, 3, n) for n in range(30)]
    assert values == sorted(values)
