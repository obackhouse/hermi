"""Tests for Hermite expansions and Boys functions."""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from hermi.integrals.boys import boys
from hermi.integrals.expansion import Expansion


def test_expansion_for_p_s_pair(basis_factory) -> None:
    """Check the Cartesian-to-Hermite expansion for a p-s pair."""
    basis = basis_factory([[0, 0, 0], [2, 0, 0]], [1, 2], [1, 0])
    pair = basis[0:1, 1:2]
    matrix = Expansion(pair).compute()
    attenuation = np.exp(-8 / 3)

    assert matrix.shape == (1, 3, 4)
    np.testing.assert_allclose(matrix[0, 0, 0], 4 / 3 * attenuation)
    np.testing.assert_allclose(matrix[0, 0, 3], attenuation / 6)


def test_expansion_derivatives_on_multiple_axes(basis_factory) -> None:
    """Include each axis shift when inferring the maximum Hermite order."""
    basis = basis_factory([[0, 0, 0]], [1], [0])
    shell = basis[0:1]
    expansion = Expansion(shell).differentiate(0).differentiate(1).compute()

    assert expansion.shape == (1, 1, 10)


def test_boys_values_at_zero_and_positive_argument() -> None:
    """Check the zero limits and erf-derived zeroth order."""
    arguments = np.asarray([0.0, 2.0])
    values = boys(arguments, max_order=1)

    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[0], [1.0, 1 / 3])
    expected_f0 = 0.5 * np.sqrt(np.pi / 2) * erf(np.sqrt(2))
    np.testing.assert_allclose(values[1, 0], expected_f0)
    np.testing.assert_allclose(values[1, 1], (expected_f0 - np.exp(-2)) / 4)
