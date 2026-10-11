"""Tests for operator algebra."""

from __future__ import annotations

import numpy as np

from hermi.operators.kinetic import Kinetic
from hermi.operators.overlap import Overlap
from hermi.operators.operator import OperatorSum


def test_operator_sum_adds_matrices_with_shared_expansions(basis_factory) -> None:
    """Add Hermite operator matrices directly when their expansions are identical."""
    basis = basis_factory([[0, 0, 0], [1, 0, 0]], [1, 0.8], [0, 1])
    views = (basis[0:1], basis[1:2])
    overlap = Overlap()
    kinetic = Kinetic()
    operator = sum((overlap, kinetic, overlap))

    assert isinstance(operator, OperatorSum)
    assert len(operator.operators) == 3

    matrix = operator.get_operator_matrix(views)
    expected = 2 * overlap.get_operator_matrix(views) + kinetic.get_operator_matrix(views)
    assert matrix.shape == expected.shape
    np.testing.assert_allclose(matrix, expected)

    expansion = operator.get_expansion_matrices(views)
    assert expansion[0] is not None
    assert expansion[1] is not None
    assert expansion[0].shape[0] == 1
    assert expansion[1].shape[0] == 1

    actual = operator.contract(views)
    expected = 2 * overlap.contract(views) + kinetic.contract(views)
    np.testing.assert_allclose(actual, expected)


def test_operator_sum_stacks_matrices_with_different_expansions(basis_factory) -> None:
    """Stack Hermite operator matrices when their expansion coefficients differ."""
    basis = basis_factory([[0, 0, 0], [1, 0, 0]], [1, 0.8], [0, 0])
    views = (basis[0:1, 1:2], None)
    overlap = Overlap()
    kinetic = Kinetic()
    operator = overlap + kinetic

    matrix = operator.get_operator_matrix(views)
    overlap_matrix = overlap.get_operator_matrix(views)
    kinetic_matrix = kinetic.get_operator_matrix(views)
    assert matrix.shape[0] == 2
    np.testing.assert_allclose(matrix[0, ..., : overlap_matrix.shape[-2], : overlap_matrix.shape[-1]], overlap_matrix)
    np.testing.assert_allclose(matrix[1, ..., : kinetic_matrix.shape[-2], : kinetic_matrix.shape[-1]], kinetic_matrix)
    np.testing.assert_allclose(matrix[0, ..., overlap_matrix.shape[-2] :, :], 0.0)

    expansion = operator.get_expansion_matrices(views)
    assert expansion[0] is not None
    assert expansion[0].shape[0] == 2
    assert expansion[1] is None

    actual = operator.contract(views)
    expected = overlap.contract(views) + kinetic.contract(views)
    np.testing.assert_allclose(actual, expected)


def test_operator_scaling_preserves_operator_expansions(basis_factory) -> None:
    """Scale both the operator matrix and its contracted integral."""
    basis = basis_factory([[0, 0, 0], [1, 0, 0]], [1, 0.8], [0, 0])
    views = (basis[0:1, 1:2], None)
    kinetic = Kinetic()
    scale = 2.5

    for operator in (kinetic * scale, scale * kinetic):
        matrix = operator.get_operator_matrix(views)
        np.testing.assert_allclose(matrix, scale * kinetic.get_operator_matrix(views))
        np.testing.assert_allclose(operator.contract(views), scale * kinetic.contract(views))
