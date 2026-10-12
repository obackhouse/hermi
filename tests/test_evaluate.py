"""Tests for basis function evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.systems.evaluate import evaluate_basis


@pytest.mark.parametrize("spherical", (False, True))
def test_evaluate_basis_matches_pyscf(example_system, spherical: bool) -> None:
    """Compare Cartesian and spherical basis values with PySCF grid evaluation."""
    points = np.asarray(
        [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.1, 0.5], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    evaluator = "GTOval_sph" if spherical else "GTOval_cart"
    expected = example_system.reference.eval_gto(evaluator, points)
    suffix = "sph" if spherical else "cart"
    overlap = example_system.reference.intor(f"int1e_ovlp_{suffix}")
    expected *= (1 / np.sqrt(np.diag(overlap)))[None, :]

    actual = example_system.basis.evaluate(points, spherical=spherical)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("spherical", (False, True))
def test_evaluate_basis_ao_slice_matches_full_evaluation(example_system, spherical: bool) -> None:
    """Select strided AOs in the original basis order."""
    points = np.asarray([[0.2, -0.1, 0.4], [-0.2, 0.3, 0.1]], dtype=np.float64)
    full = evaluate_basis(example_system.basis, points, spherical=spherical, block_size=1)
    ao_slice = slice(1, None, 2)

    actual = example_system.basis.evaluate(points, ao_slice=ao_slice, spherical=spherical)

    np.testing.assert_allclose(actual, full[:, ao_slice], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("spherical", (False, True))
def test_evaluate_basis_screening_respects_threshold(basis_factory, spherical: bool) -> None:
    """Screened values are zero only when bounded by the requested threshold."""
    basis = basis_factory([[0, 0, 0]] * 3, [0.5, 0.5, 0.5], [0, 1, 2])
    points = np.asarray([[4.0, 0.0, 0.0]], dtype=np.float64)
    full = evaluate_basis(basis, points, spherical=spherical)
    threshold = 1e-2

    screened = evaluate_basis(basis, points, spherical=spherical, threshold=threshold)

    assert np.any(full != 0)
    assert np.any((screened == 0) & (full != 0))
    assert np.max(np.abs(full - screened)) <= threshold


def test_evaluate_basis_screening_handles_blocks_containing_shell_center(basis_factory) -> None:
    """Use the nearest distance to the whole point-block bounds when screening."""
    basis = basis_factory([[0, 0, 0]], [0.5], [0])
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    threshold = 0.3
    full = evaluate_basis(basis, points)

    screened = evaluate_basis(basis, points, block_size=2, threshold=threshold)

    assert full[0, 0] > threshold
    assert np.max(np.abs(full - screened)) <= threshold
