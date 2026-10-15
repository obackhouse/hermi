"""Tests for molecular quadrature grids."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

from hermi.systems.grid import (
    MolecularGrid,
    _angular_grid,
    _becke_partition_weights,
    _radial_grid,
    _radial_integrals,
    _radial_overlap,
    _simultaneous_diagonalize,
)
from hermi.systems.molecule import Molecule


def test_angular_grid_has_unit_directions_and_spherical_weights() -> None:
    directions, weights = _angular_grid(5)

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert weights.sum() == pytest.approx(4 * np.pi)


def test_radial_grid_maps_to_positive_radii() -> None:
    radii, weights = _radial_grid(8, 1.2)

    assert np.all(radii > 0)
    assert np.all(np.diff(radii) > 0)
    assert weights.shape == radii.shape
    assert np.all(weights > 0)


def test_becke_partition_weights_sum_to_one() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [1.2, 0.0, 0.0]])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    points = np.array([[0.2, 0.1, 0.3], [-0.4, 0.2, 1.1]])

    partitions = np.array([_becke_partition_weights(points, i, positions, distances, 3) for i in range(len(positions))])

    np.testing.assert_allclose(partitions.sum(axis=0), 1.0)
    assert np.all(partitions >= 0)


def test_becke_partition_matches_pairwise_reference() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [1.2, 0.0, 0.0]])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    points = np.array([[0.2, 0.1, 0.3], [-0.4, 0.2, 1.1]])
    point_distances = np.linalg.norm(points[:, None] - positions[None, :], axis=-1)
    log_partition = np.zeros((len(positions), len(points)))
    for first in range(len(positions)):
        for second in range(len(positions)):
            if first == second:
                continue
            mu = (point_distances[:, first] - point_distances[:, second]) / distances[first, second]
            switching = np.clip(mu, -1, 1)
            for _ in range(3):
                switching = 0.5 * switching * (3 - switching**2)
            log_partition[first] += np.log(np.clip(0.5 * (1 - switching), np.finfo(float).tiny, 1))

    for atom_index in range(len(positions)):
        expected = np.exp(log_partition[atom_index] - np.logaddexp.reduce(log_partition, axis=0))
        actual = _becke_partition_weights(points, atom_index, positions, distances, 3)
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_radial_gaussian_integrals() -> None:
    exponents_i = np.array([0.4, 1.1])
    coefficients_i = np.array([0.7, -0.2])
    exponents_j = np.array([0.6])
    coefficients_j = np.array([1.3])
    exponent_sum = exponents_i[:, None] + exponents_j[None, :]
    coefficient_product = coefficients_i[:, None] * coefficients_j[None, :]

    expected_overlap = np.sum(coefficient_product * 0.5 * gamma(1.5) / exponent_sum**1.5)
    expected_position = np.sum(coefficient_product * 0.5 / exponent_sum**2)
    actual_overlap, actual_position = _radial_integrals(exponents_i, coefficients_i, exponents_j, coefficients_j)
    assert actual_overlap == pytest.approx(expected_overlap)
    assert actual_position == pytest.approx(expected_position)
    assert _radial_overlap(exponents_i, coefficients_i, exponents_j, coefficients_j) == pytest.approx(expected_overlap)


def test_simultaneous_diagonalize_recovers_shared_basis() -> None:
    rng = np.random.default_rng(5)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    diagonal_values = rng.normal(size=(4, 5))
    matrices = np.stack([orthogonal @ np.diag(values) @ orthogonal.T for values in diagonal_values])

    diagonalized, rotation = _simultaneous_diagonalize(matrices, tolerance=1e-12, max_sweeps=100)
    off_diagonal = diagonalized.copy()
    indices = np.arange(matrices.shape[1])
    off_diagonal[:, indices, indices] = 0

    assert matrices.shape[1] > 1
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(
        diagonalized,
        np.einsum("ia,kij,jb->kab", rotation, matrices, rotation),
        atol=1e-12,
    )
    assert np.linalg.norm(off_diagonal) / np.linalg.norm(matrices) < 1e-12


def test_simultaneous_diagonalize_reduces_noncommuting_residual() -> None:
    rng = np.random.default_rng(23)
    matrices = rng.normal(size=(3, 6, 6))
    matrices = 0.5 * (matrices + matrices.transpose(0, 2, 1))

    diagonalized, rotation = _simultaneous_diagonalize(matrices, tolerance=1e-12, max_sweeps=100)
    off_diagonal = diagonalized.copy()
    initial_off_diagonal = matrices.copy()
    indices = np.arange(matrices.shape[1])
    off_diagonal[:, indices, indices] = 0
    initial_off_diagonal[:, indices, indices] = 0

    assert matrices.shape[1] > 1
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(6), atol=1e-12)
    assert np.linalg.norm(off_diagonal) < np.linalg.norm(initial_off_diagonal)


def test_from_becke_builds_expected_product_grid() -> None:
    molecule = Molecule.from_list(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
    radial_order = 4
    angular_order = 3
    grid = MolecularGrid.from_becke(molecule, radial_order=radial_order, angular_order=angular_order)
    expected_points_per_atom = radial_order * angular_order * (2 * angular_order + 1)

    assert len(grid) == 2 * expected_points_per_atom
    assert grid.points.shape == (len(grid), 3)
    assert grid.weights.shape == (len(grid),)
    assert np.all(np.isfinite(grid.points))
    assert np.all(np.isfinite(grid.weights))


def test_radial_and_finite_basis_dvr_grid_builders(basis_factory) -> None:
    positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    molecule = Molecule.from_list(["H", "H"], positions, units="bohr")
    basis = basis_factory(positions, [1.0, 0.5], [0, 0])
    parent_grid = MolecularGrid.from_becke(molecule, radial_order=8, angular_order=3)

    radial_grid = MolecularGrid.from_r_dvr(molecule, basis, angular_order=3)
    finite_dvr_grid = MolecularGrid.from_f_dvr(
        basis,
        parent_grid,
        include_candidate_grid=False,
        max_sweeps=2,
    )
    fitted_grid = MolecularGrid.from_f_dvr(basis, parent_grid, max_sweeps=2).reweight_nnls(basis)

    assert len(radial_grid) > 0
    assert len(finite_dvr_grid) == basis.num_contracted_spherical_aos
    assert len(fitted_grid) > 0
    assert np.all(radial_grid.weights >= 0)
    assert np.all(finite_dvr_grid.weights > 0)
    assert np.all(fitted_grid.weights > 0)


def test_nnls_reweight_can_be_chained_after_grid_construction(basis_factory) -> None:
    molecule = Molecule.from_list(["H"], [[0.0, 0.0, 0.0]], units="bohr")
    basis = basis_factory([[0.0, 0.0, 0.0]], [1.0], [0])
    options = {"angular_order": 3}
    grids = (
        MolecularGrid.from_becke(molecule, radial_order=8, **options).reweight_nnls(basis),
        MolecularGrid.from_handy(molecule, {"H": np.array([0.2, 0.6, 1.2])}, **options).reweight_nnls(basis),
        MolecularGrid.from_r_dvr(molecule, basis, **options).reweight_nnls(basis),
    )

    for grid in grids:
        assert len(grid) > 0
        assert np.all(grid.weights > 0)


def test_from_becke_builds_a_single_atom_grid() -> None:
    molecule = Molecule.from_list(["H"], [[0.0, 0.0, 0.0]], units="bohr")
    grid = MolecularGrid.from_becke(molecule, radial_order=4, angular_order=2)

    assert len(grid) == 4 * 2 * 5
    assert np.all(grid.weights > 0)
