"""Tensor hypercontraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg.lapack import dpstrf

from hermi.operators.coulomb import Coulomb
from hermi.operators.point_charge import PointCharge
from hermi.systems.grid import MolecularGrid
from hermi.utils.slices import iter_slices

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis

BLOCK_SIZE = 2048
HILBERT_BITS = 16


def _hilbert_argsort(points: array[float64], bits: int) -> array[np.int64]:
    """Return indices that sort three-dimensional points along a Hilbert curve."""
    points = points - np.min(points, axis=0)
    extent = np.ptp(points, axis=0)
    scaled = np.divide(points, extent, out=np.zeros_like(points), where=extent != 0)
    coordinates = np.floor(scaled * (2**bits - 1)).astype(np.int64)

    # Convert coordinates to Hilbert transpose form.
    coordinates = coordinates.copy()
    mask = 1 << (bits - 1)
    for _ in range(bits):
        lower = mask - 1
        for axis in (2, 1, 0):
            selected = (coordinates[:, axis] & mask) != 0
            coordinates[selected, 0] ^= lower
            exchange = (coordinates[~selected, 0] ^ coordinates[~selected, axis]) & lower
            coordinates[~selected, 0] ^= exchange
            coordinates[~selected, axis] ^= exchange
        mask >>= 1

    coordinates[:, 1] ^= coordinates[:, 0]
    coordinates[:, 2] ^= coordinates[:, 1]
    correction = np.zeros(points.shape[0], dtype=np.int64)
    mask = 1 << (bits - 1)
    for _ in range(bits):
        selected = (coordinates[:, 2] & mask) != 0
        correction[selected] ^= mask - 1
        mask >>= 1
    coordinates ^= correction[:, None]

    # Interleave the Hilbert transpose bits into scalar curve indices
    distance = np.zeros(points.shape[0], dtype=np.int64)
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            distance = (distance << 1) | ((coordinates[:, axis] >> bit) & 1)

    # Use the distances to sort indices
    indices = np.argsort(distance, kind="stable")

    return indices


def _grid_pivots(
    values: array[float64],
    *,
    tolerance: float,
) -> array[np.int64]:
    """Select representative grid points using pivoted Cholesky decomposition."""
    overlap = values @ values.T
    pair_overlap = np.ascontiguousarray(overlap**2)
    _, pivots, rank, _ = dpstrf(pair_overlap, tol=tolerance)
    return (pivots[:rank] - 1).astype(np.int64)


def _select_grid_points(
    values: array[float64],
    order: array[np.int64],
    *,
    block_size: int,
    cholesky_tolerance: float,
) -> array[np.int64]:
    """Return indices of representative points in an iteratively pruned grid."""
    selected = order

    while True:
        pivots = [
            _grid_pivots(
                values[selected[block]],
                tolerance=cholesky_tolerance,
            )
            + block.start
            for block in iter_slices(0, selected.size, block_size)
        ]
        indices = np.sort(np.concatenate(pivots))
        if indices.size == selected.size:
            break
        selected = selected[indices]

    return selected


def tensor_hypercontraction(
    ao_basis: Basis,
    aux_basis: Basis,
    grid: MolecularGrid,
    *,
    spherical: bool = False,
    block_size: int = BLOCK_SIZE,
    hilbert_bits: int = HILBERT_BITS,
    cholesky_tolerance: float = 1e-10,
    metric_tolerance: float = 1e-12,
    regularization: float = 1e-12,
) -> tuple[array[float64], array[float64]]:
    r"""Construct a THC factorisation using the block tensor decomposition method.

    Args:
        ao_basis: Basis used for the collocation matrix.
        aux_basis: Auxiliary basis used for the Coulomb metric decomposition.
        grid: Dense molecular grid used for the collocation matrix and Coulomb kernel fit.
        spherical: Use real spherical functions instead of Cartesian functions.
        block_size: Number of points in each pivoted Cholesky block.
        hilbert_bits: Number of bits used to sort grid points along the Hilbert curve.
        cholesky_tolerance: Absolute pivoted Cholesky cutoff for grid selection.
        metric_tolerance: Relative eigenvalue cutoff for the auxiliary Coulomb metric.
        regularization: Diagonal shift used when fitting the contracted Coulomb kernel.

    Returns:
        The collocation matrix and Coulomb kernel matrix for the tensor hypercontraction factorisation.
    """
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    dense_values = np.empty((len(grid), num_aos), dtype=np.float64)
    for block in iter_slices(0, len(grid), block_size):
        dense_values[block] = ao_basis.evaluate(grid.points[block], spherical=spherical)
        dense_values[block] *= np.expand_dims(np.sqrt(grid.weights[block]), axis=1)

    dense_order = _hilbert_argsort(grid.points, hilbert_bits)
    selected = _select_grid_points(
        dense_values,
        dense_order,
        block_size=block_size,
        cholesky_tolerance=cholesky_tolerance,
    )
    sparse_points = grid.points[selected]
    collocation = dense_values[selected]

    coulomb = Coulomb()
    metric = coulomb.collect(aux_basis, (1, 1), spherical=spherical)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    retained = eigenvalues > metric_tolerance * np.max(eigenvalues)
    metric_inverse_sqrt = (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])).T

    overlap = collocation @ collocation.T
    right_hand_side = np.zeros((sparse_points.shape[0], metric_inverse_sqrt.shape[0]), dtype=np.float64)
    sparse_gram = overlap**2
    sparse_gram += regularization * np.eye(sparse_gram.shape[0])

    for block in iter_slices(0, len(grid), block_size):
        dense_overlap = dense_values[block] @ collocation.T
        pair_overlap = dense_overlap**2

        point_charge = PointCharge(grid.points[block])
        point_potentials = point_charge.collect(aux_basis, spherical=spherical, batch_dim=True)
        half_kernel = point_potentials @ metric_inverse_sqrt.T
        right_hand_side += pair_overlap.T @ half_kernel

    fitted_half_kernel = np.linalg.solve(sparse_gram, right_hand_side)
    kernel = fitted_half_kernel @ fitted_half_kernel.T

    return collocation, kernel


del BLOCK_SIZE, HILBERT_BITS
