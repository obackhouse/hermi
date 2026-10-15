"""Tensor hypercontraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


def _select_grid_points_global(
    values: array[float64],
    order: array[np.int64],
    *,
    cholesky_tolerance: float,
) -> array[np.int64]:
    """Select grid points with global matrix-free pivoted Cholesky."""
    ordered_values = np.ascontiguousarray(values[order])
    num_points = ordered_values.shape[0]
    residual = np.einsum("ij,ij->i", ordered_values, ordered_values) ** 2
    capacity = min(num_points, 64)
    factor = np.empty((num_points, capacity), dtype=np.float64)
    pivots = np.empty(num_points, dtype=np.int64)
    rank = 0

    while rank < num_points:
        pivot = int(np.argmax(residual))
        pivot_residual = residual[pivot]
        if pivot_residual <= cholesky_tolerance:
            break

        column = (ordered_values @ ordered_values[pivot]) ** 2
        if rank:
            column -= factor[:, :rank] @ factor[pivot, :rank]
        new_factor = column / np.sqrt(pivot_residual)

        if rank == capacity:
            capacity = min(num_points, 2 * capacity)
            expanded_factor = np.empty((num_points, capacity), dtype=np.float64)
            expanded_factor[:, :rank] = factor[:, :rank]
            factor = expanded_factor

        factor[:, rank] = new_factor
        residual -= new_factor**2
        np.maximum(residual, 0.0, out=residual)
        residual[pivot] = 0.0
        pivots[rank] = pivot
        rank += 1

    return order[np.sort(pivots[:rank])]


def _select_grid_points_hybrid(
    values: array[float64],
    order: array[np.int64],
    *,
    block_size: int,
    cholesky_tolerance: float,
    candidate_oversampling: float,
) -> array[np.int64]:
    """Locally prune the grid, then globally prune the candidate points."""
    if not np.isfinite(candidate_oversampling) or candidate_oversampling < 1.0:
        raise ValueError("candidate_oversampling must be at least 1.0")

    candidate_tolerance = cholesky_tolerance / candidate_oversampling
    candidates = [
        _grid_pivots(values[order[block]], tolerance=candidate_tolerance) + block.start
        for block in iter_slices(0, order.size, block_size)
    ]
    candidate_positions = np.sort(np.concatenate(candidates))
    candidate_indices = order[candidate_positions]
    candidate_pivots = _select_grid_points_global(
        values[candidate_indices],
        np.arange(candidate_indices.size, dtype=np.int64),
        cholesky_tolerance=cholesky_tolerance,
    )
    return candidate_indices[candidate_pivots]


def _grid_collocation_values(
    ao_basis: Basis,
    grid: MolecularGrid,
    *,
    spherical: bool,
    block_size: int,
) -> tuple[array[float64], array[float64]]:
    """Evaluate weighted AO values for THC fitting and grid selection."""
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    dense_values = np.empty((len(grid), num_aos), dtype=np.float64)
    selection_values = np.empty_like(dense_values)
    for block in iter_slices(0, len(grid), block_size):
        ao_values = ao_basis.evaluate(grid.points[block], spherical=spherical)
        weights = grid.weights[block, None]
        dense_values[block] = ao_values * np.sqrt(weights)
        # Pair features carry one square root of the quadrature weight. Since
        # each pair feature is a product of two AO rows, its AO rows need w^1/4.
        selection_values[block] = ao_values * np.sqrt(np.sqrt(weights))
    return dense_values, selection_values


def _prune_grid_indices(
    selection_values: array[float64],
    points: array[float64],
    *,
    grid_selection: Literal["block", "global", "hybrid"],
    block_size: int,
    selection_block_size: int,
    candidate_oversampling: float,
    hilbert_bits: int,
    cholesky_tolerance: float,
) -> array[np.int64]:
    """Select THC grid indices from precomputed AO features."""
    order = _hilbert_argsort(points, hilbert_bits)
    if grid_selection == "block":
        return _select_grid_points(
            selection_values,
            order,
            block_size=selection_block_size,
            cholesky_tolerance=cholesky_tolerance,
        )
    if grid_selection == "global":
        return _select_grid_points_global(
            selection_values,
            order,
            cholesky_tolerance=cholesky_tolerance,
        )
    if grid_selection == "hybrid":
        return _select_grid_points_hybrid(
            selection_values,
            order,
            block_size=selection_block_size,
            cholesky_tolerance=cholesky_tolerance,
            candidate_oversampling=candidate_oversampling,
        )
    raise ValueError(f"grid_selection must be 'block', 'global', or 'hybrid', got {grid_selection!r}")


def _fit_coulomb_kernel(
    ao_basis: Basis,
    aux_basis: Basis,
    sparse_points: array[float64],
    collocation: array[float64],
    fit_grid: MolecularGrid,
    *,
    spherical: bool,
    block_size: int,
    metric_tolerance: float,
    regularization: float,
    fit_values: array[float64] | None = None,
) -> array[float64]:
    """Fit the THC Coulomb kernel after grid selection has completed."""
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    if fit_values is None:
        fit_values = np.empty((len(fit_grid), num_aos), dtype=np.float64)
        for block in iter_slices(0, len(fit_grid), block_size):
            ao_values = ao_basis.evaluate(fit_grid.points[block], spherical=spherical)
            fit_values[block] = ao_values * np.sqrt(fit_grid.weights[block, None])

    coulomb = Coulomb()
    metric = coulomb.collect(aux_basis, (1, 1), spherical=spherical)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    retained = eigenvalues > metric_tolerance * np.max(eigenvalues)
    metric_inverse_sqrt = (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])).T

    overlap = collocation @ collocation.T
    right_hand_side = np.zeros((sparse_points.shape[0], metric_inverse_sqrt.shape[0]), dtype=np.float64)
    sparse_gram = overlap**2
    sparse_gram += regularization * np.eye(sparse_gram.shape[0])

    for block in iter_slices(0, len(fit_grid), block_size):
        dense_overlap = fit_values[block] @ collocation.T
        pair_overlap = dense_overlap**2

        point_charge = PointCharge(fit_grid.points[block])
        point_potentials = point_charge.collect(aux_basis, spherical=spherical, batch_dim=True)
        half_kernel = point_potentials @ metric_inverse_sqrt.T
        right_hand_side += pair_overlap.T @ half_kernel

    fitted_half_kernel = np.linalg.solve(sparse_gram, right_hand_side)
    return fitted_half_kernel @ fitted_half_kernel.T


def tensor_hypercontraction(
    ao_basis: Basis,
    aux_basis: Basis,
    grid: MolecularGrid,
    *,
    fit_grid: MolecularGrid | None = None,
    spherical: bool = False,
    grid_selection: Literal["block", "global", "hybrid"] = "block",
    block_size: int = BLOCK_SIZE,
    selection_block_size: int | None = None,
    candidate_oversampling: float = 2.0,
    hilbert_bits: int = HILBERT_BITS,
    cholesky_tolerance: float = 1e-10,
    metric_tolerance: float = 1e-12,
    regularization: float = 1e-14,
) -> tuple[array[float64], array[float64]]:
    r"""Construct a THC factorisation using the block tensor decomposition method.

    Args:
        ao_basis: Basis used for the collocation matrix.
        aux_basis: Auxiliary basis used for the Coulomb metric decomposition.
        grid: Candidate grid used for point selection and the THC collocation matrix.
        fit_grid: Integration grid used to fit the Coulomb kernel. Defaults to ``grid``.
        spherical: Use real spherical functions instead of Cartesian functions.
        grid_selection: Use blockwise, global, or hybrid pivoted Cholesky for grid pruning.
        block_size: Batch size used to evaluate grid points and assemble the fit.
        selection_block_size: Tile size for grid selection; defaults to ``block_size``.
        candidate_oversampling: Tighten local pruning by this factor before global cleanup.
        hilbert_bits: Number of bits used to sort grid points along the Hilbert curve.
        cholesky_tolerance: Absolute pivoted Cholesky cutoff for grid selection.
        metric_tolerance: Relative eigenvalue cutoff for the auxiliary Coulomb metric.
        regularization: Diagonal shift for the contracted Coulomb fit (default ``1e-14``).

    Returns:
        The collocation matrix and Coulomb kernel matrix for the tensor hypercontraction factorisation.
    """
    fit_grid = grid if fit_grid is None else fit_grid
    selection_block_size = block_size if selection_block_size is None else selection_block_size
    dense_values, selection_values = _grid_collocation_values(
        ao_basis, grid, spherical=spherical, block_size=block_size
    )
    selected = _prune_grid_indices(
        selection_values,
        grid.points,
        grid_selection=grid_selection,
        block_size=block_size,
        selection_block_size=selection_block_size,
        candidate_oversampling=candidate_oversampling,
        hilbert_bits=hilbert_bits,
        cholesky_tolerance=cholesky_tolerance,
    )
    sparse_points = grid.points[selected]
    collocation = dense_values[selected]
    kernel = _fit_coulomb_kernel(
        ao_basis,
        aux_basis,
        sparse_points,
        collocation,
        fit_grid,
        spherical=spherical,
        block_size=block_size,
        metric_tolerance=metric_tolerance,
        regularization=regularization,
        fit_values=dense_values if fit_grid is grid else None,
    )

    return collocation, kernel


del BLOCK_SIZE, HILBERT_BITS
