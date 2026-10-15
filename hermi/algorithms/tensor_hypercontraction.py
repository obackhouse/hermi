"""Tensor hypercontraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
        selection_values[block] = ao_values * np.sqrt(np.sqrt(weights))
    return dense_values, selection_values


@dataclass
class GridPruner(ABC):
    """Base class that prepares a grid and delegates selection to a subclass."""

    @abstractmethod
    def select_indices(self, selection_values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        pass

    def __call__(
        self,
        grid: MolecularGrid,
        *,
        ao_basis: Basis,
        spherical: bool = False,
        ao_values: array[float64] | None = None,
        block_size: int = BLOCK_SIZE,
    ) -> MolecularGrid:
        """Evaluate selection features and return the reduced grid."""
        if ao_values is None:
            ao_values = self._selection_values(
                grid,
                ao_basis,
                spherical=spherical,
                block_size=block_size,
            )
        return self._subset(grid, self.select_indices(ao_values, grid.points))

    def _selection_values(
        self,
        grid: MolecularGrid,
        ao_basis: Basis,
        *,
        spherical: bool,
        block_size: int,
    ) -> array[float64]:
        """Evaluate the normalized AO values."""
        num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
        values = np.empty((len(grid), num_aos), dtype=np.float64)
        for block in iter_slices(0, len(grid), block_size):
            ao_values = ao_basis.evaluate(grid.points[block], spherical=spherical)
            values[block] = ao_values * np.sqrt(np.sqrt(grid.weights[block, None]))
        return values

    @staticmethod
    def _subset(grid: MolecularGrid, indices: array[np.int64]) -> MolecularGrid:
        return MolecularGrid(points=grid.points[indices], weights=grid.weights[indices])


@dataclass
class BlockTensorDecomposition(GridPruner):
    """Prune each spatial tile independently until the grid stops shrinking.

    The implementation follows the Hilbert-sorted block pivoted-Cholesky grid
    selection in Zhang et al. (2025).

    References:
        Y. Zhang, X. Xiong, W. Wu, and P. Su, J. Chem. Phys. 163, 174109 (2025).
        https://doi.org/10.1063/5.0289370
    """

    block_size: int = BLOCK_SIZE
    cholesky_tolerance: float = 1e-10
    hilbert_bits: int = HILBERT_BITS

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        selected = _hilbert_argsort(points, self.hilbert_bits)

        while True:
            block_pivots = [
                _grid_pivots(values[selected[block]], tolerance=self.cholesky_tolerance) + block.start
                for block in iter_slices(0, selected.size, self.block_size)
            ]
            indices = np.sort(np.concatenate(block_pivots))
            if indices.size == selected.size:
                break
            selected = selected[indices]

        return selected


@dataclass
class GlobalCholesky(GridPruner):
    """Prune the full grid with matrix-free pivoted Cholesky.

    The pivoted-Cholesky grid selection follows the LS-THC grid optimization
    approach of Matthews (2020).

    References:
        D. A. Matthews, J. Chem. Theory Comput. 16, 1382–1385 (2020).
        https://doi.org/10.1021/acs.jctc.9b01205
    """

    cholesky_tolerance: float = 1e-10
    hilbert_bits: int = HILBERT_BITS

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        order = _hilbert_argsort(points, self.hilbert_bits)
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
            if pivot_residual <= self.cholesky_tolerance:
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


@dataclass
class RandomPhaseCholesky(GridPruner):
    """Prune with accelerated randomly pivoted Cholesky.

    References:
        E. N. Epperly, J. A. Tropp, and R. J. Webber, SIAM J. Matrix Anal. Appl.
        46, 2527–2557 (2025). https://doi.org/10.1137/24M1699048
    """

    block_size: int = 16
    random_seed: int = 0
    cholesky_tolerance: float = 1e-10
    hilbert_bits: int = HILBERT_BITS

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        order = _hilbert_argsort(points, self.hilbert_bits)
        ordered_values = np.ascontiguousarray(values[order])
        num_points = ordered_values.shape[0]
        residual = np.einsum("ij,ij->i", ordered_values, ordered_values) ** 2
        capacity = min(num_points, 64)
        factor = np.empty((num_points, capacity), dtype=np.float64)
        pivots = np.empty(num_points, dtype=np.int64)
        rank = 0
        rng = np.random.default_rng(self.random_seed)

        while rank < num_points and float(np.max(residual)) > self.cholesky_tolerance:
            proposal_diagonal = residual.copy()
            trace = float(proposal_diagonal.sum())
            if trace <= 0.0:
                break
            proposals = rng.choice(
                num_points,
                size=self.block_size,
                replace=True,
                p=proposal_diagonal / trace,
            )

            block_residual = (ordered_values @ ordered_values[proposals].T) ** 2
            if rank:
                block_residual -= factor[:, :rank] @ factor[proposals, :rank].T

            for column_index, pivot in enumerate(proposals):
                if rank == num_points or float(np.max(residual)) <= self.cholesky_tolerance:
                    break

                pivot_residual = float(residual[pivot])
                envelope = float(proposal_diagonal[pivot])
                if pivot_residual <= 0.0 or envelope <= 0.0:
                    continue
                acceptance = min(pivot_residual / envelope, 1.0)
                if rng.random() >= acceptance:
                    continue

                column = block_residual[:, column_index] / np.sqrt(pivot_residual)
                if rank == capacity:
                    capacity = min(num_points, 2 * capacity)
                    expanded_factor = np.empty((num_points, capacity), dtype=np.float64)
                    expanded_factor[:, :rank] = factor[:, :rank]
                    factor = expanded_factor

                factor[:, rank] = column
                residual -= column**2
                np.maximum(residual, 0.0, out=residual)
                residual[pivot] = 0.0
                pivots[rank] = pivot
                rank += 1
                block_residual -= np.outer(column, column[proposals])

        return order[np.sort(pivots[:rank])]


@dataclass
class HybridGridPruner(GridPruner):
    """Prune candidates with one strategy, then remove redundancy globally.

    ``candidate_pruner`` first reduces the input grid. ``global_pruner`` then applies
    pivoted Cholesky to that candidate set, so the final stage can remove redundant
    candidates but cannot restore points discarded by the first stage.
    """

    candidate_pruner: GridPruner = field(default_factory=lambda: BlockTensorDecomposition(cholesky_tolerance=5e-11))
    global_pruner: GlobalCholesky = field(default_factory=GlobalCholesky)

    def select_indices(self, selection_values: array[float64], points: array[float64]) -> array[np.int64]:
        candidates = self.candidate_pruner.select_indices(selection_values, points)
        cleanup = self.global_pruner.select_indices(selection_values[candidates], points[candidates])
        return candidates[cleanup]


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
    grid_pruner: GridPruner | None = None,
    block_size: int = BLOCK_SIZE,
    metric_tolerance: float = 1e-12,
    regularization: float = 1e-14,
) -> tuple[array[float64], array[float64]]:
    r"""Construct a THC factorisation.

    Args:
        ao_basis: Basis used for the collocation matrix.
        aux_basis: Auxiliary basis used for the Coulomb metric decomposition.
        grid: Candidate grid used for point selection and the THC collocation matrix.
        fit_grid: Integration grid used to fit the Coulomb kernel. Defaults to ``grid``.
        spherical: Use real spherical functions instead of Cartesian functions.
        grid_pruner: Strategy instance used to prune the candidate grid. Defaults to block pruning.
        block_size: Batch size used to evaluate grid points and assemble the fit.
        metric_tolerance: Relative eigenvalue cutoff for the auxiliary Coulomb metric.
        regularization: Diagonal shift for the contracted Coulomb fit (default ``1e-14``).

    Returns:
        The collocation matrix and Coulomb kernel matrix for the tensor hypercontraction factorisation.
    """
    fit_grid = grid if fit_grid is None else fit_grid
    if grid_pruner is None:
        grid_pruner = BlockTensorDecomposition(block_size=block_size)
    dense_values, selection_values = _grid_collocation_values(
        ao_basis, grid, spherical=spherical, block_size=block_size
    )
    pruned_grid = grid_pruner(
        grid,
        ao_basis=ao_basis,
        spherical=spherical,
        ao_values=selection_values,
        block_size=block_size,
    )
    sparse_points = pruned_grid.points
    collocation = ao_basis.evaluate(sparse_points, spherical=spherical) * np.sqrt(pruned_grid.weights[:, None])
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
