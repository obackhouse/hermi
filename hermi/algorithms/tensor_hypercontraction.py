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

AO_BLOCK_SIZE = 2048


def _quantize_points(points: array[float64], scale: int) -> array[np.int64]:
    """Map three-dimensional points to integer coordinates on a regular cube."""
    points = points - np.min(points, axis=0)
    extent = np.ptp(points, axis=0)
    scaled = np.divide(points, extent, out=np.zeros_like(points), where=extent != 0)
    return np.floor(scaled * scale).astype(np.int64)


@dataclass
class GridSorter(ABC):
    """Return a permutation of grid points that defines their processing order."""

    @abstractmethod
    def sort_indices(self, points: array[float64]) -> array[np.int64]:
        """Return point indices in the desired order."""
        pass


@dataclass
class HilbertSort(GridSorter):
    """Order points along a Hilbert space-filling curve."""

    bits: int = 16

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 21:
            raise ValueError("bits must be between 1 and 21 for signed 64-bit Morton keys")

    def sort_indices(self, points: array[float64]) -> array[np.int64]:
        """Return point indices in the desired order."""
        coordinates = _quantize_points(points, 2**self.bits - 1)

        # Convert coordinates to Hilbert transpose form.
        coordinates = coordinates.copy()
        mask = 1 << (self.bits - 1)
        for _ in range(self.bits):
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
        mask = 1 << (self.bits - 1)
        for _ in range(self.bits):
            selected = (coordinates[:, 2] & mask) != 0
            correction[selected] ^= mask - 1
            mask >>= 1
        coordinates ^= correction[:, None]

        # Interleave the Hilbert transpose bits into scalar curve indices
        distance = np.zeros(points.shape[0], dtype=np.int64)
        for bit in range(self.bits - 1, -1, -1):
            for axis in range(3):
                distance = (distance << 1) | ((coordinates[:, axis] >> bit) & 1)

        # Use the distances to sort indices
        indices = np.argsort(distance, kind="stable")

        return indices


@dataclass
class MortonSort(GridSorter):
    """Order points by interleaving coordinate bits (Morton or Z order)."""

    bits: int = 16

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 21:
            raise ValueError("bits must be between 1 and 21 for signed 64-bit Morton keys")

    def sort_indices(self, points: array[float64]) -> array[np.int64]:
        """Return point indices in the desired order."""
        coordinates = _quantize_points(points, 2**self.bits - 1)
        keys = np.zeros(points.shape[0], dtype=np.int64)
        for bit in range(self.bits - 1, -1, -1):
            for axis in range(3):
                keys = (keys << 1) | ((coordinates[:, axis] >> bit) & 1)
        return np.argsort(keys, kind="stable")


@dataclass
class InputOrder(GridSorter):
    """Keep the grid's original order, useful as a baseline."""

    def sort_indices(self, points: array[float64]) -> array[np.int64]:
        """Return point indices in the desired order."""
        return np.arange(points.shape[0], dtype=np.int64)


@dataclass
class RecursiveSpatialBisection(GridSorter):
    """Order points by recursive median cuts along each region's widest axis."""

    leaf_size: int = 2048

    def sort_indices(self, points: array[float64]) -> array[np.int64]:
        """Return point indices in the desired order."""
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be positive")

        leaves: list[array[np.int64]] = []
        pending = [np.arange(points.shape[0], dtype=np.int64)]
        while pending:
            indices = pending.pop()
            if indices.size <= self.leaf_size:
                extent = np.ptp(points[indices], axis=0)
                axis = int(np.argmax(extent))
                leaves.append(indices[np.lexsort((indices, points[indices, axis]))])
                continue

            extent = np.ptp(points[indices], axis=0)
            axis = int(np.argmax(extent))
            ordered = indices[np.lexsort((indices, points[indices, axis]))]
            midpoint = ordered.size // 2
            pending.append(ordered[midpoint:])
            pending.append(ordered[:midpoint])

        return np.concatenate(leaves) if leaves else np.empty(0, dtype=np.int64)


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
) -> tuple[array[float64], array[float64]]:
    """Evaluate weighted AO values for THC fitting and grid selection."""
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    dense_values = np.empty((len(grid), num_aos), dtype=np.float64)
    selection_values = np.empty_like(dense_values)
    for block in iter_slices(0, len(grid), AO_BLOCK_SIZE):
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
    ) -> MolecularGrid:
        """Evaluate selection features and return the reduced grid."""
        if ao_values is None:
            ao_values = self._selection_values(
                grid,
                ao_basis,
                spherical=spherical,
            )
        return self._subset(grid, self.select_indices(ao_values, grid.points))

    def _selection_values(
        self,
        grid: MolecularGrid,
        ao_basis: Basis,
        *,
        spherical: bool,
    ) -> array[float64]:
        """Evaluate the normalized AO values."""
        num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
        values = np.empty((len(grid), num_aos), dtype=np.float64)
        for block in iter_slices(0, len(grid), AO_BLOCK_SIZE):
            ao_values = ao_basis.evaluate(grid.points[block], spherical=spherical)
            values[block] = ao_values * np.sqrt(np.sqrt(grid.weights[block, None]))
        return values

    @staticmethod
    def _subset(grid: MolecularGrid, indices: array[np.int64]) -> MolecularGrid:
        return MolecularGrid(points=grid.points[indices], weights=grid.weights[indices])


@dataclass
class BlockTensorDecomposition(GridPruner):
    """Prune each spatial tile independently until the grid stops shrinking.

    Points are sorted before being split into blocks. Hilbert ordering is the default;
    ``sorter`` can provide another ordering. The block pivoted-Cholesky procedure follows
    Zhang et al. (2025).

    References:
        Y. Zhang, X. Xiong, W. Wu, and P. Su, J. Chem. Phys. 163, 174109 (2025).
        https://doi.org/10.1063/5.0289370
    """

    block_size: int = 2048
    cholesky_tolerance: float = 1e-10
    sorter: GridSorter = field(default_factory=HilbertSort)

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        selected = self.sorter.sort_indices(points)

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

    The sorter sets the internal point order; Hilbert ordering is the default. It can
    affect locality and tie-breaking, but does not change the global residual criterion.
    The pivoted-Cholesky grid selection follows Matthews (2020).

    References:
        D. A. Matthews, J. Chem. Theory Comput. 16, 1382–1385 (2020).
        https://doi.org/10.1021/acs.jctc.9b01205
    """

    cholesky_tolerance: float = 1e-10
    sorter: GridSorter = field(default_factory=HilbertSort)

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        order = self.sorter.sort_indices(points)
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

    The sorter sets the internal point order and the mapping from a fixed random seed
    to physical points. Hilbert ordering is the default.

    References:
        E. N. Epperly, J. A. Tropp, and R. J. Webber, SIAM J. Matrix Anal. Appl.
        46, 2527–2557 (2025). https://doi.org/10.1137/24M1699048
    """

    block_size: int = 16
    random_seed: int = 0
    cholesky_tolerance: float = 1e-10
    sorter: GridSorter = field(default_factory=HilbertSort)

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        order = self.sorter.sort_indices(points)
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
        """Return the indices selected by this pruning strategy."""
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
    metric_tolerance: float,
    regularization: float,
    fit_values: array[float64] | None = None,
) -> array[float64]:
    """Fit the THC Coulomb kernel after grid selection has completed."""
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    if fit_values is None:
        fit_values = np.empty((len(fit_grid), num_aos), dtype=np.float64)
        for block in iter_slices(0, len(fit_grid), AO_BLOCK_SIZE):
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

    for block in iter_slices(0, len(fit_grid), AO_BLOCK_SIZE):
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
    grid_pruner: GridPruner | None = None,
    spherical: bool = False,
    metric_tolerance: float = 1e-12,
    regularization: float = 1e-14,
) -> tuple[array[float64], array[float64]]:
    r"""Construct a THC factorisation.

    Args:
        ao_basis: Basis set for the atomic orbitals.
        aux_basis: Basis set for the auxiliary Coulomb fit.
        grid: Candidate grid.
        fit_grid: Integration grid. If ``None``, the candidate grid is used for the fit.
        grid_pruner: Strategy instance used to prune the candidate grid. Defaults to block pruning.
        spherical: Use real spherical functions instead of Cartesian functions.
        metric_tolerance: Relative eigenvalue cutoff for the auxiliary Coulomb metric.
        regularization: Diagonal shift for the contracted Coulomb fit.

    Returns:
        The collocation matrix and Coulomb kernel matrix for the tensor hypercontraction factorisation.
    """
    fit_grid = grid if fit_grid is None else fit_grid
    if grid_pruner is None:
        grid_pruner = BlockTensorDecomposition()
    dense_values, selection_values = _grid_collocation_values(ao_basis, grid, spherical=spherical)
    pruned_grid = grid_pruner(
        grid,
        ao_basis=ao_basis,
        spherical=spherical,
        ao_values=selection_values,
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
        metric_tolerance=metric_tolerance,
        regularization=regularization,
        fit_values=dense_values if fit_grid is grid else None,
    )

    return collocation, kernel
