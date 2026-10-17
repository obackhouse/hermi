"""Tensor hypercontraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

from hermi.operators.coulomb import Coulomb
from hermi.operators.point_charge import PointCharge
from hermi.systems.grid import MolecularGrid
from hermi.utils.linear_algebra import (
    greedy_pivoted_cholesky_pivots,
    pivoted_cholesky_pivots,
)
from hermi.utils.slices import iter_slices

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis

AO_BLOCK_SIZE = 4096


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
    ``sorter`` can provide another ordering. Each block uses LAPACK's complete
    pivoted Cholesky by default, with ``cholesky_tolerance`` as its maximum
    residual-diagonal cutoff. ``pivot_function`` can select another pivot routine that
    accepts the block's pair-overlap operator and the tolerance. Additional arguments
    for that routine can be passed with ``pivot_function_kwargs``.

    References:
        Y. Zhang, X. Xiong, W. Wu, and P. Su, J. Chem. Phys. 163, 174109 (2025).
        https://doi.org/10.1063/5.0289370
    """

    block_size: int = 2048
    cholesky_tolerance: float = 1e-10
    sorter: GridSorter = field(default_factory=HilbertSort)
    pivot_function: Callable[..., array[np.int64]] = partial(pivoted_cholesky_pivots)
    pivot_function_kwargs: dict[str, object] = field(default_factory=dict)

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        selected = self.sorter.sort_indices(points)

        while True:
            block_pivots = []
            for block in iter_slices(0, selected.size, self.block_size):
                block_values = values[selected[block]]
                pair_overlap = _GridPairOverlapOperator(block_values)
                pivots = self.pivot_function(
                    pair_overlap,
                    tolerance=self.cholesky_tolerance,
                    **self.pivot_function_kwargs,
                )
                block_pivots.append(pivots + block.start)
            indices = np.sort(np.concatenate(block_pivots))
            if indices.size == selected.size:
                break
            selected = selected[indices]

        return selected


class _GridPairOverlapOperator(LinearOperator):
    """Matrix-free AO-pair Gram operator with efficient column access."""

    def __init__(self, values: array[float64]) -> None:
        self.values = values
        norms = np.einsum("ij,ij->i", values, values)
        self.diagonal = norms**2
        super().__init__(dtype=np.dtype(np.float64), shape=(len(values), len(values)))  # type: ignore[call-arg]

    def _matvec(self, vector: array[float64]) -> array[float64]:
        weighted_pair_metric = self.values.T @ (self.values * vector[:, None])
        result = np.einsum("pi,ij,pj->p", self.values, weighted_pair_metric, self.values, optimize=True)
        return result  # type: ignore[no-any-return]

    def column(self, index: int) -> array[float64]:
        overlap = self.values @ self.values[index]
        return overlap**2  # type: ignore[no-any-return]

    def columns(self, indices: array[np.int64]) -> array[float64]:
        overlap = self.values @ self.values[indices].T
        return overlap**2

    def dense(self) -> array[float64]:
        overlap = self.values @ self.values.T
        return np.ascontiguousarray(overlap**2)


@dataclass
class GlobalDecomposition(GridPruner):
    """Prune the full grid with a configurable pivoted-Cholesky routine.

    The sorter sets the internal point order; Hilbert ordering is the default. It can
    affect locality and tie-breaking, but does not change the global residual criterion.
    The default matrix-free greedy pivot selection follows Matthews (2020). Set
    ``pivot_function`` and ``pivot_function_kwargs`` to use another pivot routine.

    References:
        D. A. Matthews, J. Chem. Theory Comput. 16, 1382–1385 (2020).
        https://doi.org/10.1021/acs.jctc.9b01205
    """

    cholesky_tolerance: float = 1e-10
    sorter: GridSorter = field(default_factory=HilbertSort)
    pivot_function: Callable[..., array[np.int64]] = partial(greedy_pivoted_cholesky_pivots)
    pivot_function_kwargs: dict[str, object] = field(default_factory=dict)

    def select_indices(self, values: array[float64], points: array[float64]) -> array[np.int64]:
        """Return the indices selected by this pruning strategy."""
        order = self.sorter.sort_indices(points)
        ordered_values = np.ascontiguousarray(values[order])
        pair_overlap = _GridPairOverlapOperator(ordered_values)
        pivots = self.pivot_function(
            pair_overlap,
            tolerance=self.cholesky_tolerance,
            **self.pivot_function_kwargs,
        )
        return order[pivots]


@dataclass
class HybridGridPruner(GridPruner):
    """Prune candidates with one strategy, then remove redundancy globally.

    ``candidate_pruner`` first reduces the input grid. ``global_pruner`` then applies
    pivoted Cholesky to that candidate set, so the final stage can remove redundant
    candidates but cannot restore points discarded by the first stage.
    """

    candidate_pruner: GridPruner = field(
        default_factory=lambda: BlockTensorDecomposition(
            cholesky_tolerance=5e-11,
            pivot_function=greedy_pivoted_cholesky_pivots,
        )
    )
    global_pruner: GlobalDecomposition = field(default_factory=GlobalDecomposition)

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

    # Compute the inverse square root of the auxiliary Coulomb metric
    coulomb = Coulomb()
    metric = coulomb.collect(aux_basis, (1, 1), spherical=spherical)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    retained = eigenvalues > metric_tolerance * np.max(eigenvalues)
    metric_inverse_sqrt = (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])).T

    # Compute the sparse pair-overlap matrix for the selected grid points
    overlap = collocation @ collocation.T
    sparse_gram = overlap**2
    sparse_gram += regularization * np.eye(sparse_gram.shape[0])

    # Accumulate the right-hand side of the linear system for the kernel matrix
    right_hand_side = np.zeros((sparse_points.shape[0], metric_inverse_sqrt.shape[0]), dtype=np.float64)
    for block in iter_slices(0, len(fit_grid), AO_BLOCK_SIZE):
        dense_overlap = fit_values[block] @ collocation.T
        pair_overlap = dense_overlap**2

        point_charge = PointCharge(fit_grid.points[block])
        point_potentials = point_charge.collect(aux_basis, spherical=spherical, batch_dim=True)
        right_hand_side += np.linalg.multi_dot((pair_overlap.T, point_potentials, metric_inverse_sqrt.T))

    # Solve the linear system for the kernel matrix
    gram_factor = cho_factor(
        sparse_gram,
        lower=True,
        overwrite_a=True,
        check_finite=False,
    )
    fitted_half_kernel = cho_solve(
        gram_factor,
        right_hand_side,
        overwrite_b=True,
        check_finite=False,
    )
    kernel = fitted_half_kernel @ fitted_half_kernel.T

    return kernel


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
