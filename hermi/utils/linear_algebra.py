"""Array operations and linear algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.linalg.lapack import dpstrf

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array


class PositiveSemidefiniteOperator(Protocol):
    """Matrix-free access needed by pivoted Cholesky routines."""

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix dimensions."""
        ...

    @property
    def diagonal(self) -> array[float64]:
        """Return the matrix diagonal."""
        ...

    def column(self, index: int) -> array[float64]:
        """Return one matrix column without forming the full matrix."""
        ...

    def columns(self, indices: array[np.int64]) -> array[float64]:
        """Return selected matrix columns without forming the full matrix."""
        ...

    def dense(self) -> array[float64]:
        """Return the explicitly formed matrix."""
        ...


def pivoted_cholesky_pivots(
    matrix: array[float64] | PositiveSemidefiniteOperator,
    *,
    tolerance: float,
) -> array[np.int64]:
    """Return DPSTRF pivots for a dense positive-semidefinite matrix."""
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.dense()
    _, pivots, rank, _ = dpstrf(matrix, tol=tolerance)  # type: ignore[arg-type]
    return (pivots[:rank] - 1).astype(np.int64)


def greedy_pivoted_cholesky_pivots(
    matrix: PositiveSemidefiniteOperator,
    *,
    tolerance: float,
) -> array[np.int64]:
    """Return maximum-residual pivots without forming the dense matrix."""
    num_points = matrix.shape[0]
    residual = matrix.diagonal.copy()
    capacity = min(num_points, 64)
    factor = np.empty((num_points, capacity), dtype=np.float64)
    pivots = np.empty(num_points, dtype=np.int64)
    rank = 0

    while rank < num_points:
        pivot = int(np.argmax(residual))
        pivot_residual = float(residual[pivot])
        if pivot_residual <= tolerance:
            break

        column = matrix.column(pivot)
        if rank:
            column -= factor[:, :rank] @ factor[pivot, :rank]
        column /= np.sqrt(pivot_residual)

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

    return np.sort(pivots[:rank])


def randomized_pivoted_cholesky_pivots(
    matrix: PositiveSemidefiniteOperator,
    *,
    tolerance: float,
    block_size: int,
    rng: np.random.Generator,
) -> array[np.int64]:
    """Return accelerated randomized pivots from a matrix-free PSD operator."""
    num_points = matrix.shape[0]
    residual = matrix.diagonal.copy()
    capacity = min(num_points, 64)
    factor = np.empty((num_points, capacity), dtype=np.float64)
    pivots = np.empty(num_points, dtype=np.int64)
    rank = 0

    while rank < num_points and float(np.max(residual)) > tolerance:
        proposal_diagonal = residual.copy()
        trace = float(proposal_diagonal.sum())
        if trace <= 0.0:
            break
        proposals = rng.choice(
            num_points,
            size=block_size,
            replace=True,
            p=proposal_diagonal / trace,
        )

        block_residual = matrix.columns(proposals)
        if rank:
            block_residual -= factor[:, :rank] @ factor[proposals, :rank].T

        for column_index, pivot in enumerate(proposals):
            if rank == num_points or float(np.max(residual)) <= tolerance:
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

    return np.sort(pivots[:rank])


def outer_powers(
    base: array[float64],
    max_order: int,
) -> array[float64]:
    """Compute consecutive integer powers of each base value, optionally scaled.

    Args:
        base: Values whose powers to compute.
        max_order: Largest nonnegative integer power.

    Returns:
        Consecutive powers from zero through ``max_order``.
    """
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    elif max_order == 0:
        return np.ones((*base.shape, 1), dtype=np.float64)
    elif max_order == 1:
        return np.stack((np.ones_like(base), base), axis=-1)
    powers = np.ones((*base.shape, max_order + 1), dtype=np.float64)
    powers[..., 1] = base
    for order in range(2, max_order + 1):
        powers[..., order] = powers[..., order - 1] * base
    return powers
