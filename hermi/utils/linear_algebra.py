"""Array operations and linear algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.linalg import solve_triangular
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
    """Return accelerated randomized pivots from a matrix-free PSD operator.

    Proposals are drawn in blocks and rejection-sampled using their small residual
    submatrix. Accepted columns are then added to the Cholesky factor together, so
    the updates over the full point set use matrix operations rather than a Python
    loop over individual columns.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")

    num_points = matrix.shape[0]
    if num_points == 0:
        return np.empty(0, dtype=np.int64)

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

        # Rejection sampling works on the small residual Gram matrix of the
        # proposals. Its Schur complement is updated only when a proposal passes
        proposal_residual = block_residual[proposals].copy()
        proposal_residual = (proposal_residual + proposal_residual.T) * 0.5
        proposal_factor = np.zeros((block_size, block_size), dtype=np.float64)
        proposal_envelopes = proposal_diagonal[proposals]
        accepted = np.empty(block_size, dtype=np.int64)
        num_accepted = 0

        for proposal_index in range(block_size):
            pivot_residual = float(proposal_residual[proposal_index, proposal_index])
            envelope = float(proposal_envelopes[proposal_index])
            if pivot_residual <= 0.0 or envelope <= 0.0:
                continue
            acceptance = min(pivot_residual / envelope, 1.0)
            if rng.random() >= acceptance:
                continue

            accepted[num_accepted] = proposal_index
            num_accepted += 1
            root = np.sqrt(pivot_residual)
            proposal_factor[proposal_index, proposal_index] = root
            if proposal_index + 1 < block_size:
                lower_column = proposal_residual[proposal_index + 1 :, proposal_index] / root
                proposal_factor[proposal_index + 1 :, proposal_index] = lower_column
                proposal_residual[proposal_index + 1 :, proposal_index + 1 :] -= np.outer(lower_column, lower_column)

        if num_accepted == 0:
            continue

        accepted = accepted[:num_accepted]
        lower = proposal_factor[np.ix_(accepted, accepted)]
        panel = solve_triangular(
            lower,
            block_residual[:, accepted].T,
            lower=True,
            check_finite=False,
        ).T

        # Preserve the stopping criterion even if it is reached inside a panel
        residual_path = np.cumsum(panel * panel, axis=1)
        residual_path *= -1.0
        residual_path += residual[:, None]
        np.maximum(residual_path, 0.0, out=residual_path)
        stop_positions = np.flatnonzero(residual_path.max(axis=0) <= tolerance)
        stopped = stop_positions.size > 0
        if stopped:
            panel_size = int(stop_positions[0]) + 1
            panel = panel[:, :panel_size]
            accepted = accepted[:panel_size]

        panel_rank = panel.shape[1]  # type: ignore[misc]
        if rank + panel_rank > capacity:
            capacity = min(num_points, max(rank + panel_rank, 2 * capacity))
            expanded_factor = np.empty((num_points, capacity), dtype=np.float64)
            expanded_factor[:, :rank] = factor[:, :rank]
            factor = expanded_factor

        factor[:, rank : rank + panel_rank] = panel
        new_pivots = proposals[accepted]
        pivots[rank : rank + panel_rank] = new_pivots
        residual -= np.einsum("ij,ij->i", panel, panel)
        np.maximum(residual, 0.0, out=residual)
        residual[new_pivots] = 0.0
        rank += panel_rank
        if stopped:
            break

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
