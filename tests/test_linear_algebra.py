"""Tests for matrix-free linear algebra routines."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.utils.linear_algebra import randomized_pivoted_cholesky_pivots


class DensePSDOperator:
    """Small dense PSD matrix exposed through the matrix-free protocol."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.shape = matrix.shape
        self.diagonal = np.diag(matrix).copy()

    def column(self, index: int) -> np.ndarray:
        return self.matrix[:, index].copy()

    def columns(self, indices: np.ndarray) -> np.ndarray:
        return self.matrix[:, indices].copy()

    def dense(self) -> np.ndarray:
        return self.matrix.copy()


@pytest.mark.parametrize("block_size", (1, 4, 16))
def test_randomized_pivoted_cholesky_reconstructs_low_rank_matrix(block_size: int) -> None:
    """Panel updates retain the low-rank approximation quality."""
    generator = np.random.default_rng(17)
    features = generator.normal(size=(32, 6))
    matrix = features @ features.T
    operator = DensePSDOperator(matrix)

    pivots = randomized_pivoted_cholesky_pivots(
        operator,
        tolerance=1e-12,
        block_size=block_size,
        rng=np.random.default_rng(23),
    )

    assert pivots.size == 6
    assert np.unique(pivots).size == pivots.size
    selected = matrix[np.ix_(pivots, pivots)]
    approximation = matrix[:, pivots] @ np.linalg.solve(selected, matrix[pivots, :])
    np.testing.assert_allclose(approximation, matrix, rtol=1e-10, atol=1e-10)


def test_randomized_pivoted_cholesky_stops_at_residual_tolerance() -> None:
    """A diagonal PSD matrix stops once its largest residual is below tolerance."""
    matrix = np.diag([4.0, 1.0, 0.09, 0.01])
    pivots = randomized_pivoted_cholesky_pivots(
        DensePSDOperator(matrix),
        tolerance=0.1,
        block_size=8,
        rng=np.random.default_rng(4),
    )

    assert pivots.size == 2
    np.testing.assert_allclose(np.max(np.diag(matrix)[np.setdiff1d(np.arange(4), pivots)]), 0.09)


def test_randomized_pivoted_cholesky_requires_positive_block_size() -> None:
    """Reject a block size that would otherwise prevent progress."""
    operator = DensePSDOperator(np.eye(2))
    with pytest.raises(ValueError, match="block_size must be positive"):
        randomized_pivoted_cholesky_pivots(
            operator,
            tolerance=1e-12,
            block_size=0,
            rng=np.random.default_rng(0),
        )
