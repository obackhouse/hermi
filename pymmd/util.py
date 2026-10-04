"""Utilities."""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from numpy import float64, int64

    from pymmd._typing import array


def outer_powers(
    base: array[tuple[int, ...], float64],
    max_order: int,
    multiplier: array[tuple[int, ...], float64] | None = None,
) -> array[tuple[int, ...], float64]:
    """Compute consecutive integer powers of each base value, optionally scaled.

    Args:
        base: Values whose powers to compute.
        max_order: Largest nonnegative integer power.
        multiplier: Optional factor to multiply into each power.

    Returns:
        Consecutive powers from zero through ``max_order``.
    """
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")

    powers = np.ones((*base.shape, max_order + 1), dtype=np.float64)
    if max_order:
        powers[..., 1:] = base[..., None]
    np.multiply.accumulate(powers, axis=-1, out=powers)
    if multiplier is not None:
        np.multiply(powers, multiplier, out=powers)
    return powers


@lru_cache(maxsize=None)
def cartesian_indices(angular_momentum: int) -> array[tuple[int, int], int64]:
    """Return Cartesian indices in lexicographic order."""
    return np.asarray(
        [
            (x, y, angular_momentum - x - y)
            for x in range(angular_momentum, -1, -1)
            for y in range(angular_momentum - x, -1, -1)
        ],
        dtype=np.int64,
    )


CARTESIAN_INDICES = list(map(cartesian_indices, range(5)))


@overload
def num_cartesian_indices(angular_momentum: int) -> int: ...
@overload
def num_cartesian_indices[S: tuple[int, ...]](angular_momentum: array[S, int64]) -> array[S, int64]: ...


def num_cartesian_indices(angular_momentum: int | array[tuple[int, ...], int64]) -> int | array[tuple[int, ...], int64]:
    """Return the number of Cartesian functions for a given angular momentum."""
    return (angular_momentum + 1) * (angular_momentum + 2) // 2


@lru_cache(maxsize=None)
def cartesian_normalization(angular_momentum: int) -> array[tuple[int], np.float64]:
    """Return the Cartesian part of primitive normalization factors."""
    factors = []
    for triple in cartesian_indices(angular_momentum):
        denominator = prod(prod(range(2 * int(power) - 1, 0, -2)) for power in triple)
        factors.append(1.0 / np.sqrt(denominator))
    return np.asarray(factors, dtype=np.float64)
