"""Boys function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erf

from hermi.integrals._boys_table import (
    BOYS_INTERPOLATION_ORDER,
    BOYS_INVERSE_FACTORIALS,
    BOYS_INVERSE_ODD,
    BOYS_LONG_RANGE_THRESHOLD,
    BOYS_MAX_ORDER,
    BOYS_RESOLUTION,
    BOYS_TABLE,
)

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array


def boys(argument: array[tuple[int, ...], float64], max_order: int) -> array[tuple[int, ...], float64]:
    r"""Compute Boys function values through ``max_order``.

    Args:
        argument: Boys function arguments.
        max_order: Maximum order of the Boys function.

    Returns:
        Boys function values.
    """
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if max_order > BOYS_MAX_ORDER:
        raise NotImplementedError(f"max_order must be less than or equal to {BOYS_MAX_ORDER}")

    # Zeroth-order: use the error function
    if max_order == 0:
        root = np.sqrt(argument)
        value = np.ones_like(root, dtype=np.float64)
        np.divide(0.5 * np.sqrt(np.pi) * erf(root), root, out=value, where=root != 0)
        return value[..., None]

    # Separate the argument into short- and long-range regions
    long_range = argument > BOYS_LONG_RANGE_THRESHOLD
    short_range = ~long_range

    # Compute the highest order Boys function value
    highest_order = np.empty_like(argument, dtype=np.float64)
    highest_order[short_range] = _interpolate(argument[short_range], max_order)
    highest_order[long_range] = _long_range(argument[long_range], max_order)

    # Fill the Boys function values using downward recursion
    values = np.empty((*argument.shape, max_order + 1), dtype=np.float64)
    values[..., max_order] = highest_order
    exponential = np.exp(-argument)
    for order in range(max_order - 1, -1, -1):
        values[..., order] = (2 * argument * values[..., order + 1] + exponential) * BOYS_INVERSE_ODD[order]

    return values


def _interpolate(argument: array[tuple[int, ...], float64], order: int) -> array[tuple[int, ...], float64]:
    """Interpolate the Boys function value using a precomputed table."""
    scaled_argument = argument * BOYS_RESOLUTION
    grid_index = (scaled_argument + 0.5).astype(np.intp)
    offset = (grid_index - scaled_argument) / BOYS_RESOLUTION

    value = BOYS_TABLE[grid_index, order + BOYS_INTERPOLATION_ORDER - 1] * BOYS_INVERSE_FACTORIALS[-1]
    for term in range(BOYS_INTERPOLATION_ORDER - 2, -1, -1):
        value *= offset
        value += BOYS_TABLE[grid_index, order + term] * BOYS_INVERSE_FACTORIALS[term]

    return value  # type: ignore[no-any-return]


def _long_range(argument: array[tuple[int, ...], float64], order: int) -> array[tuple[int, ...], float64]:
    """Compute the Boys function value using the long-range approximation."""
    value: array[tuple[int, ...], float64] = np.sqrt(np.pi) / (2 * np.sqrt(argument))
    for current_order in range(order):
        value *= (2 * current_order + 1) / (2 * argument)
    return value
