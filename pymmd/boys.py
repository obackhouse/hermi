"""Boys function."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from pymmd._boys_table import (
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

    from pymmd._typing import array

_BOYS_TABLE_FLAT = BOYS_TABLE.ravel()
_BOYS_TABLE_ORDERS = np.add.outer(np.arange(BOYS_MAX_ORDER + 1), np.arange(BOYS_INTERPOLATION_ORDER))
_CHEBYSHEV_ORDER = 3
_CHEBYSHEV_RESOLUTION = 128
_CHEBYSHEV_NODES = np.cos(np.pi * (2 * np.arange(_CHEBYSHEV_ORDER + 1) + 1) / (2 * (_CHEBYSHEV_ORDER + 1)))
_CHEBYSHEV_TRANSFORM = np.linalg.inv(np.polynomial.chebyshev.chebvander(_CHEBYSHEV_NODES, _CHEBYSHEV_ORDER))


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

    positive = argument > 0
    table_argument = np.where(positive, argument, 0.0)
    table_path = table_argument <= BOYS_LONG_RANGE_THRESHOLD

    highest_order = np.empty_like(table_argument, dtype=np.float64)
    if np.any(table_path):
        highest_order[table_path] = _interpolate(table_argument[table_path], max_order)

    long_range = ~table_path
    if np.any(long_range):
        highest_order[long_range] = _long_range_value(table_argument[long_range], max_order)

    highest_order = np.where(positive, highest_order, BOYS_INVERSE_ODD[max_order])
    exponential = np.exp(-table_argument)
    values = np.empty((*argument.shape, max_order + 1), dtype=np.float64)
    values[..., max_order] = highest_order

    for order in range(max_order - 1, -1, -1):
        values[..., order] = (2 * table_argument * values[..., order + 1] + exponential) * BOYS_INVERSE_ODD[order]

    return values


def boys_chebyshev(argument: array[tuple[int, ...], float64], max_order: int) -> array[tuple[int, ...], float64]:
    r"""Compute Boys function values through ``max_order`` using cubic interpolation."""
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if max_order > BOYS_MAX_ORDER:
        raise NotImplementedError(f"max_order must be less than or equal to {BOYS_MAX_ORDER}")

    positive = argument > 0
    table_argument = np.where(positive, argument, 0.0)
    table_path = table_argument <= BOYS_LONG_RANGE_THRESHOLD
    chebyshev_path = table_path & (table_argument < BOYS_LONG_RANGE_THRESHOLD)
    highest_order = np.empty_like(table_argument, dtype=np.float64)

    if np.any(chebyshev_path):
        highest_order[chebyshev_path] = _interpolate_chebyshev(table_argument[chebyshev_path], max_order)

    boundary = table_path & ~chebyshev_path
    if np.any(boundary):
        highest_order[boundary] = _interpolate(table_argument[boundary], max_order)

    long_range = ~table_path
    if np.any(long_range):
        highest_order[long_range] = _long_range_value(table_argument[long_range], max_order)

    highest_order = np.where(positive, highest_order, BOYS_INVERSE_ODD[max_order])
    exponential = np.exp(-table_argument)
    values = np.empty((*argument.shape, max_order + 1), dtype=np.float64)
    values[..., max_order] = highest_order

    for order in range(max_order - 1, -1, -1):
        values[..., order] = (2 * table_argument * values[..., order + 1] + exponential) * BOYS_INVERSE_ODD[order]

    return values


def _interpolate(argument: array[tuple[int, ...], float64], order: int) -> array[tuple[int, ...], float64]:
    scaled_argument = argument * BOYS_RESOLUTION
    grid_index = np.floor(scaled_argument + 0.5).astype(np.intp)
    offset = (grid_index - scaled_argument) / BOYS_RESOLUTION
    table_indices = grid_index[..., np.newaxis] * BOYS_TABLE.shape[1] + _BOYS_TABLE_ORDERS[order]
    derivatives = _BOYS_TABLE_FLAT[table_indices]

    value: array[tuple[int, ...], float64] = derivatives[..., -1] * BOYS_INVERSE_FACTORIALS[-1]
    for term in range(BOYS_INTERPOLATION_ORDER - 2, -1, -1):
        value *= offset
        value += derivatives[..., term] * BOYS_INVERSE_FACTORIALS[term]

    return value


def _interpolate_chebyshev(argument: array[tuple[int, ...], float64], order: int) -> array[tuple[int, ...], float64]:
    scaled_argument = argument * _CHEBYSHEV_RESOLUTION
    grid_index = np.floor(scaled_argument).astype(np.intp)
    offset = 2 * (scaled_argument - grid_index) - 1
    coefficients = _make_chebyshev_coefficients()[grid_index, order]

    b1 = np.zeros_like(offset)
    b2 = np.zeros_like(offset)
    for term in range(_CHEBYSHEV_ORDER, 0, -1):
        b0 = 2 * offset * b1 - b2 + coefficients[..., term]
        b2, b1 = b1, b0

    value: array[tuple[int, ...], float64] = offset * b1 - b2 + coefficients[..., 0]
    return value


@cache
def _make_chebyshev_coefficients() -> array[tuple[int, int, int], float64]:
    intervals = int(BOYS_LONG_RANGE_THRESHOLD * _CHEBYSHEV_RESOLUTION)
    points = (np.arange(intervals)[:, None] + (_CHEBYSHEV_NODES + 1) / 2) / _CHEBYSHEV_RESOLUTION
    values = boys(points.ravel(), BOYS_MAX_ORDER).reshape(intervals, _CHEBYSHEV_ORDER + 1, BOYS_MAX_ORDER + 1)
    coefficients: array[tuple[int, int, int], float64] = np.einsum(
        "kn,inm->imk", _CHEBYSHEV_TRANSFORM, values, optimize=True
    )
    return coefficients


def _long_range_value(argument: array[tuple[int, ...], float64], order: int) -> array[tuple[int, ...], float64]:
    value: array[tuple[int, ...], float64] = np.sqrt(np.pi) / (2 * np.sqrt(argument))
    for current_order in range(order):
        value *= (2 * current_order + 1) / (2 * argument)
    return value
