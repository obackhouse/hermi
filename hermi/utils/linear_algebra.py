"""Array operations and linear algebra."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Callable

    from numpy import float64, int64

    from hermi._typing import array

    type NumbaAddReduceatKernel = Callable[[array[float64], array[int64]], array[float64]]

NUMBA_REDUCEAT_MIN_ELEMENTS = 16_384


def _add_reduceat_numpy(values: array[float64], indices: array[int64], axis: int = 0) -> array[float64]:
    """Sum slices of an array using NumPy's ``reduceat`` implementation."""
    return np.add.reduceat(values, indices, axis=axis)


@lru_cache(maxsize=1)
def _get_numba_add_reduceat() -> NumbaAddReduceatKernel | None:
    """Return the compiled reduction kernel when Numba is installed."""
    try:
        from numba import njit, prange
    except ImportError:
        return None

    def _add_reduceat_kernel(values: array[float64], indices: array[int64]) -> array[float64]:
        """Sum segments along the middle axis of an ``(N, M, K)`` array."""
        result = np.empty((values.shape[0], indices.size, values.shape[2]), dtype=values.dtype)
        for group in prange(indices.size):
            for outer in range(values.shape[0]):
                start = indices[group]
                stop = indices[group + 1] if group + 1 < indices.size else values.shape[1]
                for inner in range(values.shape[2]):
                    result[outer, group, inner] = values[outer, start, inner]
                for middle in range(start + 1, max(stop, start + 1)):
                    for inner in range(values.shape[2]):
                        result[outer, group, inner] += values[outer, middle, inner]
        return result

    return njit(cache=True, nogil=True, parallel=True)(_add_reduceat_kernel)  # type: ignore[no-any-return]


def _add_reduceat_numba(values: array[float64], indices: array[int64], axis: int = 0) -> array[float64]:
    """Sum slices after reshaping the array around the reduced axis."""
    reducer = _get_numba_add_reduceat()
    assert reducer is not None
    axis %= values.ndim
    shape = values.shape
    num_outer = int(np.prod(shape[:axis]))
    num_inner = int(np.prod(shape[axis + 1 :]))
    reshaped = values.reshape((num_outer, shape[axis], num_inner))
    result = reducer(reshaped, indices)
    return result.reshape((*shape[:axis], indices.size, *shape[axis + 1 :]))


def add_reduceat(values: array[float64], indices: array[int64], axis: int = 0) -> array[float64]:
    """Sum consecutive segments along an axis, using Numba for sufficiently large arrays.

    Args:
        values: Array to reduce.
        indices: Indices of the first element of each segment to sum.
        axis: Axis along which to reduce.

    Returns:
        Array of summed segments, with the same shape as ``values`` except for the reduced axis
        which has length ``len(indices)``.
    """
    if values.size < NUMBA_REDUCEAT_MIN_ELEMENTS or values.dtype != np.float64:
        return _add_reduceat_numpy(values, indices, axis)
    reducer = _get_numba_add_reduceat()
    if reducer is None:
        return _add_reduceat_numpy(values, indices, axis)
    return _add_reduceat_numba(values, indices, axis)


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
