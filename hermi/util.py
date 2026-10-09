"""Utilities."""

from __future__ import annotations

from functools import lru_cache
from math import factorial, prod
from typing import TYPE_CHECKING, overload

import numpy as np
from numba import njit, prange
from scipy.special import lpmv

if TYPE_CHECKING:
    from typing import Never

    from numpy import float64, int64

    from hermi._typing import array


class DefaultHash:
    """Base class for objects with the default ``object.__hash__`` and ``object.__eq__``."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultHash):
            return NotImplemented
        return object.__eq__(self, other)

    def __hash__(self) -> int:
        return object.__hash__(self)


@njit(parallel=True, cache=True)
def _add_reduceat_3d(
    values: array[tuple[int, int, int], float64],
    indices: array[tuple[int], int64],
) -> array[tuple[int, int, int], float64]:
    """Reduce a 3D array along the middle axis at the given segment starts."""
    shape = (values.shape[0], indices.size, values.shape[2])
    result = np.empty(shape, dtype=values.dtype)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            start = indices[j]
            stop = values.shape[1] if j + 1 == shape[1] else indices[j + 1]
            for k in range(shape[2]):
                total = values[i, start, k]
                for l in range(start + 1, stop):
                    total += values[i, l, k]
                result[i, j, k] = total
    return result


def add_reduceat[S: tuple[int, ...]](
    values: array[S, float64],
    indices: array[tuple[int], int64],
    axis: int = 0,
) -> array[S, float64]:
    """Sum segments along an axis using a parallel Numba kernel.

    Args:
        values: Values to reduce.
        indices: Segment start indices.
        axis: Axis along which to reduce.

    Returns:
        Segment sums with the reduced axis replaced by the number of indices.
    """
    shape = values.shape
    values_3d = values.reshape(prod(shape[:axis]), shape[axis], prod(shape[axis + 1 :]))
    reduced = _add_reduceat_3d(values_3d, indices)
    reduced_shape = (*shape[:axis], indices.size, *shape[axis + 1 :])
    return reduced.reshape(reduced_shape)  # type: ignore[return-value]


def _add_reduceat_numpy[S: tuple[int, ...]](
    values: array[S, float64],
    indices: array[tuple[int], int64],
    axis: int = 0,
) -> array[S, float64]:
    """Reduce segments with NumPy."""
    return np.add.reduceat(values, indices, axis=axis)  # type: ignore[return-value]


def outer_powers(
    base: array[tuple[int, ...], float64],
    max_order: int,
) -> array[tuple[int, ...], float64]:
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


@lru_cache(maxsize=None)
def cartesian_to_spherical(angular_momentum: int) -> array[tuple[int, int], float64]:
    """Return the normalized Cartesian-to-real-spherical transformation."""
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be nonnegative")
    cartesian = cartesian_indices(angular_momentum)
    if angular_momentum < 2:
        return np.eye(len(cartesian), dtype=np.float64)

    azimuth = 2 * np.pi * np.arange(2 * angular_momentum + 1)
    azimuth /= azimuth.size

    z_points, z_weights = np.polynomial.legendre.leggauss(angular_momentum + 1)
    z_grid, azimuth_grid = np.meshgrid(z_points, azimuth, indexing="ij")
    radial = np.sqrt(1 - z_grid**2)

    points = np.stack((radial * np.cos(azimuth_grid), radial * np.sin(azimuth_grid), z_grid), axis=-1).reshape(-1, 3)
    weights = np.broadcast_to(z_weights[:, None] * (2 * np.pi / azimuth.size), z_grid.shape).ravel()

    cartesian_values = np.prod(points[:, None] ** cartesian[None, :], axis=-1)
    cartesian_values *= cartesian_normalization(angular_momentum)[None, :]  # type: ignore[arg-type]

    spherical_values = np.empty((points.shape[0], azimuth.size), dtype=np.float64)
    for magnetic in range(angular_momentum + 1):
        normalization = np.sqrt(
            factorial(angular_momentum - magnetic) / factorial(angular_momentum + magnetic) * azimuth.size / (4 * np.pi)
        )
        harmonic = (
            normalization
            * lpmv(magnetic, angular_momentum, points[:, 2])
            * np.exp(1j * magnetic * np.arctan2(points[:, 1], points[:, 0]))
        )
        if magnetic == 0:
            spherical_values[:, angular_momentum] = harmonic.real
        else:
            phase = (-1) ** magnetic * np.sqrt(2)
            spherical_values[:, angular_momentum + magnetic] = phase * harmonic.real
            spherical_values[:, angular_momentum - magnetic] = phase * harmonic.imag

    weighted_cartesian = cartesian_values * np.sqrt(weights[:, None])
    weighted_spherical = spherical_values * np.sqrt(weights[:, None])

    transform = np.linalg.lstsq(weighted_cartesian, weighted_spherical, rcond=None)[0]
    transform *= np.linalg.norm(weighted_cartesian, axis=0)[0]

    return transform


@overload
def num_cartesian_functions(angular_momentum: int) -> int: ...
@overload
def num_cartesian_functions[S: tuple[int, ...]](angular_momentum: array[S, int64]) -> array[S, int64]: ...


def num_cartesian_functions(
    angular_momentum: int | array[tuple[int, ...], int64],
) -> int | array[tuple[int, ...], int64]:
    """Return the number of Cartesian functions for a given angular momentum."""
    return (angular_momentum + 1) * (angular_momentum + 2) // 2


@overload
def num_spherical_functions(angular_momentum: int) -> int: ...
@overload
def num_spherical_functions[S: tuple[int, ...]](angular_momentum: array[S, int64]) -> array[S, int64]: ...


def num_spherical_functions(
    angular_momentum: int | array[tuple[int, ...], int64],
) -> int | array[tuple[int, ...], int64]:
    """Return the number of real spherical functions for an angular momentum."""
    return 2 * angular_momentum + 1


@lru_cache(maxsize=None)
def cartesian_normalization(angular_momentum: int) -> array[tuple[int], np.float64]:
    """Return the Cartesian part of primitive normalization factors."""
    factors = []
    for triple in cartesian_indices(angular_momentum):
        denominator = prod(prod(range(2 * int(power) - 1, 0, -2)) for power in triple)
        factors.append(1.0 / np.sqrt(denominator))
    return np.asarray(factors, dtype=np.float64)


def rebase_slice(source: slice, target: slice) -> slice:
    """Rebase a source slice to a target slice.

    Args:
        source: Slice to rebase.
        target: Slice to rebase onto.

    Returns:
        Rebasing of the source slice onto the target slice.
    """
    start = source.start - target.start if source.start is not None else None
    stop = source.stop - target.start if source.stop is not None else None
    return slice(start, stop, source.step)


def unreachable() -> Never:
    """Raise an error indicating that the code is unreachable."""
    raise AssertionError("unreachable code, this should never happen")
