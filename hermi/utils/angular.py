"""Cartesian and spherical basis functions."""

from __future__ import annotations

from functools import lru_cache
from math import factorial, prod
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.special import lpmv

if TYPE_CHECKING:
    from hermi._typing import array


@lru_cache(maxsize=None)
def cartesian_indices(angular_momentum: int) -> array[np.int64]:
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
def cartesian_to_spherical(angular_momentum: int) -> array[np.float64]:
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
def num_cartesian_functions(angular_momentum: array[np.int64]) -> array[np.int64]: ...


def num_cartesian_functions(
    angular_momentum: int | array[np.int64],
) -> int | array[np.int64]:
    """Return the number of Cartesian functions for a given angular momentum."""
    return (angular_momentum + 1) * (angular_momentum + 2) // 2


@overload
def num_spherical_functions(angular_momentum: int) -> int: ...


@overload
def num_spherical_functions(angular_momentum: array[np.int64]) -> array[np.int64]: ...


def num_spherical_functions(
    angular_momentum: int | array[np.int64],
) -> int | array[np.int64]:
    """Return the number of real spherical functions for an angular momentum."""
    return 2 * angular_momentum + 1


@lru_cache(maxsize=None)
def cartesian_normalization(angular_momentum: int) -> array[np.float64]:
    """Return the Cartesian part of primitive normalization factors."""
    factors = []
    for triple in cartesian_indices(angular_momentum):
        denominator = prod(prod(range(2 * int(power) - 1, 0, -2)) for power in triple)
        factors.append(1.0 / np.sqrt(denominator))
    return np.asarray(factors, dtype=np.float64)
