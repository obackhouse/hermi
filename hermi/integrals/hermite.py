"""Hermite Gaussian expansions."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from numpy import float64, int64

    from hermi._typing import array


@lru_cache(maxsize=None)
def hermite_indices(max_order: int) -> list[tuple[int, int, int]]:
    """Return Hermite triples in total-order-first lexicographic order."""
    return [
        (t, u, v)
        for order in range(max_order + 1)
        for t in range(order + 1)
        for u in range(order - t + 1)
        for v in (order - t - u,)
    ]


@lru_cache(maxsize=None)
def hermite_auxiliary_layout(max_order: int) -> tuple[array[int64], int]:
    """Return packed auxiliary offsets and the number of valid entries."""
    offsets = np.full((max_order + 1,) * 3, -1, dtype=np.int64)
    size = 0
    for t, u, v in hermite_indices(max_order):
        offsets[t, u, v] = size
        size += max_order - t - u - v + 1
    return offsets, size


@lru_cache(maxsize=None)
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: tuple[tuple[int, int, int], ...] | None = None,
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[
    array[int64],
    array[float64],
]:
    r"""Map pairs of Hermite indices to operator indices and return parity signs.

    Args:
        angular_momentum_i: Maximum Hermite order for the first expansion.
        angular_momentum_j: Maximum Hermite order for the second expansion.
        shifts: Hermite index shifts, or ``None`` for the unshifted mapping.
        sign_side: Expansion side whose Hermite indices determine the signs.

    Returns:
        A single index map for ``None``, or one map per shift, and parity signs.
    """
    single_shift = shifts is None
    if shifts is None:
        shifts = ((0, 0, 0),)
    if not shifts:
        raise ValueError("at least one Hermite index shift is required")
    if any(component < 0 for shift in shifts for component in shift):
        raise ValueError("Hermite index shifts must be nonnegative")
    if sign_side not in ("bra", "ket"):
        raise ValueError(f"invalid sign side: {sign_side!r}")

    max_order = angular_momentum_i + angular_momentum_j + max(map(sum, shifts))
    hermite_i = np.asarray(hermite_indices(angular_momentum_i), dtype=np.int64)
    hermite_j = np.asarray(hermite_indices(angular_momentum_j), dtype=np.int64)
    hermite_all = np.asarray(hermite_indices(max_order), dtype=np.int64)

    hermite_map = np.full((max_order + 1,) * 3, -1, dtype=np.int64)
    hermite_map[tuple(hermite_all.T)] = np.arange(len(hermite_all), dtype=np.int64)

    hermite_sums = (
        hermite_i[:, None, None, :] + hermite_j[None, :, None, :] + np.asarray(shifts, dtype=np.int64)[None, None, :, :]
    )
    indices = hermite_map[tuple(hermite_sums.transpose(3, 2, 0, 1))]
    if single_shift:
        indices = indices[0]

    signed_indices = hermite_i if sign_side == "bra" else hermite_j
    signs = np.where(np.sum(signed_indices, axis=-1) % 2 == 0, 1.0, -1.0)

    return indices, signs


def hermite_integral_recursion_zero_order(
    auxiliary: array[float64],
    max_order: int,
    x: array[float64],
    y: array[float64],
    z: array[float64],
    scale: array[float64] | float = 1.0,
) -> array[float64]:
    r"""Recursively compute zeroth-order Hermite integrals :math:`R^{tuv}` from the fundamental value :math:`R^{000}`.

    Args:
        auxiliary: Output array for the Hermite integrals. Modified in-place.
        max_order: Maximum Hermite order to compute.
        x: x-axis displacement from the center of the Hermite expansion.
        y: y-axis displacement from the center of the Hermite expansion.
        z: z-axis displacement from the center of the Hermite expansion.
        scale: Scale factor for the integrals.

    Returns:
        The updated auxiliary array containing the computed Hermite integrals.

    Note:
        The fundamental value :math:`R^{000}` must be precomputed and stored in ``auxiliary[0, 0, 0]``.
    """
    if np.size(scale) == 1 and np.all(scale == 1.0):
        x_scaled, y_scaled, z_scaled = x, y, z
    else:
        x_scaled = x * scale
        y_scaled = y * scale
        z_scaled = z * scale
    for t, u, v in hermite_indices(max_order)[1:]:
        if v:
            auxiliary[t, u, v] = auxiliary[t, u, v - 1] * z_scaled
            if v > 1:
                auxiliary[t, u, v] += (v - 1) * scale * auxiliary[t, u, v - 2]
        elif u:
            auxiliary[t, u, v] = auxiliary[t, u - 1, v] * y_scaled
            if u > 1:
                auxiliary[t, u, v] += (u - 1) * scale * auxiliary[t, u - 2, v]
        else:
            auxiliary[t, u, v] = auxiliary[t - 1, u, v] * x_scaled
            if t > 1:
                auxiliary[t, u, v] += (t - 1) * scale * auxiliary[t - 2, u, v]
    return auxiliary


def hermite_integral_recursion(
    auxiliary: array[float64],
    max_order: int,
    x: array[float64],
    y: array[float64],
    z: array[float64],
    scale: array[float64] | float = 1.0,
) -> array[float64]:
    r"""Recursively compute Hermite integrals :math:`R_n^{tuv}` from the fundamental value :math:`R_n^{000}`.

    Args:
        auxiliary: Output array for the Hermite integrals. Modified in-place.
        max_order: Maximum Hermite order to compute.
        x: x-axis displacement from the center of the Hermite expansion.
        y: y-axis displacement from the center of the Hermite expansion.
        z: z-axis displacement from the center of the Hermite expansion.
        scale: Scale factor for the integrals.

    Returns:
        The updated auxiliary array containing the computed Hermite integrals.

    Note:
        The fundamental values :math:`R_n^{000}` must be precomputed in the first ``max_order + 1`` entries.
    """
    if np.size(scale) == 1 and np.all(scale == 1.0):
        x_scaled, y_scaled, z_scaled = x, y, z
    else:
        x_scaled = x * scale
        y_scaled = y * scale
        z_scaled = z * scale
    offsets, _ = hermite_auxiliary_layout(max_order)
    for t, u, v in hermite_indices(max_order)[1:]:
        order = t + u + v
        valid = max_order - order + 1
        target = offsets[t, u, v]
        if v:
            parent = offsets[t, u, v - 1] + 1
            auxiliary[target : target + valid] = auxiliary[parent : parent + valid] * z_scaled
            if v > 1:
                parent = offsets[t, u, v - 2] + 1
                auxiliary[target : target + valid] += (v - 1) * scale * auxiliary[parent : parent + valid]
        elif u:
            parent = offsets[t, u - 1, v] + 1
            auxiliary[target : target + valid] = auxiliary[parent : parent + valid] * y_scaled
            if u > 1:
                parent = offsets[t, u - 2, v] + 1
                auxiliary[target : target + valid] += (u - 1) * scale * auxiliary[parent : parent + valid]
        else:
            parent = offsets[t - 1, u, v] + 1
            auxiliary[target : target + valid] = auxiliary[parent : parent + valid] * x_scaled
            if t > 1:
                parent = offsets[t - 2, u, v] + 1
                auxiliary[target : target + valid] += (t - 1) * scale * auxiliary[parent : parent + valid]
    return auxiliary
