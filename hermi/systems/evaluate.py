"""Basis function evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from hermi.utils.angular import cartesian_indices, cartesian_normalization, cartesian_to_spherical
from hermi.utils.linear_algebra import outer_powers
from hermi.utils.slices import iter_slices

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis, ShellView


BLOCK_SIZE = 4096


def _get_screening_radius(shell: ShellView, threshold: float, *, spherical: bool) -> float:
    """Return a radius outside which every function in a shell is below a threshold."""
    angular_momentum = shell.angular_momentum
    normalization = np.max(cartesian_normalization(angular_momentum))
    if spherical:
        transform = cartesian_to_spherical(angular_momentum)
        normalization *= np.max(np.sum(np.abs(transform), axis=0))
    peak_radius = np.sqrt(angular_momentum / (2 * shell.exponent))

    def bound(radius: float) -> float:
        value = radius**angular_momentum * np.exp(-shell.exponent * radius**2)
        return float(normalization * np.dot(np.abs(shell.coefficient), value))

    lower = float(np.max(peak_radius))
    if bound(lower) < threshold:
        return lower

    upper = max(2 * lower, 1.0)
    while bound(upper) >= threshold:
        upper *= 2

    radius = brentq(lambda radius: bound(radius) - threshold, lower, upper)
    radius = float(np.nextafter(radius, np.inf))
    if bound(radius) >= threshold:
        return upper

    return radius


def evaluate_basis(
    basis: Basis,
    points: array[float64],
    *,
    ao_slice: slice = slice(None),
    spherical: bool = False,
    block_size: int = BLOCK_SIZE,
    threshold: float = 1e-10,
) -> array[float64]:
    r"""Evaluate contracted Gaussian basis functions at points in space.

    Args:
        basis: Basis set to evaluate.
        points: Cartesian coordinates of the points.
        ao_slice: AO slice in the original basis order.
        spherical: Evaluate real spherical instead of Cartesian functions.
        block_size: Number of points to evaluate in a single block.
        threshold: Set shell-block values to zero when their absolute bound is below this value.

    Returns:
        Basis function values on the points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    selected_aos = np.arange(*ao_slice.indices(num_aos), dtype=np.int64)
    blocks = tuple(iter_slices(0, points.shape[0], block_size))
    if threshold:
        block_minimum = np.asarray([np.min(points[block], axis=0) for block in blocks]).reshape((-1, 3))
        block_maximum = np.asarray([np.max(points[block], axis=0) for block in blocks]).reshape((-1, 3))

    # Loop over shells
    values = np.zeros((points.shape[0], selected_aos.size))
    for shell in basis.iter_shells():
        shell_ao_slice = shell.spherical_ao_slice if spherical else shell.cartesian_ao_slice
        selected = (selected_aos >= shell_ao_slice.start) & (selected_aos < shell_ao_slice.stop)
        shell_indices = selected_aos[selected] - shell_ao_slice.start
        if not np.any(selected):
            continue

        # Get Cartesian indices and normalization factors for the shell's angular momentum
        angular_momentum = shell.angular_momentum
        exponent = np.expand_dims(shell.exponent, axis=(0, 2))
        cartesian = cartesian_indices(angular_momentum)
        normalization = np.expand_dims(cartesian_normalization(angular_momentum), axis=(0, 1))
        screening_radius = _get_screening_radius(shell, threshold, spherical=spherical) if threshold else 0.0

        # Screen the blocks
        if screening_radius:
            minimum = block_minimum - shell.position[0]
            maximum = block_maximum - shell.position[0]
            closest = np.maximum(np.maximum(minimum, -maximum), 0)
            screened_blocks = np.einsum("bi,bi->b", closest, closest, optimize=True) > screening_radius**2
        else:
            screened_blocks = np.zeros(len(blocks), dtype=np.bool_)

        # Loop over points in blocks to avoid memory issues
        for block_index, block in enumerate(blocks):
            if screened_blocks[block_index]:
                continue
            displacement = points[block, None, :] - shell.position[None, :, :]
            radius_squared = np.einsum("...x,...x->...", displacement, displacement, optimize=True)
            radius_squared = np.expand_dims(radius_squared, axis=2)

            polynomial = np.ones((*displacement.shape[:2], cartesian.shape[0]))
            for axis in range(3):
                polynomial *= outer_powers(displacement[..., axis], angular_momentum)[..., cartesian[:, axis]]

            primitive_values = np.exp(-exponent * radius_squared) * polynomial * normalization
            block_values = np.einsum("a,gan->gn", shell.coefficient, primitive_values, optimize=True)
            if spherical:
                block_values = block_values @ cartesian_to_spherical(angular_momentum)
            values[block, selected] = block_values[:, shell_indices]

    return values


del BLOCK_SIZE
