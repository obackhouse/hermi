"""Hermite Gaussian expansions."""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING, overload

import numpy as np

from hermi.util import (
    cartesian_indices,
    cartesian_normalization,
    cartesian_to_spherical,
)

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy import float64, int64

    from hermi._typing import array
    from hermi.basis import ShellPairView, ShellView


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


@overload
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: None = None,
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[array[tuple[int, int], int64], array[tuple[int], float64]]: ...


@overload
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: tuple[tuple[int, int, int], ...],
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[array[tuple[int, int, int], int64], array[tuple[int], float64]]: ...


@lru_cache(maxsize=None)
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: tuple[tuple[int, int, int], ...] | None = None,
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[
    array[tuple[int, int], int64] | array[tuple[int, int, int], int64],
    array[tuple[int], float64],
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


def expansion_pair_coefficients(
    pair: ShellPairView,
    axis: int,
    *,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}`.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion pair coefficients.
    """
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if max_order is None:
        max_order = angular_momentum_i + angular_momentum_j
    orders = np.arange(1, max_order + 1)

    # a b -> a b t
    half_reciprocal_exponent = np.expand_dims(0.5 / pair.exponent, axis=-1)

    # a b, a -> a b
    offset_a = pair.position[..., axis] - pair.shell_i.position[..., axis][:, None]

    # a b, b -> a b
    offset_b = pair.position[..., axis] - pair.shell_j.position[..., axis][None, :]

    def raise_coefficient(
        source: array[tuple[int, int, int], float64],
        target: array[tuple[int, int, int], float64],
        offset: array[tuple[int, int], float64],
    ) -> None:
        """Raise coefficients along one shell angular-momentum index.

        Args:
            source: Coefficients at the current shell angular momentum.
            target: Output coefficients at the raised angular momentum.
            offset: Center offset for the shell being raised.
        """
        # a b t, a b -> a b t
        target[...] = source * np.expand_dims(offset, axis=-1)
        target[..., 1:] += source[..., :-1] * half_reciprocal_exponent
        target[..., :-1] += orders * source[..., 1:]

    coefficients = np.zeros(
        (
            len(pair.shell_i),
            len(pair.shell_j),
            max_order + 1,
            angular_momentum_i + 1,
            angular_momentum_j + 1,
        ),
        dtype=np.float64,
    )

    # E_0^00
    coefficients[..., 0, 0, 0] = pair.attenuation[..., axis]

    # E_t^i0
    for i in range(angular_momentum_i):
        raise_coefficient(coefficients[..., :, i, 0], coefficients[..., :, i + 1, 0], offset_a)

    # E_t^ij
    for i, j in itertools.product(
        range(angular_momentum_i + 1),
        range(angular_momentum_j),
    ):
        raise_coefficient(coefficients[..., :, i, j], coefficients[..., :, i, j + 1], offset_b)

    return coefficients


def expansion_pair_coefficients_coordinate_product(
    pair: ShellPairView,
    axis: int,
    *,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
    origin: float = 0.0,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}` after multiplying by a coordinate.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.
        origin: Coordinate origin for the multiplication.

    Returns:
        Expansion pair coefficients after coordinate multiplication.
    """
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if max_order is None:
        max_order = angular_momentum_i + angular_momentum_j + 1
    coefficients = expansion_pair_coefficients(
        pair,
        axis,
        angular_momentum_i=angular_momentum_i,
        angular_momentum_j=angular_momentum_j + 1,
        max_order=max_order,
    )
    offset = np.expand_dims(pair.shell_j.position[..., axis], axis=(0, 2, 3, 4)) - origin
    return coefficients[..., 1:] + offset * coefficients[..., :-1]


def _expansion_pair_coefficients_derivative_side(
    pair: ShellPairView,
    axis: int,
    side: Literal["bra", "ket"],
    *,
    derivative_order: int = 1,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    """Differentiate one Gaussian."""
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if max_order is None:
        max_order = angular_momentum_i + angular_momentum_j + derivative_order
    if side == "bra":
        count = angular_momentum_i + 1
        exponent = pair.shell_i.exponent[:, None]
        extra_i, extra_j = derivative_order, 0
    else:
        count = angular_momentum_j + 1
        exponent = pair.shell_j.exponent[None, :]
        extra_i, extra_j = 0, derivative_order

    coefficients = expansion_pair_coefficients(
        pair,
        axis,
        angular_momentum_i=angular_momentum_i + extra_i,
        angular_momentum_j=angular_momentum_j + extra_j,
        max_order=max_order,
    )

    weights = np.zeros((*pair.exponent.shape, count, count + derivative_order), dtype=np.float64)
    indices = np.arange(count)
    weights[..., indices, indices] = 1.0
    for _ in range(derivative_order):
        updated = np.zeros_like(weights)
        powers = np.arange(1, weights.shape[-1], dtype=np.float64)
        updated[..., :-1] += weights[..., 1:] * powers
        updated[..., 1:] -= 2 * exponent[..., None, None] * weights[..., :-1]
        weights = updated

    if side == "bra":
        return np.expand_dims(weights, axis=2) @ coefficients
    return coefficients @ np.expand_dims(weights, axis=1).mT


def expansion_pair_coefficients_derivative_bra(
    pair: ShellPairView,
    axis: int,
    *,
    derivative_order: int = 1,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}` after differentiating the bra.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        derivative_order: Order of the derivative.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion pair coefficients after differentiating the bra.
    """
    return _expansion_pair_coefficients_derivative_side(
        pair,
        axis,
        "bra",
        derivative_order=derivative_order,
        angular_momentum_i=angular_momentum_i,
        angular_momentum_j=angular_momentum_j,
        max_order=max_order,
    )


def expansion_pair_coefficients_derivative_ket(
    pair: ShellPairView,
    axis: int,
    *,
    derivative_order: int = 1,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}` after differentiating the ket.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        derivative_order: Order of the derivative.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion pair coefficients after differentiating the ket.
    """
    return _expansion_pair_coefficients_derivative_side(
        pair,
        axis,
        "ket",
        derivative_order=derivative_order,
        angular_momentum_i=angular_momentum_i,
        angular_momentum_j=angular_momentum_j,
        max_order=max_order,
    )


def expansion_pair_coefficients_derivative(
    pair: ShellPairView,
    axis: int,
    *,
    derivative_order: int = 1,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}` after differentiating.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        derivative_order: Order of the derivative.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion pair coefficients after differentiating.
    """
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if max_order is None:
        max_order = angular_momentum_i + angular_momentum_j + derivative_order
    coefficients = expansion_pair_coefficients(
        pair,
        axis,
        angular_momentum_i=angular_momentum_i,
        angular_momentum_j=angular_momentum_j,
        max_order=max_order,
    )
    result = np.zeros_like(coefficients)
    if derivative_order <= max_order:
        phase = (-1) ** derivative_order
        result[..., derivative_order:] = phase * coefficients[..., : max_order + 1 - derivative_order]
    return result


def expansion_pair_matrix(
    pair: ShellPairView,
    *,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
    mode: Literal["base", "coordinate-product", "derivative-bra", "derivative-ket", "derivative"] = "base",
    mode_axis: int = 0,
    derivative_order: int = 1,
    origin: float = 0.0,
    spherical: bool = False,
) -> array[tuple[int, int, int], float64]:
    r"""Assemble a shell-pair Hermite expansion, optionally transformed along one axis.

    Args:
        pair: Shell pair view.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.
        mode: Cartesian Gaussian transformation to apply.
        mode_axis: Cartesian axis on which the transformation acts.
        derivative_order: Order of a derivative transformation.
        origin: Coordinate origin for coordinate multiplication.
        spherical: Return real spherical functions instead of Cartesian functions.

    Returns:
        Expansion pair matrix with shape ``(ab, ij, p)``.
    """
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if mode_axis not in (0, 1, 2):
        raise ValueError(f"invalid mode_axis: {mode_axis!r}")
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")

    modes = {
        "base": expansion_pair_coefficients,
        "coordinate-product": expansion_pair_coefficients_coordinate_product,
        "derivative-bra": expansion_pair_coefficients_derivative_bra,
        "derivative-ket": expansion_pair_coefficients_derivative_ket,
        "derivative": expansion_pair_coefficients_derivative,
    }
    if mode not in modes:
        raise ValueError(f"invalid expansion mode: {mode!r}")
    if max_order is None:
        extra_order = 1 if mode == "coordinate-product" else derivative_order if mode.startswith("derivative") else 0
        max_order = angular_momentum_i + angular_momentum_j + extra_order

    cartesian_i = cartesian_indices(angular_momentum_i)
    cartesian_j = cartesian_indices(angular_momentum_j)
    indices = np.asarray(hermite_indices(max_order), dtype=np.int64)

    # Compute the expansion coefficients along each Cartesian axis
    coefficients = np.ones((len(pair), len(indices), len(cartesian_i) * len(cartesian_j)), dtype=np.float64)
    for axis in range(3):
        options: dict[str, Any] = {
            "angular_momentum_i": angular_momentum_i,
            "angular_momentum_j": angular_momentum_j,
            "max_order": max_order,
        }
        if axis == mode_axis and mode != "base":
            if mode == "coordinate-product":
                options["origin"] = origin
            else:
                options["derivative_order"] = derivative_order
            table = modes[mode](pair, axis, **options)
        else:
            table = expansion_pair_coefficients(pair, axis, **options)

        table = np.take(table, indices[:, axis], axis=2)
        table = np.take(table, cartesian_i[:, axis], axis=3)
        table = np.take(table, cartesian_j[:, axis], axis=4)
        coefficients *= table.reshape(coefficients.shape)

    # ab t ij, ij -> ab t ij
    normalization = np.multiply.outer(
        cartesian_normalization(angular_momentum_i),
        cartesian_normalization(angular_momentum_j),
    ).ravel()
    coefficients *= normalization

    if spherical:
        # ab t ij, ij pq -> ab t pq
        transform = np.multiply.outer(
            cartesian_to_spherical(angular_momentum_i),
            cartesian_to_spherical(angular_momentum_j),
        )
        transform = transform.transpose(0, 2, 1, 3)
        transform = transform.reshape(len(cartesian_i) * len(cartesian_j), -1)
        coefficients = coefficients @ transform

    # ab t ij -> ab ij t
    coefficients = coefficients.mT

    return coefficients


def expansion_coefficients(
    shell: ShellView,
    *,
    angular_momentum: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion coefficients :math:`E_t^i`.

    Args:
        shell: Shell view.
        angular_momentum: Angular momentum of the shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion coefficients.
    """
    if angular_momentum is None:
        angular_momentum = shell.angular_momentum
    if max_order is None:
        max_order = shell.angular_momentum
    orders = np.arange(1, max_order + 1)

    # a -> a t
    half_reciprocal_exponent = np.expand_dims(0.5 / shell.exponent, axis=-1)

    def raise_coefficient(
        source: array[tuple[int, int, int], float64],
        target: array[tuple[int, int, int], float64],
    ) -> None:
        """Raise coefficients along the shell angular-momentum index.

        Args:
            source: Coefficients at the current shell angular momentum.
            target: Output coefficients at the raised angular momentum.
        """
        # a t, a -> a t
        target[..., 1:] += source[..., :-1] * half_reciprocal_exponent
        target[..., :-1] += orders * source[..., 1:]

    coefficients = np.zeros((len(shell), max_order + 1, angular_momentum + 1), dtype=np.float64)

    # E_0^0
    coefficients[..., 0, 0] = 1.0

    # E_t^i
    for i in range(shell.angular_momentum):
        raise_coefficient(coefficients[..., :, i], coefficients[..., :, i + 1])

    return coefficients


def expansion_matrix(
    shell: ShellView,
    *,
    angular_momentum: int | None = None,
    max_order: int | None = None,
    spherical: bool = False,
) -> array[tuple[int, int, int], float64]:
    r"""Assemble the single-shell to Hermite expansion matrix.

    Args:
        shell: Shell view.
        angular_momentum: Angular momentum of the shell.
        max_order: Maximum Hermite order to generate.
        spherical: Return real spherical functions instead of Cartesian functions.

    Returns:
        Expansion matrix.
    """
    if angular_momentum is None:
        angular_momentum = shell.angular_momentum
    if max_order is None:
        max_order = shell.angular_momentum
    cartesian = cartesian_indices(angular_momentum)
    indices = np.asarray(hermite_indices(max_order), dtype=np.int64)

    # Compute the expansion coefficients along each Cartesian axis
    coefficients = np.ones((len(shell), len(indices), len(cartesian)), dtype=np.float64)
    table_const = expansion_coefficients(shell)
    for axis in range(3):
        table = np.take(table_const, indices[:, axis], axis=1)
        table = np.take(table, cartesian[:, axis], axis=2)
        coefficients *= table.reshape(coefficients.shape)

    # a t i, i -> a t i
    coefficients *= cartesian_normalization(angular_momentum)

    if spherical:
        # a t i, i p -> a t p
        coefficients = coefficients @ cartesian_to_spherical(angular_momentum)

    # a t i -> a i t
    coefficients = coefficients.mT

    return coefficients


def hermite_integral_recursion_zero_order[S: tuple[int, ...]](
    auxiliary: array[S, float64],
    max_order: int,
    x: array[tuple[int, ...], float64],
    y: array[tuple[int, ...], float64],
    z: array[tuple[int, ...], float64],
    scale: array[tuple[int, ...], float64] | float = 1.0,
) -> array[S, float64]:
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


def hermite_integral_recursion[S: tuple[int, ...]](
    auxiliary: array[S, float64],
    max_order: int,
    x: array[tuple[int, ...], float64],
    y: array[tuple[int, ...], float64],
    z: array[tuple[int, ...], float64],
    scale: array[tuple[int, ...], float64] | float = 1.0,
) -> array[S, float64]:
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
        The fundamental value :math:`R_n^{000}` must be precomputed and stored in ``auxiliary[0, 0, 0]``.
    """
    if np.size(scale) == 1 and np.all(scale == 1.0):
        x_scaled, y_scaled, z_scaled = x, y, z
    else:
        x_scaled = x * scale
        y_scaled = y * scale
        z_scaled = z * scale
    for t, u, v in hermite_indices(max_order)[1:]:
        order = t + u + v
        valid = max_order - order + 1
        if v:
            auxiliary[t, u, v, :valid] = auxiliary[t, u, v - 1, 1 : valid + 1] * z_scaled
            if v > 1:
                auxiliary[t, u, v, :valid] += (v - 1) * scale * auxiliary[t, u, v - 2, 1 : valid + 1]
        elif u:
            auxiliary[t, u, v, :valid] = auxiliary[t, u - 1, v, 1 : valid + 1] * y_scaled
            if u > 1:
                auxiliary[t, u, v, :valid] += (u - 1) * scale * auxiliary[t, u - 2, v, 1 : valid + 1]
        else:
            auxiliary[t, u, v, :valid] = auxiliary[t - 1, u, v, 1 : valid + 1] * x_scaled
            if t > 1:
                auxiliary[t, u, v, :valid] += (t - 1) * scale * auxiliary[t - 2, u, v, 1 : valid + 1]
    return auxiliary
