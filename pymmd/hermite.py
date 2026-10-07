"""Hermite Gaussian expansions."""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from pymmd.util import cartesian_indices, cartesian_normalization, cartesian_to_spherical

if TYPE_CHECKING:
    from numpy import float64

    from pymmd._typing import array
    from pymmd.basis import ShellPairView


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


def expansion_coefficients(pair: ShellPairView, axis: int) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion coefficients :math:`E_t^{ij}`.

    Args:
        pair: Primitive shell-pair data.
        axis: Cartesian axis.

    Returns:
        Expansion coefficients.
    """
    max_order = pair.angular_momentum + 1
    orders = np.arange(1, max_order)

    # a b -> a b
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
            max_order,
            pair.shell_i.angular_momentum + 1,
            pair.shell_j.angular_momentum + 1,
        ),
        dtype=np.float64,
    )

    # E_0^00
    coefficients[..., 0, 0, 0] = pair.attenuation[..., axis]

    # E_t^i0
    for i in range(pair.shell_i.angular_momentum):
        raise_coefficient(coefficients[..., :, i, 0], coefficients[..., :, i + 1, 0], offset_a)

    # E_t^ij
    for i, j in itertools.product(
        range(pair.shell_i.angular_momentum + 1),
        range(pair.shell_j.angular_momentum),
    ):
        raise_coefficient(coefficients[..., :, i, j], coefficients[..., :, i, j + 1], offset_b)

    return coefficients


def expansion_matrix(pair: ShellPairView, spherical: bool = False) -> array[tuple[int, int, int], float64]:
    r"""Assemble the shell-pair to Hermite expansion matrix.

    Args:
        pair: Primitive shell-pair data.
        spherical: Return real spherical functions instead of Cartesian functions.

    Returns:
        Expansion matrix.
    """
    cartesian_i = cartesian_indices(pair.shell_i.angular_momentum)
    cartesian_j = cartesian_indices(pair.shell_j.angular_momentum)
    indices = np.asarray(hermite_indices(pair.angular_momentum), dtype=np.int64)

    # Compute the expansion coefficients along each Cartesian axis
    coefficients = np.ones((len(pair), len(indices), len(cartesian_i) * len(cartesian_j)), dtype=np.float64)
    for axis in range(3):
        table = expansion_coefficients(pair, axis)
        table = np.take(table, indices[:, axis], axis=2)
        table = np.take(table, cartesian_i[:, axis], axis=3)
        table = np.take(table, cartesian_j[:, axis], axis=4)
        coefficients *= table.reshape(coefficients.shape)

    # i, j -> (i j)
    normalization = np.multiply.outer(
        cartesian_normalization(pair.shell_i.angular_momentum),
        cartesian_normalization(pair.shell_j.angular_momentum),
    ).ravel()

    # (a b) t (i j), (i j) -> (a b) t (i j)
    coefficients *= normalization

    if spherical:
        # i p, j q -> (i j) (p q)
        transform = np.multiply.outer(
            cartesian_to_spherical(pair.shell_i.angular_momentum),
            cartesian_to_spherical(pair.shell_j.angular_momentum),
        )
        transform = transform.transpose(0, 2, 1, 3)
        transform = transform.reshape(len(cartesian_i) * len(cartesian_j), -1)

        # (a b) t (i j), (i j) (p q) -> (a b) t (p q)
        coefficients = coefficients @ transform

    # (a b) t (i j) -> (a b) (i j) t
    coefficients = coefficients.mT

    return coefficients
