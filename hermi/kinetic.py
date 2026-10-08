"""Kinetic integrals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hermi.hermite import expansion_matrix, hermite_indices
from hermi.integral import _reduce_primitive, _unpack_from_shell_pair_major, hermite_operator_matrix_indices
from hermi.symmetry import SymmetryN2S2

if TYPE_CHECKING:
    from typing import Literal

    from numpy import float64

    from hermi._typing import array
    from hermi.basis import Basis, ShellPairView


def kinetic_operator_hermite_11(
    pair_ab: ShellPairView,
    parity: Literal["bra", "ket"] = "bra",
) -> tuple[
    array[tuple[int, int, int, int], float64],
    array[tuple[int], float64],
]:
    r"""Compute the kinetic operator in the Hermite basis for :math:`(a|\hat{T}|b)` integrals.

    Args:
        pair_ab: Contracted shell pair.
        parity: Which side of the operator to apply the sign to.

    Returns:
        Hermite operator matrix and signs.
    """
    max_order = pair_ab.angular_momentum + 2
    if parity == "bra":
        displacement = -pair_ab.vector
    elif parity == "ket":
        displacement = pair_ab.vector
    else:
        raise ValueError(f"invalid parity: {parity!r}")
    mu = pair_ab.reduced_exponent
    argument = np.einsum("...,...x,...x->...", mu, displacement, displacement, optimize=True)
    scale = -2 * mu

    auxiliary = np.zeros((max_order + 1,) * 3 + mu.shape, dtype=np.float64)

    # G^000 fundamental
    auxiliary[0, 0, 0] = np.exp(-argument)

    for t, u, v in hermite_indices(max_order)[1:]:
        if v:
            auxiliary[t, u, v] = auxiliary[t, u, v - 1] * displacement[..., 2] * scale
            if v > 1:
                auxiliary[t, u, v] += (v - 1) * auxiliary[t, u, v - 2] * scale

        elif u:
            auxiliary[t, u, v] = auxiliary[t, u - 1, v] * displacement[..., 1] * scale
            if u > 1:
                auxiliary[t, u, v] += (u - 1) * auxiliary[t, u - 2, v] * scale

        else:
            auxiliary[t, u, v] = auxiliary[t - 1, u, v] * displacement[..., 0] * scale
            if t > 1:
                auxiliary[t, u, v] += (t - 1) * auxiliary[t - 2, u, v] * scale

    indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
    derivatives = np.moveaxis(auxiliary[*indices], 0, -1)
    angular_momentum_i = pair_ab.shell_i.angular_momentum
    angular_momentum_j = pair_ab.shell_j.angular_momentum
    shifted_indices, signs = hermite_operator_matrix_indices(
        angular_momentum_i,
        angular_momentum_j,
        shifts=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        sign_side=parity,
    )
    matrix = -0.5 * np.sum(np.take(derivatives, shifted_indices, axis=-1), axis=-3)

    return matrix, signs


def kinetic_integral_22(
    pair_ab: ShellPairView,
    expansion_a: array[tuple[int, int, int], float64] | None = None,
    expansion_b: array[tuple[int, int, int], float64] | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    r"""Compute a contracted :math:`(a|\hat{T}|b)` integral block.

    Args:
        pair_ab: Contracted shell pair.
        expansion_a: Expansion coefficients for the first shell.
        expansion_b: Expansion coefficients for the second shell.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical kinetic integral block.
    """
    if expansion_a is None:
        expansion_a = expansion_matrix(pair_ab.shell_i, spherical=spherical)
    if expansion_b is None:
        expansion_b = expansion_matrix(pair_ab.shell_j, spherical=spherical)
    operator, signs = kinetic_operator_hermite_11(pair_ab, parity="bra")

    # a i p, p -> a i p
    expansion_a = expansion_a * signs

    # a i p -> a 1 i p
    expansion_a = np.expand_dims(expansion_a, axis=1)

    # b j q -> 1 b j q
    expansion_b = np.expand_dims(expansion_b, axis=0)

    # a b p q, a b -> a b p q
    operator *= np.expand_dims(np.multiply.outer(pair_ab.shell_i.coefficient, pair_ab.shell_j.coefficient), axis=(2, 3))
    operator /= np.expand_dims(pair_ab.exponent, axis=(2, 3)) ** 1.5

    # a b p [q], 1 b j [q] -> a b p j
    contracted = operator @ expansion_b.mT

    # Reduce the b shell axis
    contracted = _reduce_primitive(contracted, pair_ab.shell_j, axis=1)

    # a 1 i [p], a b [p] j -> a b i j
    contracted = expansion_a @ contracted

    # Reduce the a shell axis
    contracted = _reduce_primitive(contracted, pair_ab.shell_i, axis=0)

    # a b i j -> ai bj
    contracted = _unpack_from_shell_pair_major(contracted, pair_ab.shell_i, pair_ab.shell_j, spherical=spherical)

    return (np.pi**1.5) * contracted  # type: ignore[no-any-return]


def kinetic_integrals_22(
    basis: Basis,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    r"""Compute all contracted :math:`(a|\hat{T}|b)` integrals.

    Args:
        basis: Basis set.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical kinetic integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    matrix = np.zeros((num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get the symmetry helper
    symmetry = SymmetryN2S2(matrix.shape)

    # Find shells, pairs and expansion coefficients
    shells = list(basis.iter_shell_groups())
    pairs = list(basis.iter_shell_pair_groups(unique=True))
    expansions = {shell: expansion_matrix(shell, spherical=spherical) for shell in shells}

    # Compute matrix blocks
    for pair_ab in pairs:
        i, j = pair_ab.spherical_ao_slices if spherical else pair_ab.cartesian_ao_slices
        block = kinetic_integral_22(
            pair_ab,
            expansion_a=expansions[pair_ab.shell_i],
            expansion_b=expansions[pair_ab.shell_j],
            spherical=spherical,
        )
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix to the original basis order
    matrix = matrix[np.ix_(indices, indices)]

    return matrix
