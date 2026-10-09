"""Hermite operator contraction."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np

from hermi.hermite import expansion_matrix, expansion_pair_matrix
from hermi.util import num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.basis import ShellPairView, ShellView
    from hermi.operator import ExpansionMatrix, OperatorMatrix


def _reduce_primitive[S: tuple[int, ...]](values: array[S, float64], shell: ShellView, axis: int) -> array[S, float64]:
    """Reduce a primitive shell axis via summation to contracted shells."""
    expanded = values.reshape(
        *values.shape[:axis],
        len(shell),
        *values.shape[axis + 1 :],
    )
    expanded = np.add.reduceat(expanded, shell.basis.contracted_shell_starts(shell.indices), axis=axis)
    expanded = expanded.reshape(
        *values.shape[:axis],
        shell.num_shells,
        *values.shape[axis + 1 :],
    )
    return expanded  # type: ignore[return-value]


def _reduce_primitive_pair[S: tuple[int, ...]](
    values: array[S, float64], pair: ShellPairView, axis: int
) -> array[S, float64]:
    """Reduce a primitive shell-pair axis via summation to contracted shells."""
    expanded = values.reshape(
        *values.shape[:axis],
        len(pair.shell_i),
        len(pair.shell_j),
        *values.shape[axis + 1 :],
    )
    starts_i = pair.basis.contracted_shell_starts(pair.shell_i.indices)
    starts_j = pair.basis.contracted_shell_starts(pair.shell_j.indices)
    lengths_i = np.diff(np.append(starts_i, len(pair.shell_i)))
    lengths_j = np.diff(np.append(starts_j, len(pair.shell_j)))
    if (
        lengths_i.size
        and lengths_j.size
        and np.all(lengths_i == lengths_i[0])
        and np.all(lengths_j == lengths_j[0])
    ):
        expanded = expanded.reshape(
            *expanded.shape[:axis],
            len(starts_i),
            lengths_i[0],
            len(starts_j),
            lengths_j[0],
            *expanded.shape[axis + 2 :],
        )
        expanded = np.sum(expanded, axis=(axis + 1, axis + 3))
    else:
        expanded = np.add.reduceat(expanded, starts_i, axis=axis)
        expanded = np.add.reduceat(expanded, starts_j, axis=axis + 1)
    expanded = expanded.reshape(
        *values.shape[:axis],
        pair.shell_i.num_shells * pair.shell_j.num_shells,
        *values.shape[axis + 1 :],
    )
    return expanded  # type: ignore[return-value]


def _unpack_from_shell_pair_major[S: tuple[int, ...]](
    values: array[S, float64], *shells: ShellView, spherical: bool = False
) -> array[S, float64]:
    """Unpack a shell-pair-major array to an interleaved array of angular momentum functions."""
    num_functions = num_spherical_functions if spherical else num_cartesian_functions
    num_ang = tuple(num_functions(shell.angular_momentum) for shell in shells)
    num_shells = tuple(shell.num_shells for shell in shells)
    transpose = sum(((i, i + len(shells)) for i in range(len(shells))), ())
    expanded = values.reshape(*num_shells, *num_ang)
    expanded = expanded.transpose(transpose)
    return expanded.reshape(tuple(map(prod, zip(num_shells, num_ang))))  # type: ignore[return-value]


def contract_operator_11(
    operator: OperatorMatrix,
    shell_a: ShellView,
    shell_b: ShellView,
    expansion_a: ExpansionMatrix | None = None,
    expansion_b: ExpansionMatrix | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    r"""Contract a Hermite operator with (1, 1) Hermite expansion matrices for :math:`(a|\hat{O}|b)`.

    Args:
        operator: Hermite operator matrix with shape ``(a, b, p, q)`` or ``(n, a, b, p, q)``.
        shell_a: First shell view.
        shell_b: Second shell view.
        expansion_a: Expansion matrix for the first shell with shape ``(a, p, i)`` or ``(n, a, p, i)``.
        expansion_b: Expansion matrix for the second shell with shape ``(b, q, j)`` or ``(n, b, q, j)``.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical :math:`(a|\hat{O}|b)` with shape ``(ai, bj)``.
    """
    if expansion_a is None:
        expansion_a = expansion_matrix(shell_a, spherical=spherical)
    if expansion_b is None:
        expansion_b = expansion_matrix(shell_b, spherical=spherical)
    coefficients = np.multiply.outer(shell_a.coefficient, shell_b.coefficient)

    # Align dimensions
    operator = np.expand_dims(operator, axis=0) if operator.ndim == 4 else operator  # a b p q -> n a b p q
    expansion_a = np.expand_dims(expansion_a, axis=1 if expansion_a.ndim == 4 else (0, 2))  # a p i -> n a 1 i p
    expansion_b = np.expand_dims(expansion_b, axis=0 if expansion_b.ndim == 4 else (0, 1))  # b q j -> n 1 b j q
    coefficients = np.expand_dims(coefficients, axis=(0, 3, 4))  # a b -> 1 a b 1 1

    # Apply the contraction coefficients
    operator = operator * coefficients  # n a b p q, 1 a b 1 1 -> n a b p q

    # Contract the operator with the right-hand expansion matrix
    contracted = operator @ expansion_b.mT  # n a b p q, n 1 b q j -> n a b p j

    # Reduce the b shell axis
    contracted = _reduce_primitive(contracted, shell_b, axis=2)  # n a b p j -> n a b p j

    # Contract the result with the left-hand expansion matrix
    contracted = expansion_a @ contracted  # n a 1 i p, n a b p j

    # Sum over batch dimension
    contracted = np.sum(contracted, axis=0)  # n a b i j -> a b i j

    # Reduce the a shell axis
    contracted = _reduce_primitive(contracted, shell_a, axis=0)  # a b i j -> a b i j

    # Unpack the contracted result into the final integral array
    contracted = _unpack_from_shell_pair_major(contracted, shell_a, shell_b, spherical=spherical)  # a b i j -> ai bj

    return contracted


def contract_operator_22(
    operator: OperatorMatrix,
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    expansion_ab: ExpansionMatrix | None = None,
    expansion_cd: ExpansionMatrix | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    r"""Contract a Hermite operator with (2, 2) Hermite expansion matrices for :math:`(ab|\hat{O}|cd)`.

    Args:
        operator: Hermite operator matrix with shape ``(ab, cd, p, q)`` or ``(n, ab, cd, p, q)``.
        pair_ab: First shell pair view.
        pair_cd: Second shell pair view.
        expansion_ab: Expansion matrix for the first shell pair with shape ``(ab, p, ij)`` or ``(n, ab, p, ij)``.
        expansion_cd: Expansion matrix for the second shell pair with shape ``(cd, q, kl)`` or ``(n, cd, q, kl)``.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical :math:`(ab|\hat{O}|cd)` integral block with shape ``(ai, bj, ck, dl)``.
    """
    if expansion_ab is None:
        expansion_ab = expansion_pair_matrix(pair_ab, spherical=spherical)
    if expansion_cd is None:
        expansion_cd = expansion_pair_matrix(pair_cd, spherical=spherical)
    coefficients = np.multiply.outer(pair_ab.coefficient.ravel(), pair_cd.coefficient.ravel())

    # Align dimensions
    expansion_ab = np.expand_dims(expansion_ab, axis=1 if expansion_ab.ndim == 4 else (0, 2))  # ab ij p -> n ab 1 ij p
    expansion_cd = np.expand_dims(expansion_cd, axis=0 if expansion_cd.ndim == 4 else (0, 1))  # cd kl q -> n 1 cd kl q
    coefficients = np.expand_dims(coefficients, axis=(0, 3, 4))  # ab cd -> 1 ab cd 1 1

    # Apply the contraction coefficients
    operator = operator * coefficients  # n ab cd p q, n ab cd 1 1 -> n ab cd p q

    # Contract the operator with the right-hand expansion matrix
    contracted = operator @ expansion_cd.mT  # n ab cd p q, n 1 cd kl q -> n ab cd p kl

    # Reduce the c and d shell axes
    contracted = _reduce_primitive_pair(contracted, pair_cd, axis=2)  # n ab cd p kl -> n ab cd p kl

    # Contract the result with the left-hand expansion matrix
    contracted = expansion_ab @ contracted  # n ab 1 ij p, n ab cd p kl -> n ab cd ij kl

    # Sum over batch dimension
    contracted = np.sum(contracted, axis=0)  # n ab cd ij kl -> ab cd ij kl

    # Reduce the a and b shell axes
    contracted = _reduce_primitive_pair(contracted, pair_ab, axis=0)  # ab cd ij kl -> ab cd ij kl

    # Unpack the contracted result into the final integral array
    contracted = _unpack_from_shell_pair_major(  # ab cd ij kl -> ai bj ck dl
        contracted,
        pair_ab.shell_i,
        pair_ab.shell_j,
        pair_cd.shell_i,
        pair_cd.shell_j,
        spherical=spherical,
    )

    return contracted


def contract_operator_20(
    operator: OperatorMatrix,
    pair_ab: ShellPairView,
    expansion_ab: ExpansionMatrix | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int], float64]:
    r"""Contract a Hermite operator with (2, 0) Hermite expansion matrices for :math:`(ab|\hat{O})`.

    Args:
        operator: Hermite operator matrix with shape ``(ab, 1, p, 1)`` or ``(n, ab, 1, p, 1)``.
        pair_ab: Shell pair view.
        expansion_ab: Expansion matrix for the first shell pair with shape ``(ab, p, ij)`` or ``(n, ab, p, ij)``.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical :math:`(ab|\hat{O})` integral block with shape ``(ai, bj)``.
    """
    if expansion_ab is None:
        expansion_ab = expansion_pair_matrix(pair_ab, spherical=spherical)
    coefficients = pair_ab.coefficient.ravel()

    # Align dimensions
    coefficients = np.expand_dims(coefficients, axis=(0, 2, 3, 4))  # ab -> 1 ab 1 1 1
    expansion_ab = np.expand_dims(expansion_ab, axis=1 if expansion_ab.ndim == 4 else (0, 2))  # ab ij p -> n ab 1 ij p

    # Apply the contraction coefficients
    operator = operator * coefficients  # n ab 1 p 1, n ab 1 1 1 -> n ab 1 p 1

    # Contract the operator with the expansion matrix
    contracted = expansion_ab @ operator  # n ab 1 ij p, n ab 1 p 1 -> n ab 1 ij 1

    # Sum over batch dimension
    contracted = np.sum(contracted, axis=0)  # n ab 1 ij 1 -> ab 1 ij 1

    # Reduce the a and b shell axes
    contracted = _reduce_primitive_pair(contracted[:, 0, :, 0], pair_ab, axis=0)  # ab ij -> ab ij

    # Unpack the contracted result into the final integral array
    contracted = _unpack_from_shell_pair_major(  # ab ij -> ai bj
        contracted, pair_ab.shell_i, pair_ab.shell_j, spherical=spherical
    )

    return contracted
