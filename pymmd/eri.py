"""Electron-repulsion integrals."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np

from pymmd.coulomb import CoulombIntermediates, auxiliary_coulomb_matrix
from pymmd.hermite import expansion_matrix
from pymmd.screening import AttenuationScreening
from pymmd.symmetry import SymmetryN4S8
from pymmd.util import add_reduceat, num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from numpy import float64

    from pymmd._typing import array
    from pymmd.basis import Basis, ShellPairView, ShellView


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
    expanded = add_reduceat(expanded, pair.basis.contracted_shell_starts(pair.shell_j.indices), axis=axis + 1)
    expanded = add_reduceat(expanded, pair.basis.contracted_shell_starts(pair.shell_i.indices), axis=axis)
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


def electron_repulsion_integral(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    expansion_ab: array[tuple[int, int, int], float64] | None = None,
    expansion_cd: array[tuple[int, int, int], float64] | None = None,
    intermediates: CoulombIntermediates | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute a contracted electron-repulsion integral block.

    Args:
        pair_ab: First contracted shell pair.
        pair_cd: Second contracted shell pair.
        expansion_ab: Expansion coefficients for the first shell pair.
        expansion_cd: Expansion coefficients for the second shell pair.
        intermediates: Precomputed intermediates for the shell pairs.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical ERI block.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
    if expansion_ab is None:
        expansion_ab = expansion_matrix(pair_ab, spherical=spherical)
    if expansion_cd is None:
        expansion_cd = expansion_matrix(pair_cd, spherical=spherical)
    coulomb, signs = auxiliary_coulomb_matrix(pair_ab, pair_cd, intermediates=intermediates, parity="bra")

    # cd kl q, q -> cd kl q
    expansion_ab = expansion_ab * signs

    # ab ij p -> ab 1 ij p
    expansion_ab = np.expand_dims(expansion_ab, axis=1)

    # cd kl q -> 1 cd kl q
    expansion_cd = np.expand_dims(expansion_cd, axis=0)

    # ab cd p q, ab cd -> ab cd p q
    coulomb *= np.expand_dims(np.multiply.outer(pair_ab.coefficient.ravel(), pair_cd.coefficient.ravel()), axis=(2, 3))
    coulomb /= np.expand_dims(intermediates.exponent_product, axis=(2, 3))
    coulomb /= np.expand_dims(np.sqrt(intermediates.exponent_sum), axis=(2, 3))

    # ab cd p [q], cd kl [q] -> ab cd p kl
    contracted = coulomb @ expansion_cd.mT

    # Reduce the c and d shell axes
    contracted = _reduce_primitive_pair(contracted, pair_cd, axis=1)

    # ab ij [p], ab cd [p] kl -> ab cd ij kl
    contracted = expansion_ab @ contracted

    # Reduce the a and b shell axes
    contracted = _reduce_primitive_pair(contracted, pair_ab, axis=0)

    # ab cd ij kl -> ai bj ck dl
    contracted = _unpack_from_shell_pair_major(
        contracted,
        pair_ab.shell_i,
        pair_ab.shell_j,
        pair_cd.shell_i,
        pair_cd.shell_j,
        spherical=spherical,
    )

    return (2 * np.pi**2.5) * contracted  # type: ignore[no-any-return]


def electron_repulsion_integrals(
    basis: Basis,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute all contracted electron-repulsion integrals.

    Args:
        basis: Basis set.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical ERI tensor.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    eri = np.zeros((num_aos, num_aos, num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get the symmetry helper
    symmetry = SymmetryN4S8(eri.shape)

    # Find pairs and expansion coefficients
    pairs = list(basis.iter_shell_pair_groups(unique=True, screening=AttenuationScreening(basis)))
    expansions = {pair: expansion_matrix(pair, spherical=spherical) for pair in pairs}

    # Compute ERI tensor blocks
    for m, pair_ab in enumerate(pairs):
        i, j = pair_ab.spherical_ao_slices if spherical else pair_ab.cartesian_ao_slices
        for n, pair_cd in enumerate(pairs):
            if n < m:
                continue
            k, l = pair_cd.spherical_ao_slices if spherical else pair_cd.cartesian_ao_slices
            intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
            block = electron_repulsion_integral(
                pair_ab,
                pair_cd,
                expansion_ab=expansions[pair_ab],
                expansion_cd=expansions[pair_cd],
                intermediates=intermediates,
                spherical=spherical,
            )
            symmetry.set_value(eri, (i, j, k, l), block)

    # Reorder the ERI tensor to match the original basis ordering
    eri = eri[indices][:, indices][:, :, indices][:, :, :, indices]

    return eri
