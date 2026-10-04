"""Electron-repulsion integrals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pymmd.coulomb import CoulombIntermediates, auxiliary_coulomb_matrix
from pymmd.hermite import expansion_matrix
from pymmd.util import num_cartesian_indices

if TYPE_CHECKING:
    from numpy import float64

    from pymmd._typing import array
    from pymmd.shell import Basis, ShellPairView


def electron_repulsion_integral(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    expansion_ab: array[tuple[int, int, int], float64] | None = None,
    expansion_cd: array[tuple[int, int, int], float64] | None = None,
    intermediates: CoulombIntermediates | None = None,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute a contracted Cartesian electron-repulsion integral block.

    Args:
        pair_ab: First contracted shell pair.
        pair_cd: Second contracted shell pair.
        expansion_ab: Expansion coefficients for the first shell pair.
        expansion_cd: Expansion coefficients for the second shell pair.
        intermediates: Precomputed intermediates for the shell pairs.

    Returns:
        Contracted Cartesian ERI block.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
    if expansion_ab is None:
        expansion_ab = expansion_matrix(pair_ab)
    if expansion_cd is None:
        expansion_cd = expansion_matrix(pair_cd)
    coulomb = auxiliary_coulomb_matrix(pair_ab, pair_cd, intermediates=intermediates)

    # ab ij p -> ab 1 ij p
    expansion_ab = np.expand_dims(expansion_ab, axis=1)

    # cd kl q -> 1 cd kl q
    expansion_cd = np.expand_dims(expansion_cd, axis=0)

    # ab ij [p], ab cd [p] q -> ab cd ij q
    transformed = expansion_ab @ coulomb

    # cd ij [q], cd kl [q] -> ab cd ij kl
    transformed = transformed @ expansion_cd.mT

    # a b, c d -> (a b) (c d)
    contraction = np.multiply.outer(pair_ab.coefficient.ravel(), pair_cd.coefficient.ravel())
    contraction /= intermediates.exponent_product
    contraction /= np.sqrt(intermediates.exponent_sum)

    # [ab cd] ij kl, [ab cd] -> ij kl
    contracted = np.tensordot(transformed, contraction, axes=([0, 1], [0, 1]))

    return (2 * np.pi**2.5) * contracted.reshape(  # type: ignore[no-any-return]
        num_cartesian_indices(pair_ab.shell_i.angular_momentum),
        num_cartesian_indices(pair_ab.shell_j.angular_momentum),
        num_cartesian_indices(pair_cd.shell_i.angular_momentum),
        num_cartesian_indices(pair_cd.shell_j.angular_momentum),
    )


def electron_repulsion_integrals(basis: Basis) -> array[tuple[int, int, int, int], float64]:
    r"""Compute all contracted Cartesian electron-repulsion integrals.

    Args:
        basis: Basis set.

    Returns:
        Contracted Cartesian ERI tensor.
    """
    eri = np.zeros(
        (basis.num_contracted_aos, basis.num_contracted_aos, basis.num_contracted_aos, basis.num_contracted_aos),
        dtype=np.float64,
    )

    pairs = list(basis.iter_shell_pairs(unique=True))
    expansions = [expansion_matrix(pair) for pair in pairs]
    for m, (pair_ab, expansion_ab) in enumerate(zip(pairs, expansions)):
        i, j = pair_ab.ao
        for n, (pair_cd, expansion_cd) in enumerate(zip(pairs, expansions)):
            if n < m:
                continue
            k, l = pair_cd.ao
            intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
            block = electron_repulsion_integral(
                pair_ab,
                pair_cd,
                expansion_ab=expansion_ab,
                expansion_cd=expansion_cd,
                intermediates=intermediates,
            )
            eri[i, j, k, l] = block
            eri[i, j, l, k] = block.transpose(0, 1, 3, 2)
            eri[j, i, k, l] = block.transpose(1, 0, 2, 3)
            eri[j, i, l, k] = block.transpose(1, 0, 3, 2)
            eri[k, l, i, j] = block.transpose(2, 3, 0, 1)
            eri[k, l, j, i] = block.transpose(2, 3, 1, 0)
            eri[l, k, i, j] = block.transpose(3, 2, 0, 1)
            eri[l, k, j, i] = block.transpose(3, 2, 1, 0)

    return eri
