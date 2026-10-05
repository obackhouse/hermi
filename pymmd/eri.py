"""Electron-repulsion integrals."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from pymmd.coulomb import CoulombIntermediates, auxiliary_coulomb_matrix
from pymmd.hermite import expansion_matrix
from pymmd.screening import AttenuationScreening, CauchySchwarzScreening
from pymmd.util import num_cartesian_functions, num_spherical_functions, rebase_slice

if TYPE_CHECKING:
    from numpy import float64, int64

    from pymmd._typing import array
    from pymmd.basis import Basis, ShellPairView, ShellView


@lru_cache(maxsize=None)
def _shell_starts(shell: ShellView) -> array[tuple[int], int64]:
    """Return primitive offsets for each contracted shell in a view."""
    _, starts = np.unique(shell.shells.shell_indices[shell.indices], return_index=True)
    return np.sort(starts)


def electron_repulsion_integral(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    expansion_ab: array[tuple[int, int, int], float64] | None = None,
    expansion_cd: array[tuple[int, int, int], float64] | None = None,
    intermediates: CoulombIntermediates | None = None,
    num_shells: tuple[int, int, int, int] = (1, 1, 1, 1),
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    r"""Compute a contracted electron-repulsion integral block.

    Args:
        pair_ab: First contracted shell pair.
        pair_cd: Second contracted shell pair.
        expansion_ab: Expansion coefficients for the first shell pair.
        expansion_cd: Expansion coefficients for the second shell pair.
        intermediates: Precomputed intermediates for the shell pairs.
        num_shells: Number of shells in each contracted shell pair.
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
    num_functions = num_spherical_functions if spherical else num_cartesian_functions
    coulomb = auxiliary_coulomb_matrix(pair_ab, pair_cd, intermediates=intermediates)

    # ab ij p -> ab 1 ij p
    expansion_ab = np.expand_dims(expansion_ab, axis=1)

    # cd kl q -> 1 cd kl q
    expansion_cd = np.expand_dims(expansion_cd, axis=0)

    # ab cd p [q], cd kl [q] -> ab cd p kl
    transformed = coulomb @ expansion_cd.mT

    # ab ij [p], ab cd [p] kl
    transformed = expansion_ab @ transformed

    # a b, c d -> (a b) (c d)
    contraction = np.multiply.outer(pair_ab.coefficient.ravel(), pair_cd.coefficient.ravel())
    contraction /= intermediates.exponent_product
    contraction /= np.sqrt(intermediates.exponent_sum)

    # ab cd ij kl, ab cd -> ab cd ij kl
    transformed *= contraction[..., None, None]

    # ab cd ij kl -> a b c d i j k l
    transformed = transformed.reshape(
        len(pair_ab.shell_i),
        len(pair_ab.shell_j),
        len(pair_cd.shell_i),
        len(pair_cd.shell_j),
        num_functions(pair_ab.shell_i.angular_momentum),
        num_functions(pair_ab.shell_j.angular_momentum),
        num_functions(pair_cd.shell_i.angular_momentum),
        num_functions(pair_cd.shell_j.angular_momentum),
    )

    # Reduce the most compressive shell axes first to shrink later intermediates.
    reductions = (
        (0, pair_ab.shell_i),
        (1, pair_ab.shell_j),
        (2, pair_cd.shell_i),
        (3, pair_cd.shell_j),
    )
    for axis, shell in sorted(reductions, key=lambda item: item[1].num_shells / len(item[1])):
        transformed = np.add.reduceat(transformed, _shell_starts(shell), axis=axis)

    # a b c d i j k l -> a i b j c k d l
    contracted = transformed.transpose(0, 4, 1, 5, 2, 6, 3, 7)

    # a i b j c k d l -> (a i) (b j) (c k) (d l)
    contracted = contracted.reshape(
        num_shells[0] * num_functions(pair_ab.shell_i.angular_momentum),
        num_shells[1] * num_functions(pair_ab.shell_j.angular_momentum),
        num_shells[2] * num_functions(pair_cd.shell_i.angular_momentum),
        num_shells[3] * num_functions(pair_cd.shell_j.angular_momentum),
    )

    return (2 * np.pi**2.5) * contracted  # type: ignore[no-any-return]


def electron_repulsion_integral_diagonal(
    basis: Basis,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    r"""Compute the diagonal of the electron-repulsion integral tensor.

    Args:
        basis: Basis set.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical ERI tensor diagonal.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    eri = np.zeros((num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Find pairs and expansion coefficients
    pairs = list(basis.iter_shell_pair_groups(unique=True, screening=AttenuationScreening(basis)))
    expansions = [expansion_matrix(pair, spherical=spherical) for pair in pairs]

    # Compute ERI tensor diagonal blocks
    for m, (pair_ab, expansion_ab) in enumerate(zip(pairs, expansions)):
        i, j = pair_ab.spherical_ao_slices if spherical else pair_ab.cartesian_ao_slices
        num_shells = (pair_ab.shell_i.num_shells, pair_ab.shell_j.num_shells)
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_ab)
        block = electron_repulsion_integral(
            pair_ab,
            pair_ab,
            expansion_ab=expansion_ab,
            expansion_cd=expansion_ab,
            intermediates=intermediates,
            num_shells=num_shells + num_shells,
            spherical=spherical,
        )
        block = block.diagonal(axis1=1, axis2=3).diagonal(axis1=0, axis2=1).T
        eri[i, j] = block
        eri[j, i] = block.T

    # Reorder the ERI tensor diagonal to match the original basis ordering
    eri = eri[indices][:, indices]

    return eri


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

    # Get the Cauchy-Schwarz screening
    metric = electron_repulsion_integral_diagonal(basis, spherical=spherical) ** 0.5
    screening = CauchySchwarzScreening(basis, metric=metric, spherical=spherical)

    # Find pairs and expansion coefficients
    pairs = list(basis.iter_shell_pair_groups(unique=True, screening=AttenuationScreening(basis)))
    expansions = {pair: expansion_matrix(pair, spherical=spherical) for pair in pairs}

    def expansions_subview(group_ab: ShellPairView, pair_ab: ShellPairView) -> array[tuple[int, int, int], float64]:
        """Get the expansion coefficients for a subview of a shell pair group."""
        if group_ab is pair_ab:
            return expansions[pair_ab]
        i = rebase_slice(pair_ab.indices_i, group_ab.indices_i)
        j = rebase_slice(pair_ab.indices_j, group_ab.indices_j)
        n_cart, n_herm = expansions[group_ab].shape[-2:]
        expansion = expansions[group_ab].reshape(len(group_ab.shell_i), len(group_ab.shell_j), n_cart, n_herm)
        return expansion[i, j].reshape(len(pair_ab), n_cart, n_herm)

    # Compute ERI tensor blocks
    for m, group_ab in enumerate(pairs):
        for n, group_cd in enumerate(pairs):
            if n < m:
                continue
            for pair_ab, pair_cd in screening((group_ab, group_cd)):
                i, j = pair_ab.spherical_ao_slices if spherical else pair_ab.cartesian_ao_slices
                k, l = pair_cd.spherical_ao_slices if spherical else pair_cd.cartesian_ao_slices
                num_shells = (
                    pair_ab.shell_i.num_shells,
                    pair_ab.shell_j.num_shells,
                    pair_cd.shell_i.num_shells,
                    pair_cd.shell_j.num_shells,
                )
                intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
                block = electron_repulsion_integral(
                    pair_ab,
                    pair_cd,
                    expansion_ab=expansions_subview(group_ab, pair_ab),
                    expansion_cd=expansions_subview(group_cd, pair_cd),
                    intermediates=intermediates,
                    num_shells=num_shells,
                    spherical=spherical,
                )
                eri[i, j, k, l] = block
                eri[i, j, l, k] = block.transpose(0, 1, 3, 2)
                eri[j, i, k, l] = block.transpose(1, 0, 2, 3)
                eri[j, i, l, k] = block.transpose(1, 0, 3, 2)
                eri[k, l, i, j] = block.transpose(2, 3, 0, 1)
                eri[k, l, j, i] = block.transpose(2, 3, 1, 0)
                eri[l, k, i, j] = block.transpose(3, 2, 0, 1)
                eri[l, k, j, i] = block.transpose(3, 2, 1, 0)

    # Reorder the ERI tensor to match the original basis ordering
    eri = eri[indices][:, indices][:, :, indices][:, :, :, indices]

    return eri
