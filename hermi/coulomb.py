"""Coulomb integrals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.boys import boys
from hermi.hermite import expansion_pair_matrix, hermite_indices
from hermi.integral import _reduce_primitive_pair, _unpack_from_shell_pair_major, hermite_operator_matrix_indices
from hermi.screening import AttenuationScreening
from hermi.symmetry import SymmetryN4S8
from hermi.util import outer_powers

if TYPE_CHECKING:
    from typing import Literal

    from numpy import float64

    from hermi._typing import array
    from hermi.basis import Basis, ShellPairView


@dataclass
class CoulombIntermediates22:
    r"""Intermediates for primitive :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integrals."""

    exponent_product: array[tuple[int, int], float64]
    exponent_sum: array[tuple[int, int], float64]

    @classmethod
    def from_shell_pairs(cls, pair_ab: ShellPairView, pair_cd: ShellPairView) -> CoulombIntermediates22:
        """Compute shared exponents for a shell quartet."""
        exponent_product = np.multiply.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        exponent_sum = np.add.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        return cls(exponent_product=exponent_product, exponent_sum=exponent_sum)


def coulomb_operator_hermite_22(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates22 | None = None,
    parity: Literal["bra", "ket"] = "ket",
) -> tuple[
    array[tuple[int, int, int, int], float64],
    array[tuple[int], float64],
]:
    r"""Compute the Coulomb operator in the Hermite basis for :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integrals.

    Args:
        pair_ab: First contracted shell pair.
        pair_cd: Second contracted shell pair.
        intermediates: Precomputed intermediates for the shell pairs.
        parity: Which side of the operator to apply the sign to.

    Returns:
        Hermite operator matrix and signs.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates22.from_shell_pairs(pair_ab, pair_cd)
    max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
    if parity == "bra":
        displacement = pair_cd.position.reshape(1, -1, 3) - pair_ab.position.reshape(-1, 1, 3)
    elif parity == "ket":
        displacement = pair_ab.position.reshape(-1, 1, 3) - pair_cd.position.reshape(1, -1, 3)
    else:
        raise ValueError(f"invalid parity: {parity!r}")
    rho = intermediates.exponent_product / intermediates.exponent_sum
    argument = np.einsum("...,...x,...x->...", rho, displacement, displacement, optimize=True)

    auxiliary = np.zeros((max_order + 1,) * 4 + (len(pair_ab), len(pair_cd)), dtype=np.float64)

    # R_n^000 fundamental
    auxiliary[0, 0, 0] = np.moveaxis(outer_powers(-2 * rho, max_order) * boys(argument, max_order), -1, 0)

    for t, u, v in hermite_indices(max_order)[1:]:
        order = t + u + v
        valid = max_order - order + 1

        if v:
            # R_n^t00
            auxiliary[t, u, v, :valid] = auxiliary[t, u, v - 1, 1 : valid + 1] * displacement[..., 2]
            if v > 1:
                auxiliary[t, u, v, :valid] += (v - 1) * auxiliary[t, u, v - 2, 1 : valid + 1]

        elif u:
            # R_n^tu0
            auxiliary[t, u, v, :valid] = auxiliary[t, u - 1, v, 1 : valid + 1] * displacement[..., 1]
            if u > 1:
                auxiliary[t, u, v, :valid] += (u - 1) * auxiliary[t, u - 2, v, 1 : valid + 1]

        else:
            # R_n^tuv
            auxiliary[t, u, v, :valid] = auxiliary[t - 1, u, v, 1 : valid + 1] * displacement[..., 0]
            if t > 1:
                auxiliary[t, u, v, :valid] += (t - 1) * auxiliary[t - 2, u, v, 1 : valid + 1]

    indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
    sum_indices, signs = hermite_operator_matrix_indices(
        pair_ab.angular_momentum,
        pair_cd.angular_momentum,
        sign_side=parity,
    )
    zero_order = np.moveaxis(auxiliary[*indices, 0], 0, -1)
    matrix = np.take(zero_order, sum_indices, axis=-1)

    return matrix, signs


def coulomb_integral_22(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    expansion_ab: array[tuple[int, int, int], float64] | None = None,
    expansion_cd: array[tuple[int, int, int], float64] | None = None,
    intermediates: CoulombIntermediates22 | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute a contracted :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integral block.

    Args:
        pair_ab: First contracted shell pair.
        pair_cd: Second contracted shell pair.
        expansion_ab: Expansion coefficients for the first shell pair.
        expansion_cd: Expansion coefficients for the second shell pair.
        intermediates: Precomputed intermediates for the shell pairs.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integral block.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates22.from_shell_pairs(pair_ab, pair_cd)
    if expansion_ab is None:
        expansion_ab = expansion_pair_matrix(pair_ab, spherical=spherical)
    if expansion_cd is None:
        expansion_cd = expansion_pair_matrix(pair_cd, spherical=spherical)
    coulomb, signs = coulomb_operator_hermite_22(pair_ab, pair_cd, intermediates=intermediates, parity="bra")

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

    # ab cd p [q], 1 cd kl [q] -> ab cd p kl
    contracted = coulomb @ expansion_cd.mT

    # Reduce the c and d shell axes
    contracted = _reduce_primitive_pair(contracted, pair_cd, axis=1)

    # ab 1 ij [p], ab cd [p] kl -> ab cd ij kl
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


def coulomb_integrals_22(
    basis: Basis,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute all contracted :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integrals.

    Args:
        basis: Basis set.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical :math:`(ab|\hat{O}_\mathrm{coulomb}|cd)` integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    tensor = np.zeros((num_aos, num_aos, num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get the symmetry helper
    symmetry = SymmetryN4S8(tensor.shape)

    # Find pairs and expansion coefficients
    pairs = list(basis.iter_shell_pair_groups(unique=True, screening=AttenuationScreening(basis)))
    expansions = {pair: expansion_pair_matrix(pair, spherical=spherical) for pair in pairs}

    # Compute tensor blocks
    for m, pair_ab in enumerate(pairs):
        i, j = pair_ab.spherical_ao_slices if spherical else pair_ab.cartesian_ao_slices
        for n, pair_cd in enumerate(pairs):
            if n < m:
                continue
            k, l = pair_cd.spherical_ao_slices if spherical else pair_cd.cartesian_ao_slices
            intermediates = CoulombIntermediates22.from_shell_pairs(pair_ab, pair_cd)
            block = coulomb_integral_22(
                pair_ab,
                pair_cd,
                expansion_ab=expansions[pair_ab],
                expansion_cd=expansions[pair_cd],
                intermediates=intermediates,
                spherical=spherical,
            )
            symmetry.set_value(tensor, (i, j, k, l), block)

    # Reorder the tensor to match the original basis ordering
    tensor = tensor[np.ix_(indices, indices, indices, indices)]

    return tensor
