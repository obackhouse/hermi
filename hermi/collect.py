"""Integral collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hermi.symmetry import SymmetryN2S1, SymmetryN2S2, SymmetryN4S8

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.basis import Basis
    from hermi.operator import Operator
    from hermi.symmetry import Symmetry


def collect_integral_11(
    basis: Basis,
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] = SymmetryN2S2,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (1, 1) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (1, 1) signature.
        symmetry_type: Symmetry helper class for the integral matrix.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    matrix = np.zeros((num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get shells and shell pairs
    shells = list(basis.iter_shell_groups())

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells), len(shells)))

    # Compute matrix blocks
    for (a, b), _ in symmetry.canonical_indices():
        shell_a = shells[a]
        shell_b = shells[b]
        i = shell_a.spherical_ao_slice if spherical else shell_a.cartesian_ao_slice
        j = shell_b.spherical_ao_slice if spherical else shell_b.cartesian_ao_slice
        block = operator.contract((shell_a, shell_b), spherical=spherical)
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(indices, indices)]

    return matrix


def collect_integral_20(
    basis: Basis,
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] = SymmetryN2S2,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (2, 0) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (2, 0) signature.
        symmetry_type: Symmetry helper class for the integral matrix.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    matrix = np.zeros((num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get shells and shell pairs
    shells = list(basis.iter_shell_groups())
    pairs = list(basis.iter_shell_pair_groups())

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells), len(shells)))

    # Compute matrix blocks
    for (a, b), _ in symmetry.canonical_indices():
        pair = pairs[symmetry.ravel_asymmetric_index((a, b))]
        i = pair.shell_i.spherical_ao_slice if spherical else pair.shell_i.cartesian_ao_slice
        j = pair.shell_j.spherical_ao_slice if spherical else pair.shell_j.cartesian_ao_slice
        block = operator.contract((pair, None), spherical=spherical)
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(indices, indices)]

    return matrix


def collect_integral_22(
    basis: Basis,
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]] = SymmetryN4S8,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    """Collect contracted AO integrals for a (2, 2) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (2, 2) signature.
        symmetry_type: Symmetry helper class for the integral matrix.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    matrix = np.zeros((num_aos, num_aos, num_aos, num_aos), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get shells and shell pairs
    shells = list(basis.iter_shell_groups())
    pairs = list(basis.iter_shell_pair_groups())

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells), len(shells), len(shells), len(shells)))
    pair_symmetry = SymmetryN2S1((len(shells), len(shells)))

    # Compute matrix blocks
    for (a, b, c, d), _ in symmetry.canonical_indices():
        pair_ab = pairs[pair_symmetry.ravel_asymmetric_index((a, b))]
        pair_cd = pairs[pair_symmetry.ravel_asymmetric_index((c, d))]
        i = pair_ab.shell_i.spherical_ao_slice if spherical else pair_ab.shell_i.cartesian_ao_slice
        j = pair_ab.shell_j.spherical_ao_slice if spherical else pair_ab.shell_j.cartesian_ao_slice
        k = pair_cd.shell_i.spherical_ao_slice if spherical else pair_cd.shell_i.cartesian_ao_slice
        l = pair_cd.shell_j.spherical_ao_slice if spherical else pair_cd.shell_j.cartesian_ao_slice
        block = operator.contract(
            (pair_ab, pair_cd),
            spherical=spherical,
        )
        symmetry.set_value(matrix, (i, j, k, l), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(indices, indices, indices, indices)]

    return matrix
