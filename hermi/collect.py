"""Integral collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from hermi.symmetry import SymmetryN2S1, SymmetryN2S2, SymmetryN4S4, SymmetryN4S8

if TYPE_CHECKING:
    from typing import Literal

    from numpy import float64, int64

    from hermi._typing import array
    from hermi.basis import Basis
    from hermi.operators.operator import Operator
    from hermi.symmetry import Symmetry


def _order_basis(
    basis: tuple[Basis, Basis], *, spherical: bool = False
) -> tuple[tuple[Basis, Basis], tuple[array[tuple[int], int64], array[tuple[int], int64]]]:
    """Order the basis set to make equal angular momentum shells contiguous.

    Args:
        basis: Basis sets for the bra and ket.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Ordered basis sets and indices to reorder the integrals back to the original order.
    """
    basis_a, basis_b = basis
    basis_a, indices_a = basis_a.ordered_by_angular_momentum(spherical=spherical)
    basis_b, indices_b = basis_b.ordered_by_angular_momentum(spherical=spherical)
    return (basis_a, basis_b), (indices_a, indices_b)


@overload
def get_shape(
    basis: tuple[Basis, Basis],
    signature: tuple[Literal[1], Literal[0]] | tuple[Literal[1], Literal[0]],
    *,
    spherical: bool = False,
) -> tuple[int]: ...
@overload
def get_shape(
    basis: tuple[Basis, Basis],
    signature: tuple[Literal[1], Literal[1]] | tuple[Literal[2], Literal[0]] | tuple[Literal[0], Literal[2]],
    *,
    spherical: bool = False,
) -> tuple[int, int]: ...
@overload
def get_shape(
    basis: tuple[Basis, Basis],
    signature: tuple[Literal[2], Literal[1]] | tuple[Literal[1], Literal[2]],
    *,
    spherical: bool = False,
) -> tuple[int, int, int]: ...
@overload
def get_shape(
    basis: tuple[Basis, Basis],
    signature: tuple[Literal[2], Literal[2]],
    *,
    spherical: bool = False,
) -> tuple[int, int, int, int]: ...


def get_shape[NBra: int, NKet: int](
    basis: tuple[Basis, Basis],
    signature: tuple[NBra, NKet],
    *,
    spherical: bool = False,
) -> tuple[int, ...]:
    """Get the shape of the integral matrix for a given basis set and integral signature.

    Args:
        basis: Basis sets for the bra and ket.
        signature: Integral signature.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Shape of the integral matrix.
    """
    basis_a, basis_b = basis
    num_aos_a = basis_a.num_contracted_spherical_aos if spherical else basis_a.num_contracted_cartesian_aos
    num_aos_b = basis_b.num_contracted_spherical_aos if spherical else basis_b.num_contracted_cartesian_aos
    return (num_aos_a,) * signature[0] + (num_aos_b,) * signature[1]


def collect_integral_11(
    basis: Basis | tuple[Basis, Basis],
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (1, 1) integral.

    Args:
        basis: Basis set to collect integrals for. Can be a pair of basis sets in which case the
            first basis set is used for the bra and the second for the ket.
        operator: Operator to collect integrals for with (1, 1) signature.
        symmetry_type: Symmetry helper class for the integral matrix. The default value depends on
            whether the basis set is split.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    if not isinstance(basis, tuple):
        basis = (basis, basis)
    if symmetry_type is None:
        symmetry_type = SymmetryN2S2 if basis[0] is basis[1] else SymmetryN2S1
    matrix = np.zeros(get_shape(basis, (1, 1), spherical=spherical), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = _order_basis(basis, spherical=spherical)

    # Get shells and shell pairs
    shells = (list(basis[0].iter_shell_groups()), list(basis[1].iter_shell_groups()))

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells[0]), len(shells[1])))

    # Compute matrix blocks
    for (a, b), _ in symmetry.canonical_indices():
        shell_a = shells[0][a]
        shell_b = shells[1][b]
        i = shell_a.spherical_ao_slice if spherical else shell_a.cartesian_ao_slice
        j = shell_b.spherical_ao_slice if spherical else shell_b.cartesian_ao_slice
        block = operator.contract((shell_a, shell_b), spherical=spherical)
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(indices[0], indices[1])]

    return matrix


def collect_integral_20(
    basis: Basis | tuple[Basis, Basis],
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (2, 0) integral.

    Args:
        basis: Basis set to collect integrals for. Can be a pair of basis sets in which case the
            first basis set is used for the bra and the second for the ket.
        operator: Operator to collect integrals for with (2, 0) signature.
        symmetry_type: Symmetry helper class for the integral matrix.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    if not isinstance(basis, tuple):
        basis = (basis, basis)
    if symmetry_type is None:
        symmetry_type = SymmetryN2S2
    matrix = np.zeros(get_shape(basis, (2, 0), spherical=spherical), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = _order_basis(basis, spherical=spherical)

    # Get shells and shell pairs
    shells = list(basis[0].iter_shell_groups())
    pairs = list(basis[0].iter_shell_pair_groups())

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
    matrix = matrix[np.ix_(indices[0], indices[0])]

    return matrix


def collect_integral_22(
    basis: Basis | tuple[Basis, Basis],
    operator: Operator,
    symmetry_type: type[Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]] | None = None,
    *,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    """Collect contracted AO integrals for a (2, 2) integral.

    Args:
        basis: Basis set to collect integrals for. Can be a pair of basis sets in which case the
            first basis set is used for the bra and the second for the ket.
        operator: Operator to collect integrals for with (2, 2) signature.
        symmetry_type: Symmetry helper class for the integral matrix.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    if not isinstance(basis, tuple):
        basis = (basis, basis)
    if symmetry_type is None:
        symmetry_type = SymmetryN4S8 if basis[0] is basis[1] else SymmetryN4S4
    matrix = np.zeros(get_shape(basis, (2, 2), spherical=spherical), dtype=np.float64)

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = _order_basis(basis, spherical=spherical)

    # Get shells and shell pairs
    shells = (list(basis[0].iter_shell_groups()), list(basis[1].iter_shell_groups()))
    pairs = (list(basis[0].iter_shell_pair_groups()), list(basis[1].iter_shell_pair_groups()))

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells[0]), len(shells[0]), len(shells[1]), len(shells[1])))
    bra_symmetry = SymmetryN2S1((len(shells[0]), len(shells[0])))
    ket_symmetry = SymmetryN2S1((len(shells[1]), len(shells[1])))

    # Compute matrix blocks
    for (a, b, c, d), _ in symmetry.canonical_indices():
        pair_ab = pairs[0][bra_symmetry.ravel_asymmetric_index((a, b))]
        pair_cd = pairs[1][ket_symmetry.ravel_asymmetric_index((c, d))]
        i = pair_ab.shell_i.spherical_ao_slice if spherical else pair_ab.shell_i.cartesian_ao_slice
        j = pair_ab.shell_j.spherical_ao_slice if spherical else pair_ab.shell_j.cartesian_ao_slice
        k = pair_cd.shell_i.spherical_ao_slice if spherical else pair_cd.shell_i.cartesian_ao_slice
        l = pair_cd.shell_j.spherical_ao_slice if spherical else pair_cd.shell_j.cartesian_ao_slice
        block = operator.contract((pair_ab, pair_cd), spherical=spherical)
        symmetry.set_value(matrix, (i, j, k, l), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(indices[0], indices[0], indices[1], indices[1])]

    return matrix
