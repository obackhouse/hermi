"""Integral collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hermi.symmetry import SymmetryN2S1, SymmetryN2S2, SymmetryN4S1, SymmetryN4S4, SymmetryN4S8

if TYPE_CHECKING:
    from typing import Iterable

    from numpy import float64, int64

    from hermi._typing import array
    from hermi.basis import Basis, ShellView
    from hermi.operators.operator import Operator
    from hermi.symmetry import Symmetry


def _get_ao_selection(
    ao_slice: slice,
    indices: array[tuple[int], int64],
    num_aos: int,
) -> tuple[array[tuple[int], int64], array[tuple[int], int64], array[tuple[int], int64]]:
    """Get AO indices and the permutation to restore their original order.

    Args:
        ao_slice: AO slice in the original basis order.
        indices: Map from original AO indices to angular-momentum-ordered indices.
        num_aos: Number of AOs in the basis.

    Returns:
        Original indices, ordered indices, and indices to restore the selected order.
    """
    selected_original = np.arange(*ao_slice.indices(num_aos), dtype=np.int64)
    selected_ordered = np.sort(indices[selected_original])
    restore = np.searchsorted(selected_ordered, indices[selected_original])
    return selected_original, selected_ordered, restore


def _get_shell_selection(
    shell: ShellView,
    selected_ordered: array[tuple[int], int64],
    *,
    spherical: bool = False,
) -> tuple[slice, array[tuple[int], int64]] | None:
    """Get the output slice and local AO indices selected from a shell group.

    Args:
        shell: Shell group to select from.
        selected_ordered: Selected AO indices in angular-momentum order.
        spherical: Use real spherical instead of Cartesian AO indices.

    Returns:
        Output slice and shell-local indices, or ``None`` if the shell is not selected.
    """
    ao_slice = shell.spherical_ao_slice if spherical else shell.cartesian_ao_slice
    selected_start = int(np.searchsorted(selected_ordered, ao_slice.start))
    selected_stop = int(np.searchsorted(selected_ordered, ao_slice.stop))
    if selected_start == selected_stop:
        return None
    local_indices = selected_ordered[selected_start:selected_stop] - ao_slice.start
    return slice(selected_start, selected_stop), local_indices


def _get_selected_shell_groups(
    shell_groups: Iterable[ShellView],
    selected_ordered: array[tuple[int], int64],
    *,
    spherical: bool = False,
) -> list[tuple[ShellView, slice, array[tuple[int], int64]]]:
    """Get the selected shell groups and their AO slices.

    Args:
        shell_groups: Shell groups in angular-momentum order.
        selected_ordered: Selected AO indices in angular-momentum order.
        spherical: Use real spherical instead of Cartesian AO indices.

    Returns:
        Selected shell groups, output slices, and shell-local AO indices.
    """
    selected_shells = []
    for shell in shell_groups:
        selection = _get_shell_selection(shell, selected_ordered, spherical=spherical)
        if selection is not None:
            output_slice, local_indices = selection
            selected_shells.append((shell, output_slice, local_indices))
    return selected_shells


def collect_integral_11(
    basis: Basis,
    operator: Operator,
    *,
    ao_slices: tuple[slice, slice] = (slice(None), slice(None)),
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] | None = None,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (1, 1) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (1, 1) signature.
        ao_slices: AO slices for the bra and ket axes in the original basis order.
        symmetry_type: Symmetry helper class for the integral matrix. The default value depends on whether the AO slices
            are equal.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get AO selections
    selected_original, selected_ordered, restore = zip(
        *tuple(_get_ao_selection(ao_slice, indices, num_aos) for ao_slice in ao_slices)
    )
    same_ao_slices = np.array_equal(selected_original[0], selected_original[1])
    if symmetry_type is None:
        symmetry_type = SymmetryN2S2 if same_ao_slices else SymmetryN2S1

    # Get selected shell groups and AO slices
    shells = [
        _get_selected_shell_groups(basis.iter_shell_groups(), selected_ordered[axis], spherical=spherical)
        for axis in range(2)
    ]

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells[0]), len(shells[1])))
    symmetry.assert_symmetric_sequence([selection.tolist() for selection in selected_original])

    # Compute matrix blocks
    matrix = np.zeros((len(selected_ordered[0]), len(selected_ordered[1])), dtype=np.float64)
    for (a, b), _ in symmetry.canonical_indices():
        shell_a, i, indices_a = shells[0][a]
        shell_b, j, indices_b = shells[1][b]
        block = operator.contract((shell_a, shell_b), spherical=spherical)
        block = block[np.ix_(indices_a, indices_b)]
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(*restore)]

    return matrix


def collect_integral_20(
    basis: Basis,
    operator: Operator,
    *,
    ao_slices: tuple[slice, slice] = (slice(None), slice(None)),
    symmetry_type: type[Symmetry[tuple[int, int], tuple[slice, slice]]] | None = None,
    spherical: bool = False,
) -> array[tuple[int, int], float64]:
    """Collect contracted AO integrals for a (2, 0) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (2, 0) signature.
        ao_slices: AO slices for the two axes in the original basis order.
        symmetry_type: Symmetry helper class for the integral matrix. The default value depends on whether the AO slices
            are equal.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get AO selections
    selected_original, selected_ordered, restore = zip(
        *tuple(_get_ao_selection(ao_slice, indices, num_aos) for ao_slice in ao_slices)
    )
    same_ao_slices = np.array_equal(selected_original[0], selected_original[1])
    if symmetry_type is None:
        symmetry_type = SymmetryN2S2 if same_ao_slices else SymmetryN2S1

    # Get shells and shell pairs
    shell_groups = list(basis.iter_shell_groups())
    shells = [
        _get_selected_shell_groups(shell_groups, selected_ordered[axis], spherical=spherical) for axis in range(2)
    ]
    pairs = list(basis.iter_shell_pair_groups())

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells[0]), len(shells[1])))
    symmetry.assert_symmetric_sequence([selection.tolist() for selection in selected_original])
    pair_symmetry = SymmetryN2S1((len(shell_groups), len(shell_groups)))
    shell_group_indices = {shell.indices.start: index for index, shell in enumerate(shell_groups)}
    matrix = np.zeros((len(selected_ordered[0]), len(selected_ordered[1])), dtype=np.float64)

    # Compute matrix blocks
    for (a, b), _ in symmetry.canonical_indices():
        shell_a, i, indices_a = shells[0][a]
        shell_b, j, indices_b = shells[1][b]
        shell_i = shell_group_indices[shell_a.indices.start]
        shell_j = shell_group_indices[shell_b.indices.start]
        pair = pairs[pair_symmetry.ravel_asymmetric_index((shell_i, shell_j))]
        block = operator.contract((pair, None), spherical=spherical)
        block = block[np.ix_(indices_a, indices_b)]
        symmetry.set_value(matrix, (i, j), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(*restore)]

    return matrix


def collect_integral_22(
    basis: Basis,
    operator: Operator,
    *,
    ao_slices: tuple[slice, slice, slice, slice] = (slice(None), slice(None), slice(None), slice(None)),
    symmetry_type: type[Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]] | None = None,
    spherical: bool = False,
) -> array[tuple[int, int, int, int], float64]:
    """Collect contracted AO integrals for a (2, 2) integral.

    Args:
        basis: Basis set.
        operator: Operator to collect integrals for with (2, 2) signature.
        ao_slices: AO slices for the four axes in the original basis order.
        symmetry_type: Symmetry helper class for the integral tensor. The default value depends on the AO slices.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Contracted Cartesian or real spherical AO integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get AO selections
    selected_original, selected_ordered, restore = zip(
        *tuple(_get_ao_selection(ao_slice, indices, num_aos) for ao_slice in ao_slices)
    )
    same_ao_pairs = np.array_equal(selected_original[0], selected_original[1]) and np.array_equal(
        selected_original[2], selected_original[3]
    )
    same_ao_slices = same_ao_pairs and np.array_equal(selected_original[0], selected_original[2])
    if symmetry_type is None:
        if same_ao_slices:
            symmetry_type = SymmetryN4S8
        elif same_ao_pairs:
            symmetry_type = SymmetryN4S4
        else:
            symmetry_type = SymmetryN4S1

    # Get selected shell groups and AO slices
    shell_groups = list(basis.iter_shell_groups())
    shells = [
        _get_selected_shell_groups(shell_groups, selected_ordered[axis], spherical=spherical) for axis in range(4)
    ]
    pairs = list(basis.iter_shell_pair_groups())

    # Get the symmetry helper
    symmetry = symmetry_type((len(shells[0]), len(shells[1]), len(shells[2]), len(shells[3])))
    symmetry.assert_symmetric_sequence([selection.tolist() for selection in selected_original])
    pair_symmetry = SymmetryN2S1((len(shell_groups), len(shell_groups)))

    # Compute matrix blocks
    matrix = np.zeros(
        (len(selected_ordered[0]), len(selected_ordered[1]), len(selected_ordered[2]), len(selected_ordered[3])),
        dtype=np.float64,
    )
    shell_group_indices = {shell.indices.start: index for index, shell in enumerate(shell_groups)}
    for (a, b, c, d), _ in symmetry.canonical_indices():
        shell_a, i, indices_a = shells[0][a]
        shell_b, j, indices_b = shells[1][b]
        shell_c, k, indices_c = shells[2][c]
        shell_d, l, indices_d = shells[3][d]
        index_ab = pair_symmetry.ravel_asymmetric_index(
            (shell_group_indices[shell_a.indices.start], shell_group_indices[shell_b.indices.start])
        )
        index_cd = pair_symmetry.ravel_asymmetric_index(
            (shell_group_indices[shell_c.indices.start], shell_group_indices[shell_d.indices.start])
        )
        pair_ab = pairs[index_ab]
        pair_cd = pairs[index_cd]
        block = operator.contract((pair_ab, pair_cd), spherical=spherical)
        block = block[np.ix_(indices_a, indices_b, indices_c, indices_d)]
        symmetry.set_value(matrix, (i, j, k, l), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(*restore)]

    return matrix
