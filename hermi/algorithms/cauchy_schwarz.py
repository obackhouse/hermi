"""Cauchy-Schwarz screening."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hermi.integrals.collect import _get_ao_selection, _get_selected_shell_groups
from hermi.integrals.expansion import Expansion
from hermi.integrals.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.operators.coulomb import Coulomb
from hermi.utils.angular import num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis


def get_cauchy_schwarz_metric(
    basis: Basis,
    *,
    spherical: bool = False,
    ao_slices: tuple[slice, slice] = (slice(None), slice(None)),
) -> array[float64]:
    r"""Compute the AO-pair diagonal Coulomb metric :math:`(ab|ab)`.

    Args:
        basis: Basis set.
        spherical: Use real spherical instead of Cartesian atomic orbitals.
        ao_slices: AO slices for the two axes in the original basis order.

    Returns:
        AO-pair diagonal Coulomb integrals.
    """
    num_aos = basis.num_contracted_spherical_aos if spherical else basis.num_contracted_cartesian_aos
    num_functions = num_spherical_functions if spherical else num_cartesian_functions

    # Order the basis set to make equal angular momentum shells contiguous
    basis, indices = basis.ordered_by_angular_momentum(spherical=spherical)

    # Get AO selections
    selected_original, selected_ordered, restore = zip(
        *tuple(_get_ao_selection(ao_slice, indices, num_aos) for ao_slice in ao_slices)
    )
    same_ao_slices = np.array_equal(selected_original[0], selected_original[1])

    # Get shells and shell pairs
    shell_groups = list(basis.iter_shells())
    shells = [
        _get_selected_shell_groups(shell_groups, selected_ordered[axis], spherical=spherical) for axis in range(2)
    ]
    symmetry = (SymmetryN2S2 if same_ao_slices else SymmetryN2S1)((len(shells[0]), len(shells[1])))
    symmetry.assert_symmetric_sequence([selection.tolist() for selection in selected_original])
    shell_indices = {shell.indices.start: index for index, shell in enumerate(shell_groups)}

    # Compute matrix blocks
    matrix = np.zeros((len(selected_ordered[0]), len(selected_ordered[1])), dtype=np.float64)
    blocks = {}
    coulomb = Coulomb()
    for (a, b), _ in symmetry.canonical_indices():
        shell_a, output_a, local_a = shells[0][a]
        shell_b, output_b, local_b = shells[1][b]
        index_a = shell_indices[shell_a.indices.start]
        index_b = shell_indices[shell_b.indices.start]

        key = (min(index_a, index_b), max(index_a, index_b))
        if key not in blocks:
            pair = basis[shell_groups[key[0]], shell_groups[key[1]]]
            operator = coulomb.get_operator_matrix((pair, pair))  # ab cd p q

            # Get the Hermite expansion of the primitive-pair
            expansion = Expansion(pair).compute(spherical=spherical)
            expansion = np.expand_dims(expansion, axis=1)  # ab ij p -> ab 1 ij p
            expansion = expansion * pair.coefficient.reshape(-1, 1, 1, 1)

            # Contract the primitive-pair and Hermite axes
            contracted = expansion @ operator  # ab 1 ij p, ab cd p q -> ab cd ij q
            contracted *= expansion.swapaxes(0, 1)  # ab cd ij q, 1 cd ij q -> ab cd ij q
            contracted = np.sum(contracted, axis=(0, 1, 3))  # ij

            # Record the block
            blocks[key] = contracted.reshape(
                num_functions(shell_groups[key[0]].angular_momentum),
                num_functions(shell_groups[key[1]].angular_momentum),
            )

        # Set the block
        block = blocks[key] if index_a <= index_b else blocks[key].T
        block = block[np.ix_(local_a, local_b)]
        symmetry.set_value(matrix, (output_a, output_b), block)

    # Reorder the matrix back to the original basis order
    matrix = matrix[np.ix_(*restore)]

    return matrix
