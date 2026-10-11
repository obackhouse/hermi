"""Density fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hermi.operators.coulomb import Coulomb

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.basis import Basis


def _decompose_potential(
    matrix: array[tuple[int, int], float64],
    *,
    threshold: float = 1e-10,
) -> array[tuple[int, int], float64]:
    r"""Decompose the potential matrix using Cholesky decomposition.

    Finds the matrix :math:`V` satisfying

    .. math::
        V V^\dagger = \left[(A|B)\right]^{-1}

    Args:
        matrix: Potential matrix.
        threshold: Threshold for eigenvalues to be considered non-zero.

    Returns:
        Decomposed potential matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    mask = eigenvalues > threshold
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]
    result = eigenvectors * np.expand_dims(eigenvalues, axis=0) ** -0.5
    return result


def density_fit(ao_basis: Basis, aux_basis: Basis, *, spherical: bool = False) -> array[tuple[int, int, int], float64]:
    """Prepare density fitting integrals.

    Args:
        ao_basis: Basis set.
        aux_basis: Auxiliary basis set.
        spherical: Return real spherical instead of Cartesian integrals.

    Returns:
        Density fitting integrals, with the first index corresponding to auxiliary AOs.
    """
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    num_aux_aos = aux_basis.num_contracted_spherical_aos if spherical else aux_basis.num_contracted_cartesian_aos
    ao_slice = slice(0, num_aos)
    aux_slice = slice(num_aos, num_aos + num_aux_aos)
    basis = ao_basis.concatenate(aux_basis)

    # Get the Coulomb operator and collect the integrals
    coulomb = Coulomb()
    matrix11 = coulomb.collect(basis, (1, 1), ao_slices=(aux_slice, aux_slice), spherical=spherical)
    matrix12 = coulomb.collect(basis, (1, 2), ao_slices=(aux_slice, ao_slice, ao_slice), spherical=spherical)

    # Decompose the (A|B) matrix
    decomposed_matrix = _decompose_potential(matrix11)

    # Compute the density fitting integrals
    matrix: array[tuple[int, int, int], float64] = np.einsum("PQ,Pij->Qij", decomposed_matrix, matrix12, optimize=True)

    return matrix
