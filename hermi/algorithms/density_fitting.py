"""Density fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.operators.coulomb import Coulomb

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis


@dataclass(frozen=True)
class PairSpaceProjector:
    """One factorized contribution to a targeted orbital-pair space.

    The coefficient matrices map AO functions to the left and right orbital
    spaces forming each pair. For example, occupied-virtual pairs use occupied
    coefficients on the left and virtual coefficients on the right. Multiple
    channels can be combined by passing a tuple of projectors. Weights should
    be nonnegative so the combined pair-space metric remains positive
    semidefinite.
    """

    left_coefficients: array[float64]
    right_coefficients: array[float64]
    weight: float = 1.0


@dataclass(frozen=True)
class NaturalAuxiliaryFunctions:
    """Coulomb-orthonormal auxiliary combinations adapted to pair spaces.

    ``auxiliary_coefficients`` has shape ``(n_aux, n_naf)`` and maps the
    original auxiliary basis to the retained natural auxiliary functions. The
    transform is specific to the pair spaces used to construct it and does not
    guarantee accuracy outside those spaces.
    """

    auxiliary_coefficients: array[float64]

    @classmethod
    def from_basis(
        cls,
        ao_basis: Basis,
        aux_basis: Basis,
        *,
        pair_spaces: tuple[PairSpaceProjector, ...] | None = None,
        threshold: float = 1e-10,
        metric_threshold: float = 1e-10,
        spherical: bool = False,
    ) -> NaturalAuxiliaryFunctions:
        """Build NAFs from AO-pair or selected orbital-pair three-centre integrals.

        Args:
            ao_basis: Basis set for the atomic orbitals.
            aux_basis: Auxiliary basis set to compress.
            pair_spaces: Optional channels defining the pair space. With
                ``None``, all AO pairs are used.
            threshold: Relative eigenvalue cutoff for the natural auxiliary
                covariance matrix.
            metric_threshold: Absolute cutoff for eigenvalues of the auxiliary
                Coulomb metric.
            spherical: Use real spherical instead of Cartesian functions.

        Returns:
            Coulomb-orthonormal auxiliary combinations adapted to ``pair_spaces``.
        """
        matrix11, matrix12 = _collect_integrals(ao_basis, aux_basis, spherical=spherical)
        factor = _decompose_potential(matrix11, threshold=metric_threshold)

        if pair_spaces is None:
            projected = np.einsum("PQ,Pij->Qij", factor, matrix12, optimize=True)
            covariance = np.einsum("Qij,Rij->QR", projected, projected, optimize=True)
        else:
            metric_rank = factor.shape[1]
            covariance = np.zeros((metric_rank, metric_rank), dtype=np.float64)
            for channel in pair_spaces:
                left, right = channel.left_coefficients, channel.right_coefficients
                channel_integrals = np.einsum("mi,Pmn,na->Pia", left, matrix12, right, optimize=True)
                projected = np.einsum("PQ,Pia->Qia", factor, channel_integrals, optimize=True)
                covariance += channel.weight * np.einsum("Qia,Ria->QR", projected, projected, optimize=True)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues.size:
            retained = eigenvalues > threshold * np.max(eigenvalues)
            natural_rotation = eigenvectors[:, retained]
        else:
            natural_rotation = eigenvectors
        coefficients = factor @ natural_rotation

        return cls(auxiliary_coefficients=coefficients)


def _collect_integrals(
    ao_basis: Basis,
    aux_basis: Basis,
    *,
    spherical: bool,
) -> tuple[array[float64], array[float64]]:
    """Collect the auxiliary metric and auxiliary-AO three-centre integrals."""
    num_aos = ao_basis.num_contracted_spherical_aos if spherical else ao_basis.num_contracted_cartesian_aos
    num_aux_aos = aux_basis.num_contracted_spherical_aos if spherical else aux_basis.num_contracted_cartesian_aos
    ao_slice = slice(0, num_aos)
    aux_slice = slice(num_aos, num_aos + num_aux_aos)
    basis = ao_basis.concatenate(aux_basis)

    coulomb = Coulomb()
    matrix11 = coulomb.collect(
        basis,
        (1, 1),
        ao_slices=(aux_slice, aux_slice),
        spherical=spherical,
    )
    matrix12 = coulomb.collect(
        basis,
        (1, 2),
        ao_slices=(aux_slice, ao_slice, ao_slice),
        spherical=spherical,
    )
    return matrix11, matrix12


def _decompose_potential(
    matrix: array[float64],
    *,
    threshold: float = 1e-10,
) -> array[float64]:
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


def density_fit(
    ao_basis: Basis,
    aux_basis: Basis,
    *,
    spherical: bool = False,
    nafs: NaturalAuxiliaryFunctions | None = None,
) -> array[float64]:
    """Prepare density fitting integrals.

    Args:
        ao_basis: Basis set.
        aux_basis: Auxiliary basis set.
        spherical: Return real spherical instead of Cartesian integrals.
        nafs: Optional pair-space-adapted auxiliary transform. Its dimensions
            and spherical convention must match ``aux_basis`` and this call.
            Accuracy is guaranteed only in the pair spaces used to build it.

    Returns:
        Density-fitting factors, with the first index corresponding to the
        metric-orthonormal auxiliary functions or supplied NAFs.
    """
    matrix11, matrix12 = _collect_integrals(ao_basis, aux_basis, spherical=spherical)
    if nafs is None:
        factor = _decompose_potential(matrix11)
    else:
        factor = nafs.auxiliary_coefficients
    matrix: array[float64] = np.einsum("PQ,Pij->Qij", factor, matrix12, optimize=True)
    return matrix
