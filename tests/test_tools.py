"""Tests for algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, gto

from hermi.algorithms.density_fitting import density_fit
from hermi.systems.basis import Basis


@pytest.mark.parametrize("spherical", (False, True))
def test_density_fit_against_pyscf(example_system, spherical: bool) -> None:
    """Compare density-fitted integrals against PySCF."""
    auxiliary_basis_name = "def2-universal-jfit"
    auxiliary_basis = Basis.from_molecule(example_system.molecule, auxiliary_basis_name)
    factor = density_fit(example_system.basis, auxiliary_basis, spherical=spherical)
    actual = np.einsum("Qij,Qkl->ijkl", factor, factor, optimize=True)

    reference = gto.M(
        atom=example_system.reference.atom,
        basis=example_system.reference.basis,
        unit="Bohr",
        cart=not spherical,
        verbose=0,
    )
    reference_factor = df.incore.cholesky_eri(
        reference,
        auxbasis=auxiliary_basis_name,
        aosym="s1",
        decompose_j2c="eig",
        lindep=1e-10,
    )
    num_aos = reference.nao_nr()
    reference_factor = reference_factor.reshape(-1, num_aos, num_aos)
    overlap = reference.intor("int1e_ovlp")
    scale = 1 / np.sqrt(np.diag(overlap))
    reference_factor *= scale[None, :, None] * scale[None, None, :]
    expected = np.einsum("Qij,Qkl->ijkl", reference_factor, reference_factor, optimize=True)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
