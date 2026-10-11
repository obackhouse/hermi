"""Tests for algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import df, gto

from hermi.algorithms.density_fitting import density_fit
from hermi.algorithms.tensor_hypercontraction import tensor_hypercontraction
from hermi.systems.basis import Basis
from hermi.systems.grid import MolecularGrid
from hermi.systems.molecule import Molecule


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


@pytest.mark.parametrize("spherical", (False, True))
def test_tensor_hypercontraction_against_pyscf(spherical: bool) -> None:
    """Compare THC integrals against raw PySCF electron-repulsion integrals."""
    molecule = Molecule.from_list(["H", "H"], [[0, 0, 0], [0, 0, 1.4]], units="bohr")
    ao_basis = Basis.from_molecule(molecule, "sto-3g")
    aux_basis = Basis.from_molecule(molecule, "def2-universal-jfit")
    grid = MolecularGrid.from_molecule(molecule, radial_order=16, angular_order=6)

    collocation, kernel = tensor_hypercontraction(ao_basis, aux_basis, grid, spherical=spherical)
    actual = np.einsum(
        "Ki,Kj,KL,Lk,Ll->ijkl", collocation, collocation, kernel, collocation, collocation, optimize=True
    )

    reference_molecule = gto.M(
        atom=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
        basis="sto-3g",
        unit="Bohr",
        cart=not spherical,
        verbose=0,
    )
    suffix = "sph" if spherical else "cart"
    expected = reference_molecule.intor(f"int2e_{suffix}", aosym="s1")

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
