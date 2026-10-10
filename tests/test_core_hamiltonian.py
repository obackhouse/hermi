"""Integration tests for the core Hamiltonian operator."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.operators import CoreHamiltonian, Kinetic, Nuclear


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_core_hamiltonian_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare core Hamiltonian integrals against PySCF for each example system."""
    actual = CoreHamiltonian(example_system.molecule).collect(example_system.basis, spherical=spherical)
    kinetic = normalized_reference(example_system, f"int1e_kin_{suffix}", suffix)
    nuclear = normalized_reference(example_system, f"int1e_nuc_{suffix}", suffix)
    reference = kinetic + nuclear

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("spherical", (False, True))
def test_core_hamiltonian_matches_constituent_operators(example_system, spherical: bool) -> None:
    """Compare core Hamiltonian integrals with the sum of its operators."""
    actual = CoreHamiltonian(example_system.molecule).collect(example_system.basis, spherical=spherical)
    kinetic = Kinetic().collect(example_system.basis, signature=(2, 0), spherical=spherical)
    nuclear = Nuclear(example_system.molecule).collect(example_system.basis, spherical=spherical)

    assert actual.shape == kinetic.shape == nuclear.shape
    np.testing.assert_allclose(actual, kinetic + nuclear, rtol=1e-12, atol=1e-12)
