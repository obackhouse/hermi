"""Integration tests for the kinetic energy operator."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto

from hermi.systems.basis import Basis, Shells
from hermi.operators.kinetic import Kinetic


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_kinetic_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare kinetic integrals against PySCF for each example system."""
    actual = Kinetic().collect(example_system.basis, spherical=spherical)
    reference = normalized_reference(example_system, f"int1e_kin_{suffix}", suffix)

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_kinetic_signature_20_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare (2, 0) kinetic integrals against PySCF for each example system."""
    actual = Kinetic().collect(example_system.basis, signature=(2, 0), spherical=spherical)
    reference = normalized_reference(example_system, f"int1e_kin_{suffix}", suffix)

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("spherical", (False, True))
def test_kinetic_20_ao_slices_match_full_array(example_system, spherical: bool) -> None:
    """Compare sliced (2, 0) kinetic integrals with slices of the full matrix."""
    operator = Kinetic()
    full = operator.collect(example_system.basis, signature=(2, 0), spherical=spherical)
    num_aos = full.shape[0]
    ao_slices = (slice(0, num_aos, 2), slice(num_aos - 1, None, -2))

    actual = operator.collect(example_system.basis, signature=(2, 0), ao_slices=ao_slices, spherical=spherical)
    indices = tuple(np.arange(*ao_slice.indices(num_aos)) for ao_slice in ao_slices)
    expected = full[np.ix_(*indices)]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_contracted_p_and_d_shells_against_pyscf() -> None:
    """Compare contracted Cartesian and spherical kinetic matrices against PySCF."""
    shells = Shells(
        position=np.asarray([[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0], [0, 1.1, 0]], dtype=np.float64),
        exponent=np.asarray([0.8, 0.2, 1.1, 0.3, 0.9], dtype=np.float64),
        coefficient=np.asarray([0.35, 0.65, 0.6, 0.4, 1.0], dtype=np.float64),
        angular_momentum=np.asarray([1, 1, 0, 0, 2], dtype=np.int64),
        shell_indices=np.asarray([0, 0, 2, 2, 4], dtype=np.int64),
    )
    basis = Basis.from_shells(shells).normalize()
    molecule = gto.M(
        atom=[("H", (0, 0, 0)), ("He", (1.3, 0, 0)), ("Li", (0, 1.1, 0))],
        basis={
            "H": [[1, [0.8, 0.35], [0.2, 0.65]]],
            "He": [[0, [1.1, 0.6], [0.3, 0.4]]],
            "Li": [[2, [0.9, 1.0]]],
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )

    for spherical, suffix in ((False, "cart"), (True, "sph")):
        actual = Kinetic().collect(basis, spherical=spherical)
        reference = molecule.intor(f"int1e_kin_{suffix}")
        overlap = molecule.intor(f"int1e_ovlp_{suffix}")
        scale = 1 / np.sqrt(np.diag(overlap))
        reference *= scale[:, None] * scale[None, :]

        assert actual.shape == reference.shape
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)
