"""Tests for the Coulomb operator."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto

from hermi.basis import Basis, Shells
from hermi.boys import boys
from hermi.operators import Coulomb


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_coulomb_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare electron-repulsion integrals against PySCF for each example system."""
    actual = Coulomb().collect(example_system.basis, spherical=spherical)
    reference = normalized_reference(example_system, f"int2e_{suffix}", suffix, aosym="s1")

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)


def test_ssss_operator_value(basis_factory) -> None:
    """Check the ssss Hermite operator matrix against the Boys value."""
    basis = basis_factory(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    )
    pair_ab = basis[0:1, 1:2]
    pair_cd = basis[2:3, 3:4]

    matrix = Coulomb().get_operator_matrix((pair_ab, pair_cd))

    exponent_product = pair_ab.exponent[0, 0] * pair_cd.exponent[0, 0]
    exponent_sum = pair_ab.exponent[0, 0] + pair_cd.exponent[0, 0]
    rho = exponent_product / exponent_sum
    argument = rho * np.sum((pair_ab.position[0, 0] - pair_cd.position[0, 0]) ** 2)
    prefactor = 2 * np.pi**2.5 / (exponent_product * np.sqrt(exponent_sum))
    expected = prefactor * boys(np.asarray(argument), max_order=0)[0]

    assert matrix.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(matrix[0, 0, 0, 0], expected)


def test_first_order_operator_indices_and_ket_sign(basis_factory) -> None:
    """Check first-order Hermite elements and the ket sign convention."""
    basis = basis_factory(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [1, 1, 1, 1],
        [1, 0, 1, 0],
    )
    pair_ab = basis[0:1, 1:2]
    pair_cd = basis[2:3, 3:4]
    matrix = Coulomb().get_operator_matrix((pair_ab, pair_cd))

    argument = 0.75
    first_order = boys(np.asarray(argument), max_order=1)[1]
    exponent_product = pair_ab.exponent[0, 0] * pair_cd.exponent[0, 0]
    exponent_sum = pair_ab.exponent[0, 0] + pair_cd.exponent[0, 0]
    prefactor = 2 * np.pi**2.5 / (exponent_product * np.sqrt(exponent_sum))

    assert matrix.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(matrix[0, 0, 3, 0], -prefactor * first_order)
    np.testing.assert_allclose(matrix[0, 0, 0, 3], prefactor * first_order)


def test_contracted_p_shell_coulomb_integrals_against_pyscf() -> None:
    """Compare multi-primitive contracted Cartesian Coulomb integrals against PySCF."""
    shells = Shells(
        position=np.asarray([[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0]], dtype=np.float64),
        exponent=np.asarray([0.8, 0.2, 1.1, 0.3], dtype=np.float64),
        coefficient=np.asarray([0.35, 0.65, 0.6, 0.4], dtype=np.float64),
        angular_momentum=np.asarray([1, 1, 0, 0], dtype=np.int64),
        shell_indices=np.asarray([0, 0, 2, 2], dtype=np.int64),
    )
    basis = Basis.from_shells(shells).normalize()
    molecule = gto.M(
        atom=[("H", (0, 0, 0)), ("Li", (1.3, 0, 0))],
        basis={"H": [[1, [0.8, 0.35], [0.2, 0.65]]], "Li": [[0, [1.1, 0.6], [0.3, 0.4]]]},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    actual = Coulomb().collect(basis)
    reference = molecule.intor("int2e_cart", aosym="s1")

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)


def test_contracted_d_shell_coulomb_integrals_against_pyscf() -> None:
    """Compare contracted spherical d-shell Coulomb integrals against PySCF."""
    shells = Shells(
        position=np.asarray([[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0]], dtype=np.float64),
        exponent=np.asarray([0.8, 0.2, 1.1, 0.3], dtype=np.float64),
        coefficient=np.asarray([0.35, 0.65, 0.6, 0.4], dtype=np.float64),
        angular_momentum=np.asarray([2, 2, 0, 0], dtype=np.int64),
        shell_indices=np.asarray([0, 0, 2, 2], dtype=np.int64),
    )
    basis = Basis.from_shells(shells).normalize()
    molecule = gto.M(
        atom=[("H", (0, 0, 0)), ("Li", (1.3, 0, 0))],
        basis={"H": [[2, [0.8, 0.35], [0.2, 0.65]]], "Li": [[0, [1.1, 0.6], [0.3, 0.4]]]},
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    actual = Coulomb().collect(basis, spherical=True)
    reference = molecule.intor("int2e", aosym="s1")

    assert actual.shape == (6, 6, 6, 6)
    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)
