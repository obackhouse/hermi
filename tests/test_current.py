"""Temporary smoke tests for the current shell and Hermite code."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf import gto
from scipy.special import erf

from pymmd.boys import boys
from pymmd.coulomb import auxiliary_coulomb_matrix, auxiliary_coulomb_terms
from pymmd.eri import electron_repulsion_integral, electron_repulsion_integrals
from pymmd.hermite import expansion_coefficients, expansion_matrix
from pymmd.basis import Basis, Shells
from pymmd.util import cartesian_normalization


def make_basis(
    positions: list[list[float]],
    exponents: list[float],
    angular_momenta: list[int],
) -> Basis:
    """Build a basis with one contracted shell per primitive."""
    count = len(exponents)
    shells = Shells(
        position=np.asarray(positions, dtype=np.float64),
        exponent=np.asarray(exponents, dtype=np.float64),
        coefficient=np.ones(count, dtype=np.float64),
        angular_momentum=np.asarray(angular_momenta, dtype=np.int64),
        shell_indices=np.arange(count, dtype=np.int64),
    )
    return Basis.from_shells(shells)


class ShellAndHermiteTests(unittest.TestCase):
    """Smoke tests for pair construction and Hermite calculations."""

    def test_shell_pair_product_data(self) -> None:
        """Check product center, exponents, and attenuation for one pair."""
        basis = make_basis([[0, 0, 0], [2, 0, 0]], [1, 2], [0, 0])
        pair = basis[0:1, 1:2]

        np.testing.assert_allclose(pair.exponent, [[3]])
        np.testing.assert_allclose(pair.reduced_exponent, [[2 / 3]])
        np.testing.assert_allclose(pair.position, [[[4 / 3, 0, 0]]])
        np.testing.assert_allclose(pair.attenuation, [[[np.exp(-8 / 3), 1, 1]]])

    def test_expansion_coefficients_for_p_s_pair(self) -> None:
        """Check the base value and first p-shell recurrence values."""
        basis = make_basis([[0, 0, 0], [2, 0, 0]], [1, 2], [1, 0])
        pair = basis[0:1, 1:2]
        coefficients = expansion_coefficients(pair, axis=0)
        attenuation = np.exp(-8 / 3)

        self.assertEqual(coefficients.shape, (1, 1, 2, 2, 1))
        np.testing.assert_allclose(coefficients[0, 0, 0, 0, 0], attenuation)
        np.testing.assert_allclose(coefficients[0, 0, 0, 1, 0], 4 / 3 * attenuation)
        np.testing.assert_allclose(coefficients[0, 0, 1, 1, 0], attenuation / 6)

    def test_expansion_matrix_for_p_s_pair(self) -> None:
        """Check the Cartesian-to-Hermite gather for a p-s pair."""
        basis = make_basis([[0, 0, 0], [2, 0, 0]], [1, 2], [1, 0])
        pair = basis[0:1, 1:2]
        matrix = expansion_matrix(pair)
        attenuation = np.exp(-8 / 3)

        self.assertEqual(matrix.shape, (1, 3, 4))
        np.testing.assert_allclose(matrix[0, 0, 0], 4 / 3 * attenuation)
        np.testing.assert_allclose(matrix[0, 0, 3], attenuation / 6)

    def test_cartesian_normalization_for_d_functions(self) -> None:
        """Check component-dependent Cartesian normalization factors."""
        np.testing.assert_allclose(
            cartesian_normalization(2),
            [1 / np.sqrt(3), 1, 1, 1 / np.sqrt(3), 1, 1 / np.sqrt(3)],
        )

    def test_radial_normalization_for_contracted_shell(self) -> None:
        """Check that a multi-primitive s contraction has unit norm."""
        exponents = np.asarray([0.5, 2.0])
        shells = Shells(
            position=np.zeros((2, 3)),
            exponent=exponents,
            coefficient=np.asarray([0.7, 0.3]),
            angular_momentum=np.asarray([0, 0]),
            shell_indices=np.asarray([0, 0]),
        )
        shell = Basis.from_shells(shells)[0:2]

        radial_factors = (2 * exponents / np.pi) ** 0.75
        normalized_primitive_coefficients = shell.coefficient / radial_factors
        overlap = (2 * np.sqrt(np.outer(exponents, exponents)) / np.add.outer(exponents, exponents)) ** 1.5
        norm_squared = np.sum(np.outer(normalized_primitive_coefficients, normalized_primitive_coefficients) * overlap)
        np.testing.assert_allclose(norm_squared, 1.0)

    def test_boys_values_at_zero_and_positive_argument(self) -> None:
        """Check the zero limits and erf-derived zeroth order."""
        arguments = np.asarray([0.0, 2.0])
        values = boys(arguments, max_order=1)

        self.assertEqual(values.shape, (2, 2))
        np.testing.assert_allclose(values[0], [1.0, 1 / 3])
        expected_f0 = 0.5 * np.sqrt(np.pi / 2) * erf(np.sqrt(2))
        np.testing.assert_allclose(values[1, 0], expected_f0)
        np.testing.assert_allclose(values[1, 1], (expected_f0 - np.exp(-2)) / 4)

    def test_ssss_auxiliary_term_shape_and_value(self) -> None:
        """Check one ssss quartet against its sole Hermite auxiliary term."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        )
        pair_ab = basis[0:1, 1:2]
        pair_cd = basis[2:3, 3:4]

        terms = auxiliary_coulomb_terms(pair_ab, pair_cd)
        matrix = auxiliary_coulomb_matrix(pair_ab, pair_cd)
        eri = electron_repulsion_integral(pair_ab, pair_cd)

        rho = pair_ab.exponent[0, 0] * pair_cd.exponent[0, 0] / (pair_ab.exponent[0, 0] + pair_cd.exponent[0, 0])
        argument = rho * np.sum((pair_ab.position[0, 0] - pair_cd.position[0, 0]) ** 2)
        expected = boys(np.asarray(argument), max_order=0)[0]
        self.assertEqual(terms.shape, (1, 1, 1, 1))
        self.assertEqual(matrix.shape, (1, 1, 1, 1))
        self.assertEqual(eri.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(terms[0, 0, 0, 0], expected)
        np.testing.assert_allclose(matrix[0, 0, 0, 0], expected)
        prefactor = 2 * np.pi**2.5 / (2 * 2 * np.sqrt(2 + 2))
        attenuation = np.prod(pair_ab.attenuation[0, 0]) * np.prod(pair_cd.attenuation[0, 0])
        radial_normalization = (2 / np.pi) ** 0.75
        np.testing.assert_allclose(eri[0, 0, 0, 0], prefactor * attenuation * expected * radial_normalization**4)

    def test_coulomb_matrix_pair_indices_and_ket_sign(self) -> None:
        """Check a first-order Hermite element and the ket sign convention."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
        )
        pair_ab = basis[0:1, 1:2]
        pair_cd = basis[2:3, 3:4]
        matrix = auxiliary_coulomb_matrix(pair_ab, pair_cd)

        argument = 0.75
        first_order = boys(np.asarray(argument), max_order=1)[1]
        self.assertEqual(matrix.shape, (1, 1, 4, 4))
        np.testing.assert_allclose(matrix[0, 0, 3, 0], -first_order)
        np.testing.assert_allclose(matrix[0, 0, 0, 3], first_order)

    def test_electron_repulsion_integrals_against_pyscf(self) -> None:
        """Compare the full Cartesian ERI tensor against PySCF."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        )
        eri = electron_repulsion_integrals(basis)

        atoms = [
            ("H", (0, 0, 0)),
            ("He", (1, 0, 0)),
            ("Li", (0, 1, 0)),
            ("Be", (0, 0, 1)),
        ]
        pyscf_basis = {
            "H": [[1, [1.0, 1.0]]],
            "He": [[0, [1.0, 1.0]]],
            "Li": [[0, [1.0, 1.0]]],
            "Be": [[0, [1.0, 1.0]]],
        }
        molecule = gto.M(atom=atoms, basis=pyscf_basis, unit="Bohr", cart=True, verbose=0)
        reference = molecule.intor("int2e_cart", aosym="s1")
        self.assertEqual(eri.shape, reference.shape)
        np.testing.assert_allclose(eri, reference, rtol=1e-10, atol=1e-12)

    def test_contracted_electron_repulsion_integrals_against_pyscf(self) -> None:
        """Compare multi-primitive contracted shells against PySCF."""
        shells = Shells(
            position=np.asarray(
                [[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0]],
                dtype=np.float64,
            ),
            exponent=np.asarray([0.8, 0.2, 1.1, 0.3], dtype=np.float64),
            coefficient=np.asarray([0.35, 0.65, 0.6, 0.4], dtype=np.float64),
            angular_momentum=np.asarray([1, 1, 0, 0], dtype=np.int64),
            shell_indices=np.asarray([0, 0, 2, 2], dtype=np.int64),
        )
        basis = Basis.from_shells(shells)
        eri = electron_repulsion_integrals(basis)

        atoms = [("H", (0, 0, 0)), ("Li", (1.3, 0, 0))]
        pyscf_basis = {
            "H": [[1, [0.8, 0.35], [0.2, 0.65]]],
            "Li": [[0, [1.1, 0.6], [0.3, 0.4]]],
        }
        molecule = gto.M(atom=atoms, basis=pyscf_basis, unit="Bohr", cart=True, verbose=0)
        reference = molecule.intor("int2e_cart", aosym="s1")
        self.assertEqual(eri.shape, reference.shape)
        np.testing.assert_allclose(eri, reference, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
