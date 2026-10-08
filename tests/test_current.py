"""Temporary smoke tests for the current shell and Hermite code."""

from __future__ import annotations

import unittest

import numpy as np
from pyscf import gto
from scipy.special import erf

from hermi.basis import Basis, Shells
from hermi.boys import boys
from hermi.coulomb import coulomb_integrals_22, coulomb_operator_hermite_22
from hermi.hermite import expansion_pair_coefficients, expansion_pair_matrix
from hermi.kinetic import kinetic_integrals_22
from hermi.overlap import overlap_integrals_22
from hermi.util import cartesian_normalization


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
    return Basis.from_shells(shells).normalize()


class ShellAndHermiteTests(unittest.TestCase):
    """Smoke tests for pair construction and Hermite calculations."""

    def test_angular_momentum_order_keeps_unique_pair_blocks_ordered(self) -> None:
        """Check unique pair blocks have a stable angular-momentum order."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [1, 1, 1],
            [2, 0, 1],
        )
        ordered_basis, _ = basis.ordered_by_angular_momentum()
        pairs = list(ordered_basis.iter_shell_pair_groups(unique=True))

        pair_orders = [(pair.shell_i.angular_momentum, pair.shell_j.angular_momentum) for pair in pairs]
        self.assertEqual(pair_orders, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])

        pair_momenta = [pair.angular_momentum for pair in pairs]
        self.assertEqual(pair_momenta, sorted(pair_momenta))
        for index, pair_ab in enumerate(pairs):
            for pair_cd in pairs[index:]:
                self.assertLessEqual(pair_ab.angular_momentum, pair_cd.angular_momentum)

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
        coefficients = expansion_pair_coefficients(pair, axis=0)
        attenuation = np.exp(-8 / 3)

        self.assertEqual(coefficients.shape, (1, 1, 2, 2, 1))
        np.testing.assert_allclose(coefficients[0, 0, 0, 0, 0], attenuation)
        np.testing.assert_allclose(coefficients[0, 0, 0, 1, 0], 4 / 3 * attenuation)
        np.testing.assert_allclose(coefficients[0, 0, 1, 1, 0], attenuation / 6)

    def test_expansion_matrix_for_p_s_pair(self) -> None:
        """Check the Cartesian-to-Hermite gather for a p-s pair."""
        basis = make_basis([[0, 0, 0], [2, 0, 0]], [1, 2], [1, 0])
        pair = basis[0:1, 1:2]
        matrix = expansion_pair_matrix(pair)
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
        shell = Basis.from_shells(shells).normalize()[0:2]

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

    def test_overlap_integrals_against_pyscf(self) -> None:
        """Compare Cartesian and spherical overlap matrices against PySCF."""
        basis = make_basis(
            [[0, 0, 0], [1.3, 0, 0], [0, 1.1, 0]],
            [0.8, 1.1, 0.9],
            [1, 0, 1],
        )
        molecule = gto.M(
            atom=[("H", (0, 0, 0)), ("He", (1.3, 0, 0)), ("Li", (0, 1.1, 0))],
            basis={
                "H": [[1, [0.8, 1.0]]],
                "He": [[0, [1.1, 1.0]]],
                "Li": [[1, [0.9, 1.0]]],
            },
            unit="Bohr",
            cart=True,
            verbose=0,
        )

        cartesian = overlap_integrals_22(basis)
        cartesian_reference = molecule.intor("int1e_ovlp_cart")
        self.assertEqual(cartesian.shape, cartesian_reference.shape)
        np.testing.assert_allclose(cartesian, cartesian_reference, rtol=1e-10, atol=1e-12)

        spherical = overlap_integrals_22(basis, spherical=True)
        spherical_reference = molecule.intor("int1e_ovlp_sph")
        self.assertEqual(spherical.shape, spherical_reference.shape)
        np.testing.assert_allclose(spherical, spherical_reference, rtol=1e-10, atol=1e-12)

    def test_contracted_kinetic_integrals_against_pyscf(self) -> None:
        """Compare contracted Cartesian and spherical kinetic matrices against PySCF."""
        shells = Shells(
            position=np.asarray(
                [[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0], [0, 1.1, 0]],
                dtype=np.float64,
            ),
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

        for spherical, integral_name in ((False, "int1e_kin_cart"), (True, "int1e_kin_sph")):
            with self.subTest(spherical=spherical):
                kinetic = kinetic_integrals_22(basis, spherical=spherical)
                reference = molecule.intor(integral_name)
                overlap_name = "int1e_ovlp_sph" if spherical else "int1e_ovlp_cart"
                reference_overlap = molecule.intor(overlap_name)
                reference_scale = 1 / np.sqrt(np.diag(reference_overlap))
                reference *= reference_scale[:, None] * reference_scale[None, :]
                self.assertEqual(kinetic.shape, reference.shape)
                np.testing.assert_allclose(kinetic, reference, rtol=1e-10, atol=1e-12)

    def test_ssss_coulomb_operator_shape_and_value(self) -> None:
        """Check the ssss Hermite operator matrix against the Boys value."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        )
        pair_ab = basis[0:1, 1:2]
        pair_cd = basis[2:3, 3:4]

        matrix, signs = coulomb_operator_hermite_22(pair_ab, pair_cd)

        rho = pair_ab.exponent[0, 0] * pair_cd.exponent[0, 0] / (pair_ab.exponent[0, 0] + pair_cd.exponent[0, 0])
        argument = rho * np.sum((pair_ab.position[0, 0] - pair_cd.position[0, 0]) ** 2)
        expected = boys(np.asarray(argument), max_order=0)[0]
        self.assertEqual(matrix.shape, (1, 1, 1, 1))
        np.testing.assert_allclose(matrix[0, 0, 0, 0], expected)
        np.testing.assert_array_equal(signs, [1.0])

    def test_coulomb_operator_pair_indices_and_ket_sign(self) -> None:
        """Check a first-order Hermite element and the ket sign convention."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
        )
        pair_ab = basis[0:1, 1:2]
        pair_cd = basis[2:3, 3:4]
        matrix, signs = coulomb_operator_hermite_22(pair_ab, pair_cd)

        argument = 0.75
        first_order = boys(np.asarray(argument), max_order=1)[1]
        self.assertEqual(matrix.shape, (1, 1, 4, 4))
        np.testing.assert_array_equal(signs, [1.0, -1.0, -1.0, -1.0])
        signed_matrix = matrix * signs[None, None, None, :]
        np.testing.assert_allclose(signed_matrix[0, 0, 3, 0], -first_order)
        np.testing.assert_allclose(signed_matrix[0, 0, 0, 3], first_order)

    def test_coulomb_integrals_against_pyscf(self) -> None:
        """Compare the full Cartesian Coulomb integral tensor against PySCF."""
        basis = make_basis(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        )
        eri = coulomb_integrals_22(basis)

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

    def test_contracted_coulomb_integrals_against_pyscf(self) -> None:
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
        basis = Basis.from_shells(shells).normalize()
        eri = coulomb_integrals_22(basis)

        atoms = [("H", (0, 0, 0)), ("Li", (1.3, 0, 0))]
        pyscf_basis = {
            "H": [[1, [0.8, 0.35], [0.2, 0.65]]],
            "Li": [[0, [1.1, 0.6], [0.3, 0.4]]],
        }
        molecule = gto.M(atom=atoms, basis=pyscf_basis, unit="Bohr", cart=True, verbose=0)
        reference = molecule.intor("int2e_cart", aosym="s1")
        self.assertEqual(eri.shape, reference.shape)
        np.testing.assert_allclose(eri, reference, rtol=1e-10, atol=1e-12)

    def test_spherical_coulomb_integrals_against_pyscf(self) -> None:
        """Compare mixed spherical shells against PySCF."""
        basis = make_basis(
            [[0, 0, 0], [0, 0, 0], [1.3, 0, 0]],
            [1.1, 0.8, 0.9],
            [0, 2, 1],
        )
        eri = coulomb_integrals_22(basis, spherical=True)

        pyscf_basis = {
            "H": [[0, [1.1, 1.0]], [2, [0.8, 1.0]]],
            "Li": [[1, [0.9, 1.0]]],
        }
        molecule = gto.M(
            atom=[("H", (0, 0, 0)), ("Li", (1.3, 0, 0))],
            basis=pyscf_basis,
            unit="Bohr",
            cart=False,
            verbose=0,
        )
        reference = molecule.intor("int2e", aosym="s1")
        self.assertEqual(eri.shape, (9, 9, 9, 9))
        self.assertEqual(eri.shape, reference.shape)
        np.testing.assert_allclose(eri, reference, rtol=1e-10, atol=1e-12)

    def test_contracted_spherical_coulomb_integrals_against_pyscf(self) -> None:
        """Compare contracted spherical shells against PySCF."""
        shells = Shells(
            position=np.asarray(
                [[0, 0, 0], [0, 0, 0], [1.3, 0, 0], [1.3, 0, 0]],
                dtype=np.float64,
            ),
            exponent=np.asarray([0.8, 0.2, 1.1, 0.3], dtype=np.float64),
            coefficient=np.asarray([0.35, 0.65, 0.6, 0.4], dtype=np.float64),
            angular_momentum=np.asarray([2, 2, 0, 0], dtype=np.int64),
            shell_indices=np.asarray([0, 0, 2, 2], dtype=np.int64),
        )
        eri = coulomb_integrals_22(Basis.from_shells(shells).normalize(), spherical=True)

        pyscf_basis = {
            "H": [[2, [0.8, 0.35], [0.2, 0.65]]],
            "Li": [[0, [1.1, 0.6], [0.3, 0.4]]],
        }
        molecule = gto.M(
            atom=[("H", (0, 0, 0)), ("Li", (1.3, 0, 0))],
            basis=pyscf_basis,
            unit="Bohr",
            cart=False,
            verbose=0,
        )
        reference = molecule.intor("int2e", aosym="s1")
        self.assertEqual(eri.shape, (6, 6, 6, 6))
        self.assertEqual(eri.shape, reference.shape)
        np.testing.assert_allclose(eri, reference, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
