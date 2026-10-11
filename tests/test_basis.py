"""Tests for basis construction and shell-pair data."""

from __future__ import annotations

import numpy as np
from pyscf import gto

from hermi.basis import Basis, Shells
from hermi.molecule import Element, Molecule
from hermi.operators import Overlap
from hermi.util import cartesian_normalization


def test_from_molecule_repeats_basis_data_for_each_atom() -> None:
    """Build a normalized hydrogen basis at both molecular centers."""
    molecule = Molecule(
        elements=[Element("H", 1, "Hydrogen", [1]), Element("H", 1, "Hydrogen", [1])],
        positions=np.asarray([[0, 0, 0], [0, 0, 1.4]], dtype=np.float64),
    )
    basis = Basis.from_molecule(molecule, "sto-3g")

    assert basis.num_primitive_shells == 6
    assert basis.num_contracted_shells == 2
    np.testing.assert_array_equal(basis.shells.shell_indices, [0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(basis.shells.position, np.repeat(molecule.positions, 3, axis=0))

    reference = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    np.testing.assert_allclose(
        Overlap().collect(basis),
        reference.intor("int1e_ovlp_cart"),
        rtol=1e-7,
        atol=1e-10,
    )


def test_from_molecule_handles_combined_angular_momenta() -> None:
    """Expand a BSE s/p shell into separate contracted shells."""
    molecule = Molecule(
        elements=[Element("C", 6, "Carbon", [2, 4])],
        positions=np.zeros((1, 3), dtype=np.float64),
    )
    basis = Basis.from_molecule(molecule, "sto-3g")

    assert basis.num_primitive_shells == 9
    assert basis.num_contracted_shells == 3
    assert basis.num_contracted_cartesian_aos == 5
    np.testing.assert_array_equal(
        basis.shells.angular_momentum[basis.contracted_shell_starts(slice(None))],
        [0, 0, 1],
    )

    reference = gto.M(atom="C 0 0 0", basis="sto-3g", unit="Bohr", cart=True, verbose=0)
    np.testing.assert_allclose(
        Overlap().collect(basis),
        reference.intor("int1e_ovlp_cart"),
        rtol=1e-7,
        atol=1e-10,
    )


def test_concatenate_rebases_shell_indices_and_rebuilds_pair_data() -> None:
    """Concatenate contracted shells and construct cross-basis primitive pairs."""
    left_shells = Shells(
        position=np.zeros((2, 3), dtype=np.float64),
        exponent=np.asarray([0.8, 0.25], dtype=np.float64),
        coefficient=np.asarray([0.4, 0.6], dtype=np.float64),
        angular_momentum=np.asarray([0, 0], dtype=np.int64),
        shell_indices=np.asarray([0, 0], dtype=np.int64),
    )
    right_shells = Shells(
        position=np.asarray([[1.1, 0, 0]], dtype=np.float64),
        exponent=np.asarray([0.7], dtype=np.float64),
        coefficient=np.asarray([1.0], dtype=np.float64),
        angular_momentum=np.asarray([1], dtype=np.int64),
        shell_indices=np.asarray([0], dtype=np.int64),
    )
    left = Basis.from_shells(left_shells).normalize()
    right = Basis.from_shells(right_shells).normalize()

    combined = left.concatenate(right)

    assert combined.num_primitive_shells == 3
    assert combined.num_contracted_shells == 2
    np.testing.assert_array_equal(combined.shells.shell_indices, [0, 0, 2])
    np.testing.assert_array_equal(
        combined.shells.position, np.concatenate((left.shells.position, right.shells.position))
    )
    np.testing.assert_array_equal(
        combined.shells.exponent, np.concatenate((left.shells.exponent, right.shells.exponent))
    )
    np.testing.assert_array_equal(
        combined.shells.coefficient, np.concatenate((left.shells.coefficient, right.shells.coefficient))
    )
    np.testing.assert_array_equal(
        combined.shells.angular_momentum,
        np.concatenate((left.shells.angular_momentum, right.shells.angular_momentum)),
    )

    pair = combined.shell_pairs
    exponent_product = left.shells.exponent[0] * right.shells.exponent[0]
    exponent_sum = left.shells.exponent[0] + right.shells.exponent[0]
    reduced_exponent = exponent_product / exponent_sum
    vector = left.shells.position[0] - right.shells.position[0]
    position = (
        left.shells.exponent[0] * left.shells.position[0] + right.shells.exponent[0] * right.shells.position[0]
    ) / exponent_sum

    assert pair.position.shape == (3, 3, 3)
    np.testing.assert_allclose(pair.vector[0, 2], vector)
    np.testing.assert_allclose(pair.position[0, 2], position)
    np.testing.assert_allclose(pair.exponent[0, 2], exponent_sum)
    np.testing.assert_allclose(pair.reduced_exponent[0, 2], reduced_exponent)
    np.testing.assert_allclose(pair.coefficient[0, 2], left.shells.coefficient[0] * right.shells.coefficient[0])
    np.testing.assert_allclose(pair.attenuation[0, 2], np.exp(-reduced_exponent * vector**2))
    np.testing.assert_array_equal(pair.angular_momentum[0, 2], 1)


def test_concatenated_basis_integrals_preserve_original_blocks() -> None:
    """Check that concatenating bases preserves each original overlap block."""
    left_shells = Shells(
        position=np.zeros((1, 3), dtype=np.float64),
        exponent=np.asarray([0.8], dtype=np.float64),
        coefficient=np.asarray([1.0], dtype=np.float64),
        angular_momentum=np.asarray([0], dtype=np.int64),
        shell_indices=np.asarray([0], dtype=np.int64),
    )
    right_shells = Shells(
        position=np.asarray([[1.1, 0, 0]], dtype=np.float64),
        exponent=np.asarray([0.7], dtype=np.float64),
        coefficient=np.asarray([1.0], dtype=np.float64),
        angular_momentum=np.asarray([1], dtype=np.int64),
        shell_indices=np.asarray([0], dtype=np.int64),
    )
    left = Basis.from_shells(left_shells).normalize()
    right = Basis.from_shells(right_shells).normalize()

    combined_overlap = Overlap().collect(left.concatenate(right))
    left_overlap = Overlap().collect(left)
    right_overlap = Overlap().collect(right)

    np.testing.assert_allclose(combined_overlap[:1, :1], left_overlap)
    np.testing.assert_allclose(combined_overlap[1:, 1:], right_overlap)
    assert np.any(combined_overlap[:1, 1:] != 0)


def test_angular_momentum_order_keeps_unique_pair_blocks_ordered(basis_factory) -> None:
    """Check that unique shell-pair blocks have a stable angular-momentum order."""
    basis = basis_factory(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [1, 1, 1],
        [2, 0, 1],
    )
    ordered_basis, _ = basis.ordered_by_angular_momentum()
    pairs = list(ordered_basis.iter_shell_pair_groups(unique=True))

    pair_orders = [(pair.shell_i.angular_momentum, pair.shell_j.angular_momentum) for pair in pairs]
    assert pair_orders == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    pair_momenta = [pair.angular_momentum for pair in pairs]
    assert pair_momenta == sorted(pair_momenta)
    for index, pair_ab in enumerate(pairs):
        for pair_cd in pairs[index:]:
            assert pair_ab.angular_momentum <= pair_cd.angular_momentum


def test_shell_pair_product_data(basis_factory) -> None:
    """Check product center, exponents, and attenuation for one pair."""
    basis = basis_factory([[0, 0, 0], [2, 0, 0]], [1, 2], [0, 0])
    pair = basis[0:1, 1:2]

    np.testing.assert_allclose(pair.exponent, [[3]])
    np.testing.assert_allclose(pair.reduced_exponent, [[2 / 3]])
    np.testing.assert_allclose(pair.position, [[[4 / 3, 0, 0]]])
    np.testing.assert_allclose(pair.attenuation, [[[np.exp(-8 / 3), 1, 1]]])


def test_cartesian_normalization_for_d_functions() -> None:
    """Check component-dependent Cartesian normalization factors."""
    np.testing.assert_allclose(
        cartesian_normalization(2),
        [1 / np.sqrt(3), 1, 1, 1 / np.sqrt(3), 1, 1 / np.sqrt(3)],
    )


def test_radial_normalization_for_contracted_shell() -> None:
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
