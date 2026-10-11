"""Shared molecular examples for operator integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from pyscf import gto

from hermi.basis import Basis, Shells
from hermi.molecule import Molecule


def _make_basis(
    positions: list[list[float]],
    exponents: list[float],
    angular_momenta: list[int],
) -> Basis:
    """Build a normalized basis with one contracted shell per primitive."""
    count = len(exponents)
    shells = Shells(
        position=np.asarray(positions, dtype=np.float64),
        exponent=np.asarray(exponents, dtype=np.float64),
        coefficient=np.ones(count, dtype=np.float64),
        angular_momentum=np.asarray(angular_momenta, dtype=np.int64),
        shell_indices=np.arange(count, dtype=np.int64),
    )
    return Basis.from_shells(shells).normalize()


@dataclass(frozen=True)
class ExampleSystem:
    """Internal molecular data and an independent PySCF reference molecule."""

    name: str
    molecule: Molecule
    basis: Basis
    reference: gto.Mole


@dataclass(frozen=True)
class ExampleSystemInput:
    """Input data used to build an example system independently in Hermi and PySCF."""

    name: str
    xyz: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    basis_set: str | None
    custom_shells: tuple[tuple[int, float], ...] | None = None


EXAMPLE_SYSTEMS = (
    ExampleSystemInput(
        name="h2_sto3g",
        xyz="2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\n",
        atoms=(("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))),
        basis_set="sto-3g",
    ),
    ExampleSystemInput(
        name="water_sto3g",
        xyz="3\nH2O\nO 0.0 0.0 0.0\nH 0.0 -1.43233673 1.10715266\nH 0.0 1.43233673 1.10715266\n",
        atoms=(
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, -1.43233673, 1.10715266)),
            ("H", (0.0, 1.43233673, 1.10715266)),
        ),
        basis_set="sto-3g",
    ),
    ExampleSystemInput(
        name="carbon_spd",
        xyz="1\nC\nC 0.0 0.0 0.0\n",
        atoms=(("C", (0.0, 0.0, 0.0)),),
        basis_set=None,
        custom_shells=((0, 1.5), (1, 0.8), (2, 0.5)),
    ),
)


@pytest.fixture
def basis_factory():
    """Provide a small basis constructor for unit tests."""
    return _make_basis


@pytest.fixture
def normalized_reference():
    """Provide a PySCF integral normalized to the internal basis convention."""
    def reference(example_system, integral: str, suffix: str, **kwargs):
        values = example_system.reference.intor(integral, **kwargs)
        overlap = example_system.reference.intor(f"int1e_ovlp_{suffix}")
        scale = 1 / np.sqrt(np.diag(overlap))
        for axis in range(values.ndim):
            shape = (1,) * axis + (scale.size,) + (1,) * (values.ndim - axis - 1)
            values *= scale.reshape(shape)
        return values

    return reference


@pytest.fixture(params=EXAMPLE_SYSTEMS, ids=lambda system: system.name)
def example_system(request: pytest.FixtureRequest) -> ExampleSystem:
    """Build Hermi and PySCF data independently from each example's input."""
    specification = request.param
    molecule = Molecule.from_xyz(specification.xyz, units="bohr")
    if specification.custom_shells is None:
        assert specification.basis_set is not None
        basis = Basis.from_molecule(molecule, specification.basis_set)
        reference_basis = specification.basis_set
    else:
        angular_momentum = np.asarray([shell[0] for shell in specification.custom_shells], dtype=np.int64)
        exponent = np.asarray([shell[1] for shell in specification.custom_shells], dtype=np.float64)
        shells = Shells(
            position=np.zeros((len(exponent), 3), dtype=np.float64),
            exponent=exponent,
            coefficient=np.ones(len(exponent), dtype=np.float64),
            angular_momentum=angular_momentum,
            shell_indices=np.arange(len(exponent), dtype=np.int64),
        )
        basis = Basis.from_shells(shells).normalize()
        symbol = molecule.elements[0].symbol
        reference_basis = {
            symbol: [[momentum, [float(value), 1.0]] for momentum, value in specification.custom_shells]
        }
    reference = gto.M(
        atom=specification.atoms,
        basis=reference_basis,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    return ExampleSystem(specification.name, molecule, basis, reference)
