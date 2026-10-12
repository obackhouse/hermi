"""Integration tests for the point-charge operator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from hermi.operators.point_charge import PointCharge
from hermi.systems.basis import Basis, Shells
from hermi.utils.angular import cartesian_indices, cartesian_normalization, cartesian_to_spherical


def _quadrature_reference(basis: Basis, position: np.ndarray, *, spherical: bool) -> np.ndarray:
    """Integrate basis functions against a point charge using radial and angular quadrature."""
    n_radial = 120
    n_z = 32
    n_azimuth = 64

    minimum_exponent = np.min(basis.shells.exponent)
    maximum_distance = np.max(np.linalg.norm(basis.shells.position - position, axis=-1))
    radial_maximum = maximum_distance + 12 / np.sqrt(minimum_exponent)

    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(n_radial)
    radial = (radial_nodes + 1) * radial_maximum / 2
    radial_weights = radial_weights * radial_maximum / 2

    z, z_weights = np.polynomial.legendre.leggauss(n_z)
    azimuth = 2 * np.pi * np.arange(n_azimuth) / n_azimuth
    z_grid, azimuth_grid = np.meshgrid(z, azimuth, indexing="ij")
    directions = np.stack(
        (
            np.sqrt(1 - z_grid**2) * np.cos(azimuth_grid),
            np.sqrt(1 - z_grid**2) * np.sin(azimuth_grid),
            z_grid,
        ),
        axis=-1,
    ).reshape(-1, 3)
    angular_weights = np.broadcast_to(z_weights[:, None] * (2 * np.pi / n_azimuth), z_grid.shape).ravel()

    points = position + radial[:, None, None] * directions[None, :, :]
    integration_weights = radial_weights[:, None] * radial[:, None] * angular_weights[None, :]

    integrals = []
    for shell in basis.iter_shells():
        cartesian = cartesian_indices(shell.angular_momentum)
        displacement = points[:, :, None, :] - shell.position[None, None, :, :]
        radius_squared = np.einsum("...x,...x->...", displacement, displacement, optimize=True)
        polynomial = np.prod(
            np.power(displacement[..., None, :], cartesian[None, None, None, :, :]),
            axis=-1,
        )
        primitive_values = (
            shell.coefficient[None, None, :, None]
            * np.exp(-shell.exponent[None, None, :, None] * radius_squared[..., None])
            * polynomial
            * cartesian_normalization(shell.angular_momentum)[None, None, None, :]
        )
        values = np.sum(primitive_values, axis=2)
        if spherical:
            values = values @ cartesian_to_spherical(shell.angular_momentum)
        integrals.append(np.einsum("ra,rai->i", integration_weights, values, optimize=True))

    return np.concatenate(integrals)


def test_point_charge_s_primitive() -> None:
    """Compare an s primitive with its closed-form point-charge integral."""
    exponent = 0.8
    position = np.asarray([0.2, -0.3, 0.4], dtype=np.float64)
    shells = Shells(
        position=np.zeros((1, 3), dtype=np.float64),
        exponent=np.asarray([exponent], dtype=np.float64),
        coefficient=np.ones(1, dtype=np.float64),
        angular_momentum=np.zeros(1, dtype=np.int64),
        shell_indices=np.zeros(1, dtype=np.int64),
    )
    basis = Basis.from_shells(shells).normalize()

    argument = exponent * np.dot(position, position)
    boys_zero = 1.0 if argument == 0 else np.sqrt(np.pi) * erf(np.sqrt(argument)) / (2 * np.sqrt(argument))
    expected = basis.shells.coefficient[0] * 2 * np.pi / exponent * boys_zero
    actual = PointCharge(position).collect(basis)

    np.testing.assert_allclose(actual, [expected], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("spherical", (False, True))
def test_point_charge_against_quadrature(example_system, spherical: bool) -> None:
    """Compare batched point-charge integrals with independent radial and angular quadrature."""
    center = example_system.basis.shells.position[0]
    offsets = np.asarray([[0.02, -0.03, 0.04], [-0.05, 0.01, 0.03]], dtype=np.float64)
    positions = center + offsets
    operator = PointCharge(positions)

    actual = operator.collect(example_system.basis, spherical=spherical, batch_dim=True)
    reference = np.stack(
        [_quadrature_reference(example_system.basis, position, spherical=spherical) for position in positions]
    )

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=2e-9)

    ao_slice = slice(None, None, 2)
    sliced = operator.collect(example_system.basis, ao_slices=(ao_slice,), spherical=spherical, batch_dim=True)
    indices = np.arange(*ao_slice.indices(actual.shape[1]))
    np.testing.assert_allclose(sliced, actual[:, indices], rtol=1e-12, atol=1e-12)
