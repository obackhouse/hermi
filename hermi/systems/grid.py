"""Molecular integration grids."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from hermi.util import DefaultHash

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.molecule import Molecule


def _radial_grid(order: int, scale: float) -> tuple[array[tuple[int], float64], array[tuple[int], float64]]:
    r"""Generate a radial quadrature grid using Gauss-Legendre quadrature on a rational map.

    The rational map is given by

    .. math::
        r(x) = \frac{r_\text{scale} x}{1 - x}, \quad x \in [0, 1]

    The Jacobian of the map is

    .. math::
        \frac{dr}{dx} = \frac{r_\text{scale}}{(1 - x)^2}

    Args:
        order: Number of points in the quadrature.
        scale: Length scale of the map.

    Returns:
        Tuple of radial points and weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    fraction = (nodes + 1) / 2
    weights /= 2
    radius = scale * fraction / (1 - fraction)
    jacobian = scale / (1 - fraction) ** 2
    weights *= jacobian * radius**2
    return radius, weights


def _angular_grid(order: int) -> tuple[array[tuple[int], float64], array[tuple[int], float64]]:
    r"""Generate an angular quadrature grid using a product of Gauss-Legendre and equally spaced azimuthal points.

    The polar Gauss-Legendre points are given by the roots of the Legendre polynomial of degree ``order``.
    The azimuthal points are given by ``2 * order + 1`` equally spaced points in [0, 2pi).

    Args:
        order: Number of polar Gauss-Legendre points. The azimuthal rule uses ``2 * order + 1`` points.

    Returns:
        Tuple of angular directions and weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    azimuth_count = 2 * order + 1
    azimuth = 2 * np.pi * np.arange(azimuth_count) / azimuth_count
    polar_grid, azimuth_grid = np.meshgrid(nodes, azimuth, indexing="ij")
    directions = np.stack(
        (
            np.sqrt(1 - polar_grid**2) * np.cos(azimuth_grid),
            np.sqrt(1 - polar_grid**2) * np.sin(azimuth_grid),
            polar_grid,
        ),
        axis=-1,
    ).reshape((-1, 3))
    weights = np.broadcast_to(weights[:, None] * (2 * np.pi / azimuth_count), polar_grid.shape).ravel()
    return directions, weights  # type: ignore[return-value]


@dataclass(frozen=True, eq=False, slots=True)
class MolecularGrid(DefaultHash):
    """Atom-centred molecular quadrature grid."""

    points: array[tuple[int, int], float64]
    weights: array[tuple[int], float64]

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be a 2D array of shape (N, 3), got {self.points.shape}")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.points.shape[0]:
            raise ValueError(f"weights must be a 1D array of shape (N,), got {self.weights.shape}")

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        *,
        radial_order: int = 48,
        angular_order: int = 12,
        radial_scale: float = 1.0,
        becke_iterations: int = 3,
    ) -> MolecularGrid:
        r"""Generate a grid for a molecule using the Becke partitioning scheme.

        The radial rule uses Gauss-Legendre quadrature on a rational map from [0, 1] to
        [0, infinity).

        The angular rule is a product of Gauss-Legendre and equally spaced azimuthal points.

        The Becke partition is used to assign weights to each atom's grid points.

        Args:
            molecule: Molecule whose coordinates are in Bohr.
            radial_order: Number of points in the radial quadrature.
            angular_order: Number of polar Gauss-Legendre points. The azimuthal rule uses
                ``2 * angular_order + 1`` points.
            radial_scale: Length scale of the radial map in Bohr.
            becke_iterations: Number of cubic iterations in the Becke partition.

        Returns:
            Molecular grid.
        """
        radius, radial_weights = _radial_grid(radial_order, radial_scale)
        directions, angular_weights = _angular_grid(angular_order)
        atom_weights = np.multiply.outer(radial_weights, angular_weights).ravel()
        atom_positions = molecule.positions
        internuclear_distances = np.linalg.norm(atom_positions[:, None] - atom_positions[None, :], axis=-1)
        radial_directions = np.multiply.outer(radius, directions).reshape((-1, 3))

        points = []
        weights = []
        for atom_index, position in enumerate(atom_positions):
            atom_points = position + radial_directions

            if atom_positions.shape[0] == 1:
                partition_weights = np.ones(atom_points.shape[0], dtype=np.float64)
            else:
                distances = np.linalg.norm(atom_points[:, None, :] - atom_positions[None, :, :], axis=-1)
                log_partition = np.zeros((atom_positions.shape[0], atom_points.shape[0]), dtype=np.float64)
                for first, second in itertools.product(range(atom_positions.shape[0]), repeat=2):
                    if first == second:
                        continue
                    mu = (distances[:, first] - distances[:, second]) / internuclear_distances[first, second]
                    switching = np.clip(mu, -1, 1)
                    for _ in range(becke_iterations):
                        switching = 0.5 * switching * (3 - switching**2)
                    partition = np.clip(0.5 * (1 - switching), np.finfo(np.float64).tiny, 1)
                    log_partition[first] += np.log(partition)
                partition_weights = np.exp(log_partition[atom_index] - logsumexp(log_partition, axis=0))

            points.append(atom_points)
            weights.append(atom_weights * partition_weights)

        return cls(points=np.concatenate(points), weights=np.concatenate(weights))

    def __len__(self) -> int:
        return self.points.shape[0]
