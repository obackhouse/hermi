"""Molecular integration grids."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special

from hermi.operators.overlap import Overlap
from hermi.systems._grid_data import BRAGG_RADII_ANGSTROM, LEBEDEV_ORBITS, TREUTLER_XI
from hermi.utils.misc import DefaultHash

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.systems.basis import Basis
    from hermi.systems.molecule import Molecule


def _becke_partition_weights(
    atom_points: array[float64],
    atom_index: int,
    atom_positions: array[float64],
    internuclear_distances: array[float64],
    becke_iterations: int,
    radius_adjustments: array[float64] | None = None,
) -> array[float64]:
    """Return Becke partition weights for one atom's points."""
    num_atoms = atom_positions.shape[0]
    if num_atoms == 1:
        return np.ones(atom_points.shape[0], dtype=np.float64)
    distances = np.linalg.norm(atom_points[:, None, :] - atom_positions[None, :, :], axis=-1)
    atom_indices = np.arange(num_atoms)

    # Get the log of the product of the pairwise partitions for each atom and point
    log_partition = np.zeros((num_atoms, atom_points.shape[0]), dtype=np.float64)
    for first in range(num_atoms):
        other_atoms = atom_indices != first
        pair_distances = internuclear_distances[first, other_atoms]
        mu = (distances[:, first, None] - distances[:, other_atoms]) / pair_distances
        if radius_adjustments is not None:
            pair_adjustments = radius_adjustments[first, other_atoms]
            mu += pair_adjustments * (1 - mu**2)
        switching = np.clip(mu, -1, 1)
        for _ in range(becke_iterations):
            switching = 0.5 * switching * (3 - switching**2)
        partition = np.clip(0.5 * (1 - switching), np.finfo(np.float64).tiny, 1)
        log_partition[first] = np.sum(np.log(partition), axis=1)

    # Compute the weights
    log_weights = log_partition[atom_index] - scipy.special.logsumexp(log_partition, axis=0)
    weights = np.exp(log_weights)

    return weights  # type: ignore[no-any-return]


def _radial_overlap(
    exponents_i: array[float64],
    coefficients_i: array[float64],
    exponents_j: array[float64],
    coefficients_j: array[float64],
) -> float:
    """Integrate the product of two radial Gaussian contractions."""
    exponent_sum = exponents_i[:, None] + exponents_j[None, :]
    coefficient_product = coefficients_i[:, None] * coefficients_j[None, :]
    return float(np.sum(coefficient_product * 0.5 * scipy.special.gamma(1.5) / exponent_sum**1.5))


def _radial_integrals(
    exponents_i: array[float64],
    coefficients_i: array[float64],
    exponents_j: array[float64],
    coefficients_j: array[float64],
) -> tuple[float, float]:
    """Return overlap and position integrals for two radial contractions."""
    exponent_sum = exponents_i[:, None] + exponents_j[None, :]
    coefficient_product = coefficients_i[:, None] * coefficients_j[None, :]
    overlap = np.sum(coefficient_product * 0.5 * scipy.special.gamma(1.5) / exponent_sum**1.5)
    position = np.sum(coefficient_product * 0.5 / exponent_sum**2)
    return float(overlap), float(position)


def _radial_dvr(
    basis: Basis,
    atom_position: array[float64],
    eigenvalue_cutoff: float,
) -> tuple[array[float64], array[float64]]:
    """Build radial R-DVR nodes and nonnegative weights for one atom."""
    starts = basis.contracted_shell_starts(slice(None))
    stops = np.append(starts[1:], basis.num_primitive_shells)
    shell_positions = basis.shells.position[starts]
    belongs_to_atom = np.all(np.isclose(shell_positions, atom_position), axis=1)

    # Assemble the normalized radial contractions for the atom
    contractions: list[tuple[array[float64], array[float64]]] = []
    for first, last in itertools.compress(zip(starts, stops), belongs_to_atom):
        selected = slice(first, last)
        exponents = basis.shells.exponent[selected]
        angular_momentum = int(basis.shells.angular_momentum[first])
        primitive_normalization = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (angular_momentum / 2)
        coefficients = basis.shells.coefficient[selected] / primitive_normalization
        radial_norm = np.sqrt(_radial_overlap(exponents, coefficients, exponents, coefficients))
        contractions.append((exponents, coefficients / radial_norm))

    # Assemble radial overlap and position matrices in the contracted-shell basis
    overlap = np.zeros((len(contractions), len(contractions)), dtype=np.float64)
    position = np.zeros_like(overlap)
    for i, j in itertools.combinations_with_replacement(range(len(contractions)), 2):
        overlap[i, j], position[i, j] = _radial_integrals(
            contractions[i][0], contractions[i][1], contractions[j][0], contractions[j][1]
        )

    # Orthogonalize the contractions and diagonalize the projected position operator
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    keep = eigenvalues > eigenvalue_cutoff * np.max(eigenvalues)
    orthogonalizer = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    position_orthogonal = orthogonalizer.T @ position @ orthogonalizer
    radii, rotation = np.linalg.eigh(position_orthogonal)

    # Evaluate the radial contractions at the DVR nodes and compute the collocation diagonal
    radial_values = np.empty((len(contractions), radii.size), dtype=np.float64)
    for i, (exponents, coefficients) in enumerate(contractions):
        radial_values[i] = np.exp(-exponents[:, None] * radii[None, :] ** 2).T @ coefficients
    dvr_values = radial_values.T @ orthogonalizer @ rotation
    diagonal = np.diag(dvr_values)
    valid = (radii > 0) & np.isfinite(diagonal) & (np.abs(diagonal) > np.finfo(np.float64).tiny)
    radii = radii[valid]
    radial_values = radial_values[:, valid]
    diagonal = diagonal[valid]

    # Refit weights to the exact overlap products to enforce nonnegativity and compensate for finite-basis loss
    row, column = np.triu_indices(len(contractions))
    row_scale = np.where(row == column, 1.0, np.sqrt(2.0))
    design = radial_values[row] * radial_values[column] * row_scale[:, None]
    target = overlap[row, column] * row_scale
    column_norms = np.linalg.norm(design, axis=0)
    nonzero_columns = column_norms > np.finfo(np.float64).tiny
    scaled_design = design[:, nonzero_columns] / column_norms[nonzero_columns]
    scaled_weights, _ = scipy.optimize.nnls(scaled_design, target)
    weights = np.zeros(radii.size, dtype=np.float64)
    weights[nonzero_columns] = scaled_weights / column_norms[nonzero_columns]
    if not np.any(weights > 0):
        weights = 1 / diagonal**2

    return radii, weights


def _simultaneous_diagonalize(
    matrices: array[float64], *, tolerance: float, max_sweeps: int
) -> tuple[array[float64], array[float64]]:
    """Approximately jointly diagonalize symmetric matrices with Jacobi sweeps."""
    _, n, _ = matrices.shape
    rotation = np.eye(n, dtype=np.float64)

    # Move the batch axis to the end for better memory alignment
    diagonalized = np.moveaxis(matrices, 0, -1).copy()

    # Round-robin edge coloring of the complete graph
    n_players = n if n % 2 == 0 else n + 1
    players = list(range(n_players))
    pair_groups: list[tuple[array[np.int64], array[np.int64]]] = []
    for _ in range(n_players - 1):
        pairs = [
            (players[index], players[-index - 1])
            for index in range(n_players // 2)
            if players[index] < n and players[-index - 1] < n
        ]
        pair_groups.append(
            (
                np.asarray([pair[0] for pair in pairs], dtype=np.int64),
                np.asarray([pair[1] for pair in pairs], dtype=np.int64),
            )
        )
        players = [players[0], players[-1], *players[1:-1]]

    # Perform Jacobi sweeps until convergence or maximum sweeps reached
    for _ in range(max_sweeps):
        largest_off_diagonal = 0.0
        for p, q in pair_groups:
            if p.size == 0:
                continue
            off = diagonalized[p, q]
            delta = 0.5 * (diagonalized[q, q] - diagonalized[p, p])
            off_magnitudes = np.max(np.abs(off), axis=-1)
            largest_off_diagonal = max(largest_off_diagonal, np.max(off_magnitudes))

            # Only rotate pairs with significant off-diagonal elements
            active = off_magnitudes > tolerance
            if not np.any(active):
                continue
            p_active, q_active = p[active], q[active]
            off = off[active]
            delta = delta[active]

            # Diagonalise the 2x2 objective matrix for each pair to get the rotation angle
            components = np.array([off, delta])
            objective = components.transpose(1, 0, 2) @ components.transpose(1, 2, 0)
            _, vectors = np.linalg.eigh(objective)
            theta = 0.5 * np.arctan2(vectors[:, 1, 0], vectors[:, 0, 0])
            cosine, sine = np.cos(theta), np.sin(theta)

            # Transform the selected columns
            old_p = diagonalized[:, p_active, :].copy()
            old_q = diagonalized[:, q_active, :].copy()
            diagonalized[:, p_active, :] = cosine[None, :, None] * old_p + sine[None, :, None] * old_q
            diagonalized[:, q_active, :] = -sine[None, :, None] * old_p + cosine[None, :, None] * old_q

            # Transform the corresponding rows
            old_p = diagonalized[p_active, :, :].copy()
            old_q = diagonalized[q_active, :, :].copy()
            diagonalized[p_active, :, :] = cosine[:, None, None] * old_p + sine[:, None, None] * old_q
            diagonalized[q_active, :, :] = -sine[:, None, None] * old_p + cosine[:, None, None] * old_q

            # Update the rotation matrix
            old_p = rotation[:, p_active].copy()
            old_q = rotation[:, q_active].copy()
            rotation[:, p_active] = cosine[None] * old_p + sine[None] * old_q
            rotation[:, q_active] = -sine[None] * old_p + cosine[None] * old_q

        if largest_off_diagonal <= tolerance:
            break

    # Return to the original axis order
    diagonalized = np.moveaxis(diagonalized, -1, 0)

    return diagonalized, rotation


def _assemble_atom_centred_grid(
    molecule: Molecule,
    radial_rules: list[tuple[array[float64], array[float64]]],
    angular_order: int,
    becke_iterations: int,
) -> tuple[array[float64], array[float64]]:
    """Combine Treutler-pruned Lebedev grids and radius-adjusted Becke weights."""
    max_angular_points = _lebedev_point_count(angular_order)
    atom_positions = molecule.positions
    distances = np.linalg.norm(atom_positions[:, None] - atom_positions[None, :], axis=-1)
    atomic_charges = np.asarray([element.atomic_charge for element in molecule.elements])
    atomic_radii = np.sqrt(np.asarray(BRAGG_RADII_ANGSTROM)[atomic_charges])
    radius_ratios = atomic_radii[:, None] / atomic_radii[None, :]
    radius_adjustments = np.clip(0.25 * (radius_ratios.T - radius_ratios), -0.5, 0.5)

    points: list[array[float64]] = []
    weights: list[array[float64]] = []
    for atom_index, (position, (radius, radial_weights)) in enumerate(zip(atom_positions, radial_rules, strict=True)):
        angular_points_by_radius = _treutler_pruned_angular_points(radius, max_angular_points)
        for angular_points in np.unique(angular_points_by_radius):
            radial_indices = np.flatnonzero(angular_points_by_radius == angular_points)
            directions, angular_weights = _lebedev_grid(int(angular_points))
            atom_points = position + np.multiply.outer(radius[radial_indices], directions).reshape((-1, 3))
            product_weights = np.multiply.outer(radial_weights[radial_indices], angular_weights).ravel()
            partition = _becke_partition_weights(
                atom_points,
                atom_index,
                atom_positions,
                distances,
                becke_iterations,
                radius_adjustments,
            )
            points.append(atom_points)
            weights.append(product_weights * partition)

    return np.concatenate(points), np.concatenate(weights)


def _treutler_radial_grid(order: int, atomic_charge: int, scale: float) -> tuple[array[float64], array[float64]]:
    """Generate Treutler-Ahlrichs radial nodes and volume weights."""
    xi = TREUTLER_XI[atomic_charge] if atomic_charge < len(TREUTLER_XI) else 1.0
    step = np.pi / (order + 1)
    index = np.arange(1, order + 1, dtype=np.float64)
    angle = index * step
    cosine = np.cos(angle)
    sine = np.sin(angle)
    log_term = np.log((1 - cosine) / 2)
    radius = -(xi / np.log(2)) * (1 + cosine) ** 0.6 * log_term
    radial_weight = (
        step * sine * (xi / np.log(2)) * (1 + cosine) ** 0.6 * (-0.6 / (1 + cosine) * log_term + 1 / (1 - cosine))
    )
    radius = scale * radius[::-1]
    radial_weight = scale * radial_weight[::-1]
    return radius, radius**2 * radial_weight


def _angular_grid(order: int) -> tuple[array[float64], array[float64]]:
    """Return the Lebedev rule matching an angular resolution order."""
    return _lebedev_grid(_lebedev_point_count(order))


def _lebedev_point_count(order: int) -> int:
    """Map angular resolution to the first adequate nonnegative Lebedev rule."""
    minimum_points = order * (2 * order + 1)
    sizes = np.fromiter(LEBEDEV_ORBITS, dtype=np.int64)
    index = int(np.searchsorted(sizes, minimum_points))
    for size in sizes[index:]:
        if all(orbit[3] >= 0 for orbit in LEBEDEV_ORBITS[int(size)]):
            return int(size)
    raise ValueError(f"angular order {order} exceeds the largest bundled nonnegative Lebedev rule")


@lru_cache(maxsize=32)
def _lebedev_grid(angular_points: int) -> tuple[array[float64], array[float64]]:
    """Return Lebedev directions and weights normalized over the sphere."""
    try:
        orbits = LEBEDEV_ORBITS[angular_points]
    except KeyError as error:
        raise ValueError(f"unsupported Lebedev point count: {angular_points}") from error
    points: list[tuple[float, float, float]] = []
    weights: list[float] = []
    seen: set[tuple[float, float, float]] = set()
    for code, a, b, weight in orbits:
        if code == 0:
            orbit = ((1.0, 0.0, 0.0),)
        elif code == 1:
            value = np.sqrt(0.5)
            orbit = ((0.0, value, value),)
        elif code == 2:
            value = np.sqrt(1.0 / 3.0)
            orbit = ((value, value, value),)
        elif code in (3, 4, 5, 6):
            if code == 3:
                orbit = ((a, a, np.sqrt(max(0.0, 1.0 - 2 * a * a))),)
            elif code == 4:
                orbit = ((a, np.sqrt(max(0.0, 1.0 - a * a)), 0.0),)
            elif code == 5:
                orbit = ((a, b, np.sqrt(max(0.0, 1.0 - a * a - b * b))),)
            else:
                orbit = ((a, b, np.sqrt(max(0.0, 1.0 - a * a - b * b))),)
        else:
            raise ValueError(f"unknown Lebedev orbit code: {code}")

        for seed in orbit:
            for permutation in set(itertools.permutations(seed)):
                for signs in itertools.product((-1.0, 1.0), repeat=3):
                    point = (permutation[0] * signs[0], permutation[1] * signs[1], permutation[2] * signs[2])
                    if point not in seen:
                        seen.add(point)
                        points.append(point)
                        weights.append(weight * 4 * np.pi)
    return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _treutler_pruned_angular_points(radial_nodes: array[float64], max_angular_points: int) -> array[np.int64]:
    """Return the Treutler angular rule size for each radial shell."""
    if max_angular_points < 50:
        return np.full(radial_nodes.size, max_angular_points, dtype=np.int64)
    points = np.empty(radial_nodes.size, dtype=np.int64)
    points[: radial_nodes.size // 3] = 14
    points[radial_nodes.size // 3 : radial_nodes.size // 2] = 50
    points[radial_nodes.size // 2 :] = max_angular_points
    return points


@dataclass(frozen=True, eq=False, slots=True)
class MolecularGrid(DefaultHash):
    """Atom-centred molecular quadrature grid."""

    points: array[float64]
    weights: array[float64]

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be a 2D array of shape (N, 3), got {self.points.shape}")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.points.shape[0]:
            raise ValueError(f"weights must be a 1D array of shape (N,), got {self.weights.shape}")

    @classmethod
    def from_becke(
        cls,
        molecule: Molecule,
        *,
        radial_order: int = 48,
        angular_order: int = 12,
        radial_scale: float = 1.0,
        becke_iterations: int = 3,
    ) -> MolecularGrid:
        """Build a product radial-angular grid with Becke partitioning.

        Args:
            molecule: Molecule with coordinates.
            radial_order: Number of Treutler-Ahlrichs radial points per atom.
            angular_order: Angular resolution; the smallest adequate nonnegative-weight Lebedev rule is selected.
            radial_scale: Scale applied to the Treutler-Ahlrichs radial nodes.
            becke_iterations: Number of cubic Becke switching steps.

        Returns:
            The Becke-partitioned molecular grid.

        References:
            A. D. Becke, J. Chem. Phys. 88, 2547–2553 (1988).
            https://doi.org/10.1063/1.454033
            A. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
            https://doi.org/10.1063/1.469408
            V. I. Lebedev and D. N. Laikov, Dokl. Math. 59, 477–481 (1999).
        """
        radial_rules = [
            _treutler_radial_grid(radial_order, element.atomic_charge, radial_scale) for element in molecule.elements
        ]
        points, weights = _assemble_atom_centred_grid(molecule, radial_rules, angular_order, becke_iterations)
        return cls(points=points, weights=weights)

    @classmethod
    def from_handy(
        cls,
        molecule: Molecule,
        radial_nodes_by_element: dict[str, array[float64]],
        *,
        angular_order: int = 12,
        radial_scale: float = 1.0,
        becke_iterations: int = 3,
    ) -> MolecularGrid:
        """Build a grid from element-specific radial nodes.

        Args:
            molecule: Molecule to grid, with coordinates.
            radial_nodes_by_element: Mapping from element symbols to increasing radial nodes.
            angular_order: Angular resolution; the smallest adequate nonnegative-weight Lebedev rule is selected.
            radial_scale: Scale of the radial coordinate map.
            becke_iterations: Number of cubic Becke switching steps.

        Returns:
            The Handy-style molecular grid.

        References:
            C. W. Murray, N. C. Handy, and G. J. Laming, Mol. Phys. 78, 997–1014 (1993).
            https://doi.org/10.1080/00268979300100651
            S. I. L. Kokkila-Schumacher et al., J. Chem. Theory Comput. 11, 3042–3052 (2015).
            https://doi.org/10.1021/acs.jctc.5b00272
        """
        rules = []
        for element in molecule.elements:
            radii = np.asarray(radial_nodes_by_element[element.symbol], dtype=np.float64)
            x = radii / (radial_scale + radii)
            boundaries = np.concatenate(([0.0], x, [1.0]))
            dx = 0.5 * (boundaries[2:] - boundaries[:-2])
            dr_dx = radial_scale / (1 - x) ** 2
            rules.append((radii, dx * dr_dx * radii**2))
        points, weights = _assemble_atom_centred_grid(molecule, rules, angular_order, becke_iterations)
        return cls(points=points, weights=weights)

    @classmethod
    def from_r_dvr(
        cls,
        molecule: Molecule,
        basis: Basis,
        *,
        angular_order: int = 12,
        becke_iterations: int = 3,
        eigenvalue_cutoff: float = 1e-10,
    ) -> MolecularGrid:
        """Build an atom-centred radial DVR grid from Gaussian basis contractions.

        Args:
            molecule: Molecule to grid, with coordinates.
            basis: Basis used to construct the nodes.
            angular_order: Angular resolution; the smallest adequate nonnegative-weight Lebedev rule is selected.
            becke_iterations: Number of cubic Becke switching steps.
            eigenvalue_cutoff: Relative cutoff for radial overlap eigenvalues.

        Returns:
            The radial-DVR molecular grid.

        References:
            R. M. Parrish, E. G. Hohenstein, T. J. Martínez, and C. D. Sherrill,
            J. Chem. Phys. 138, 194107 (2013). https://doi.org/10.1063/1.4802773
        """
        rules = [_radial_dvr(basis, position, eigenvalue_cutoff) for position in molecule.positions]
        points, weights = _assemble_atom_centred_grid(molecule, rules, angular_order, becke_iterations)
        return cls(points=points, weights=weights)

    @classmethod
    def from_f_dvr(
        cls,
        basis: Basis,
        parent_grid: MolecularGrid,
        *,
        candidate_grid: MolecularGrid | None = None,
        include_candidate_grid: bool = True,
        spherical: bool = True,
        overlap_cutoff: float = 1e-10,
        diagonalization_tolerance: float = 1e-10,
        max_sweeps: int = 100,
    ) -> MolecularGrid:
        """Build finite-basis DVR nodes from overlap and position matrices.

        Args:
            basis: Basis used to construct the nodes.
            parent_grid: Grid used to integrate overlap and position matrices.
            candidate_grid: Grid appended to the F-DVR nodes; defaults to ``parent_grid``.
            include_candidate_grid: Append support points to the nodes.
            spherical: Use spherical rather than Cartesian basis functions.
            overlap_cutoff: Relative cutoff for overlap eigenvalues.
            diagonalization_tolerance: Jacobi stopping tolerance.
            max_sweeps: Maximum joint-diagonalization sweeps.

        Returns:
            The finite-basis DVR grid with raw inverse-square weights.

        References:
            R. M. Parrish, E. G. Hohenstein, T. J. Martínez, and C. D. Sherrill,
            J. Chem. Phys. 138, 194107 (2013). https://doi.org/10.1063/1.4802773
        """
        # Orthogonalize the basis in the parent-grid overlap metric
        values = basis.evaluate(parent_grid.points, spherical=spherical)
        values *= np.sqrt(parent_grid.weights[:, None])
        overlap = values.T @ values
        eigenvalues, eigenvectors = np.linalg.eigh(overlap)
        mask = eigenvalues > overlap_cutoff * np.max(eigenvalues)
        orthogonalizer = eigenvectors[:, mask] / np.sqrt(eigenvalues[mask])

        # Project the three position operators into the orthonormal basis.
        position_matrices = np.empty((3, overlap.shape[0], overlap.shape[1]), dtype=np.float64)
        for axis in range(3):
            position_matrices[axis] = values.T @ (values * parent_grid.points[:, axis, None])
        position_matrices = np.einsum(
            "ia,kij,jb->kab", orthogonalizer, position_matrices, orthogonalizer, optimize=True
        )

        # Normalize the components so the molecular long axis does not dominate the initial rotation
        matrix_scales = np.linalg.norm(position_matrices, axis=(1, 2))
        matrix_scales = np.maximum(matrix_scales, np.finfo(np.float64).tiny)
        seed_operator = np.sum(position_matrices / matrix_scales[:, None, None], axis=0)
        seed_rotation = np.linalg.eigh(seed_operator)[1]
        seeded_matrices = np.einsum("ia,kij,jb->kab", seed_rotation, position_matrices, seed_rotation, optimize=True)
        diagonalized, rotation = _simultaneous_diagonalize(
            seeded_matrices, tolerance=diagonalization_tolerance, max_sweeps=max_sweeps
        )
        rotation = seed_rotation @ rotation

        # Use the joint diagonal as the nodes and the collocation diagonal for weights
        points = np.moveaxis(np.diagonal(diagonalized, axis1=1, axis2=2), 0, -1)
        dvr_coefficients = orthogonalizer @ rotation
        collocation = basis.evaluate(points, spherical=spherical) @ dvr_coefficients
        weights = 1 / np.diag(collocation) ** 2

        # Add the support grid
        if include_candidate_grid:
            support_grid = parent_grid if candidate_grid is None else candidate_grid
            points = np.concatenate((points, support_grid.points))
            weights = np.concatenate((weights, support_grid.weights))

        return cls(points=points, weights=weights)

    def reweight_nnls(
        self,
        basis: Basis,
        *,
        spherical: bool = True,
        relative_weight_cutoff: float = 1e-12,
        maxiter: int | None = None,
    ) -> MolecularGrid:
        """Fit nonnegative weights to the AO overlap matrix with dense NNLS.

        Args:
            basis: Basis used to construct the AO overlap matrix.
            spherical: Use spherical rather than Cartesian basis functions.
            relative_weight_cutoff: Discard weights below this fraction of the maximum.
            maxiter: Maximum NNLS iterations; defaults to ``None`` for the SciPy default

        Returns:
            The molecular grid with nonnegative weights fitted to the AO overlap matrix.
        """
        values = basis.evaluate(self.points, spherical=spherical)
        overlap = Overlap().collect(basis, spherical=spherical)

        # Build the symmetric AO-pair fitting system with Frobenius scaling
        rows, columns = np.triu_indices(values.shape[1])
        row_scale = np.where(rows == columns, 1.0, np.sqrt(2.0))
        design = (values[:, rows] * values[:, columns]).T * row_scale[:, None]
        target = overlap[rows, columns] * row_scale
        column_norms = np.linalg.norm(design, axis=0)
        nonzero_columns = column_norms > np.finfo(np.float64).tiny
        scaled_design = design[:, nonzero_columns]
        scaled_design /= column_norms[nonzero_columns]

        # Select a numerically independent spanning set before solving and solve
        triangular, pivots = scipy.linalg.qr(scaled_design, mode="r", pivoting=True, check_finite=False)
        diagonal = np.abs(np.diag(triangular))
        rank = int(np.count_nonzero(diagonal > diagonal[0] * np.finfo(np.float64).eps * max(scaled_design.shape)))
        if rank == 0:
            raise ValueError("candidate AO-pair products are numerically rank deficient")
        selected_columns = pivots[:rank]
        selected_design = scaled_design[:, selected_columns]
        scaled_weights, _ = scipy.optimize.nnls(selected_design, target, maxiter=maxiter)

        # Construct the final grid with nonnegative weights and discard small weights
        active_indices = np.flatnonzero(nonzero_columns)[selected_columns]
        fitted_weights = scaled_weights / column_norms[active_indices]
        keep = scaled_weights > relative_weight_cutoff * np.max(scaled_weights)
        points = self.points[active_indices[keep]]
        weights = fitted_weights[keep]

        return self.__class__(points=points, weights=weights)

    def __len__(self) -> int:
        return self.points.shape[0]
