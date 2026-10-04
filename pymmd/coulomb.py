"""Auxiliary Coulomb terms in the Hermite basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from pymmd.boys import boys
from pymmd.hermite import hermite_indices
from pymmd.util import outer_powers

if TYPE_CHECKING:
    from typing import Iterator

    from numpy import float64, int64

    from pymmd._typing import array
    from pymmd.basis import ShellPairView


@dataclass
class CoulombIntermediates:
    """Reusable quantities for a primitive shell-pair quartet."""

    exponent_product: array[tuple[int, int], float64]
    exponent_sum: array[tuple[int, int], float64]
    rho: array[tuple[int, int], float64]

    @classmethod
    def from_shell_pairs(cls, pair_ab: ShellPairView, pair_cd: ShellPairView) -> CoulombIntermediates:
        """Compute shared exponents for a shell quartet."""
        exponent_product = np.multiply.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        exponent_sum = np.add.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        rho = exponent_product / exponent_sum
        return cls(exponent_product=exponent_product, exponent_sum=exponent_sum, rho=rho)


@lru_cache(maxsize=None)
def _auxiliary_coulomb_recurrence_indices(
    max_order: int,
) -> tuple[array[tuple[int, int], int64], ...]:
    """Return auxiliary-term target and parent indices by total order."""
    indices = hermite_indices(max_order)
    order_indices = [
        [index for index, triple in enumerate(indices) if sum(triple) == order] for order in range(max_order + 1)
    ]
    order_maps = [
        {indices[index]: local_index for local_index, index in enumerate(order_indices)}
        for order_indices in order_indices
    ]
    recurrence: list[array[tuple[int, int], int64]] = []

    for order in range(1, max_order + 1):
        target_indices = order_indices[order]
        targets = [indices[index] for index in target_indices]
        topology = np.zeros((5, len(targets)), dtype=np.int64)

        for index, target in enumerate(targets):
            axis = next(axis for axis, value in enumerate(target) if value > 0)
            parent = list(target)
            parent[axis] -= 1

            topology[0, index] = target_indices[index]
            topology[1, index] = order_maps[order - 1][tuple(parent)]  # type: ignore[index]
            topology[3, index] = axis

            if target[axis] > 1:
                grandparent = parent.copy()
                grandparent[axis] -= 1

                topology[2, index] = order_maps[order - 2][tuple(grandparent)]  # type: ignore[index]
                topology[4, index] = target[axis] - 1

        recurrence.append(topology)

    return tuple(recurrence)


@lru_cache(maxsize=None)
def _coulomb_matrix_indices(
    angular_momentum_ab: int,
    angular_momentum_cd: int,
) -> tuple[array[tuple[int, int], int64], array[tuple[int], float64]]:
    """Return Hermite sum indices and ket signs for a shell-pair quartet."""
    max_order = angular_momentum_ab + angular_momentum_cd
    hermite_ab = np.asarray(hermite_indices(angular_momentum_ab), dtype=np.int64)
    hermite_cd = np.asarray(hermite_indices(angular_momentum_cd), dtype=np.int64)
    hermite_all = np.asarray(hermite_indices(max_order), dtype=np.int64)

    hermite_map = np.full((max_order + 1,) * 3, -1, dtype=np.int64)
    hermite_map[tuple(hermite_all.T)] = np.arange(len(hermite_all), dtype=np.int64)

    hermite_sums = hermite_ab[:, None, :] + hermite_cd[None, :, :]
    sum_indices = hermite_map[tuple(hermite_sums.transpose(2, 0, 1))]
    signs = np.where(np.sum(hermite_cd, axis=-1) % 2 == 0, 1.0, -1.0)

    return sum_indices, signs


def auxiliary_coulomb_terms(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates | None = None,
) -> array[tuple[int, int, int, int], float64]:
    r"""Compute Hermite Coulomb auxiliary terms :math:`R^n_{tuv}`.

    Args:
        pair_ab: First primitive shell pair.
        pair_cd: Second primitive shell pair.
        intermediates: Precomputed quantities for the shell quartet.

    Returns:
        Auxiliary terms with triples flattened in total-order-first lexicographic order.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
    max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
    indices = hermite_indices(max_order)
    coefficients = np.zeros((len(pair_ab), len(pair_cd), max_order + 1, len(indices)), dtype=np.float64)
    for layer_indices, layer in _auxiliary_coulomb_layers(pair_ab, pair_cd, intermediates):
        valid = layer.shape[-1]
        coefficients[..., :valid, layer_indices] = np.swapaxes(layer, -1, -2)
    return coefficients


def _auxiliary_coulomb_layers(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates,
) -> Iterator[tuple[array[tuple[int, ...], int64], array[tuple[int, ...], float64]]]:
    """Yield each Hermite total-order layer in target-major order."""
    max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
    displacement = pair_ab.position.reshape(-1, 3)[:, None] - pair_cd.position.reshape(-1, 3)[None, :]
    argument = np.einsum("...,...x,...x->...", intermediates.rho, displacement, displacement, optimize=True)
    boys_values = boys(argument, max_order)

    seeds = outer_powers(-2 * intermediates.rho, max_order, boys_values)
    source0 = seeds[..., None, :]
    source1: array[tuple[int, ...], float64] | None = None
    yield np.asarray([0], dtype=np.int64), source0

    for order, topology in enumerate(_auxiliary_coulomb_recurrence_indices(max_order), start=1):
        valid = max_order - order + 1
        indices0, indices1, indices2, axes, factors = topology
        parent = np.take(source0, indices1, axis=-2)[..., 1 : valid + 1]
        target = parent * displacement[..., axes, None]

        if order > 1:
            assert source1 is not None
            grandparent = np.take(source1, indices2, axis=-2)[..., 1 : valid + 1]
            target += grandparent * factors[..., None]

        yield indices0, target
        source1, source0 = source0, target


def _auxiliary_coulomb_zero_order(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates,
) -> array[tuple[int, int, int], float64]:
    """Compute only the zero-order Coulomb terms needed for matrix assembly."""
    max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
    coefficients = np.empty((len(pair_ab), len(pair_cd), len(hermite_indices(max_order))), dtype=np.float64)
    for indices, layer in _auxiliary_coulomb_layers(pair_ab, pair_cd, intermediates):
        coefficients[..., indices] = layer[..., 0]
    return coefficients


def auxiliary_coulomb_matrix(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates | None = None,
) -> array[tuple[int, int, int, int], float64]:
    r"""Assemble the signed Hermite Coulomb interaction matrix.

    Args:
        pair_ab: First primitive shell pair.
        pair_cd: Second primitive shell pair.
        intermediates: Precomputed quantities for the shell quartet.

    Returns:
        Coulomb matrix.
    """
    if intermediates is None:
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
    sum_indices, signs = _coulomb_matrix_indices(pair_ab.angular_momentum, pair_cd.angular_momentum)
    auxiliary = _auxiliary_coulomb_zero_order(pair_ab, pair_cd, intermediates)
    return auxiliary[..., sum_indices] * signs
