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
    from typing import Literal

    from numpy import float64, int64

    from pymmd._typing import array
    from pymmd.basis import ShellPairView


@dataclass
class CoulombIntermediates:
    """Reusable quantities for a primitive shell-pair quartet."""

    exponent_product: array[tuple[int, int], float64]
    exponent_sum: array[tuple[int, int], float64]

    @classmethod
    def from_shell_pairs(cls, pair_ab: ShellPairView, pair_cd: ShellPairView) -> CoulombIntermediates:
        """Compute shared exponents for a shell quartet."""
        exponent_product = np.multiply.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        exponent_sum = np.add.outer(pair_ab.exponent.ravel(), pair_cd.exponent.ravel())
        return cls(exponent_product=exponent_product, exponent_sum=exponent_sum)


@lru_cache(maxsize=None)
def _coulomb_matrix_indices(
    angular_momentum_ab: int,
    angular_momentum_cd: int,
    sign_side: Literal["bra", "ket"] = "ket",
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

    signed_indices = hermite_ab if sign_side == "bra" else hermite_cd
    signs = np.where(np.sum(signed_indices, axis=-1) % 2 == 0, 1.0, -1.0)

    return sum_indices, signs


def auxiliary_coulomb_matrix(
    pair_ab: ShellPairView,
    pair_cd: ShellPairView,
    intermediates: CoulombIntermediates | None = None,
    parity: Literal["bra", "ket"] = "ket",
) -> tuple[
    array[tuple[int, int, int, int], float64],
    array[tuple[int], float64],
]:
    """Assemble the signed Hermite Coulomb interaction matrix."""
    if intermediates is None:
        intermediates = CoulombIntermediates.from_shell_pairs(pair_ab, pair_cd)
    max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
    if parity == "bra":
        displacement = pair_cd.position.reshape(1, -1, 3) - pair_ab.position.reshape(-1, 1, 3)
    elif parity == "ket":
        displacement = pair_ab.position.reshape(-1, 1, 3) - pair_cd.position.reshape(1, -1, 3)
    else:
        raise ValueError(f"invalid parity: {parity!r}")
    rho = intermediates.exponent_product / intermediates.exponent_sum
    argument = np.einsum("...,...x,...x->...", rho, displacement, displacement, optimize=True)

    auxiliary = np.zeros((max_order + 1,) * 4 + (len(pair_ab), len(pair_cd)), dtype=np.float64)

    # R_n^000
    auxiliary[0, 0, 0] = np.moveaxis(outer_powers(-2 * rho, max_order) * boys(argument, max_order), -1, 0)

    # R_n^t00
    for t in range(max_order):
        valid = max_order - t
        auxiliary[t + 1, 0, 0, :valid] = auxiliary[t, 0, 0, 1 : valid + 1] * displacement[..., 0]
        if t:
            auxiliary[t + 1, 0, 0, :valid] += t * auxiliary[t - 1, 0, 0, 1 : valid + 1]

    # R_n^tu0
    for t in range(max_order + 1):
        for u in range(max_order - t):
            valid = max_order - t - u
            auxiliary[t, u + 1, 0, :valid] = auxiliary[t, u, 0, 1 : valid + 1] * displacement[..., 1]
            if u:
                auxiliary[t, u + 1, 0, :valid] += u * auxiliary[t, u - 1, 0, 1 : valid + 1]

    # R_n^tuv
    for t in range(max_order + 1):
        for u in range(max_order - t + 1):
            for v in range(max_order - t - u):
                valid = max_order - t - u - v
                auxiliary[t, u, v + 1, :valid] = auxiliary[t, u, v, 1 : valid + 1] * displacement[..., 2]
                if v:
                    auxiliary[t, u, v + 1, :valid] += v * auxiliary[t, u, v - 1, 1 : valid + 1]

    indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
    sum_indices, signs = _coulomb_matrix_indices(pair_ab.angular_momentum, pair_cd.angular_momentum, sign_side=parity)
    zero_order = np.moveaxis(auxiliary[*indices, 0], 0, -1)
    matrix = np.take(zero_order, sum_indices, axis=-1)

    return matrix, signs
