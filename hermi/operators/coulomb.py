"""Coulomb operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.boys import boys
from hermi.hermite import hermite_indices, hermite_integral_recursion, hermite_operator_matrix_indices
from hermi.operators.operator import Operator, check_views_rank
from hermi.symmetry import SymmetryN4S1, SymmetryN4S8
from hermi.util import outer_powers, unreachable

if TYPE_CHECKING:
    from hermi.basis import ShellPairView
    from hermi.operators.operator import OperatorMatrix, View
    from hermi.symmetry import Symmetry


@dataclass
class Coulomb(Operator):
    r"""Coulomb operator.

    The Coulomb operator is defined as

    .. math::
        \hat{O} = \frac{1}{|r_1 - r_2|}.
    """

    @property
    def default_signature(self) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (2, 2)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((2, 2),)

    @property
    def default_symmetry(
        self,
    ) -> type[Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]]:
        r"""Default symmetry of the operator."""
        return SymmetryN4S8

    @property
    def supported_symmetries(
        self,
    ) -> tuple[type[Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]], ...]:
        r"""Supported symmetries of the operator."""
        return (
            SymmetryN4S1,
            SymmetryN4S8,
        )

    def _get_operator_matrix_22(self, views: tuple[ShellPairView, ShellPairView]) -> OperatorMatrix:
        r"""Compute the (2, 2) Coulomb operator matrix in the Hermite basis."""
        pair_ab, pair_cd = views
        max_order = pair_ab.angular_momentum + pair_cd.angular_momentum
        displacement = pair_ab.position.reshape(-1, 1, 3) - pair_cd.position.reshape(1, -1, 3)
        exponent_product = pair_ab.exponent.reshape(-1, 1) * pair_cd.exponent.reshape(1, -1)
        exponent_sum = pair_ab.exponent.reshape(-1, 1) + pair_cd.exponent.reshape(1, -1)
        exponent = exponent_product / exponent_sum
        argument = np.einsum("...,...x,...x->...", exponent, displacement, displacement, optimize=True)

        auxiliary = np.zeros((max_order + 1,) * 4 + (len(pair_ab), len(pair_cd)), dtype=np.float64)

        # Fundamental value
        fundamental = np.moveaxis(outer_powers(-2 * exponent, max_order) * boys(argument, max_order), -1, 0)
        fundamental /= exponent_product
        fundamental /= np.sqrt(exponent_sum)
        fundamental *= 2 * (np.pi**2.5)
        auxiliary[0, 0, 0] = fundamental

        # Recursion for Hermite integrals
        hermite_integral_recursion(
            auxiliary,
            max_order,
            displacement[..., 0],
            displacement[..., 1],
            displacement[..., 2],
        )

        # Construct operator matrix
        indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
        sum_indices, signs = hermite_operator_matrix_indices(pair_ab.angular_momentum, pair_cd.angular_momentum)
        zero_order = np.moveaxis(auxiliary[*indices, 0], 0, -1)
        operator = np.take(zero_order, sum_indices, axis=-1)
        operator *= signs  # cheaper to apply to E, but we want operators self-contained

        return operator

    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the Coulomb operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix. Shape is ``(n_primitive_bra, n_primitive_ket, n_hermite_bra, n_hermite_ket)``,
            where ``n_primitive_bra`` and ``n_primitive_ket`` are the number of primitive Gaussians in the bra and ket
            shells, and ``n_hermite_bra`` and ``n_hermite_ket`` are the number of Hermite Gaussians in the bra and ket
            shells.
        """
        if check_views_rank(views, (2, 2)):
            return self._get_operator_matrix_22(views)
        return unreachable()
