"""Coulomb operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.integrals.boys import boys
from hermi.integrals.hermite import hermite_indices, hermite_integral_recursion, hermite_operator_matrix_indices
from hermi.integrals.symmetry import (
    SymmetryN2S1,
    SymmetryN2S2,
    SymmetryN3S1,
    SymmetryN3S2Ket,
    SymmetryN4S1,
    SymmetryN4S4,
    SymmetryN4S8,
)
from hermi.operators.operator import Operator, check_views_rank
from hermi.utils.linear_algebra import outer_powers
from hermi.utils.misc import unreachable

if TYPE_CHECKING:
    from typing import Any

    from numpy import float64

    from hermi._typing import array
    from hermi.integrals.symmetry import Symmetry
    from hermi.operators.operator import OperatorMatrix, View
    from hermi.systems.basis import ShellPairView, ShellView


@dataclass(frozen=True)
class Coulomb(Operator):
    r"""Coulomb operator.

    The Coulomb operator is defined as

    .. math::
        \hat{O} = \frac{1}{|r_1 - r_2|}.
    """

    def get_default_signature(
        self,
        signature: tuple[int, int] | None = None,
        ao_slices: tuple[slice, ...] | None = None,
    ) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (2, 2)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((1, 1), (1, 2), (2, 2))

    @property
    def supported_symmetries(
        self,
    ) -> tuple[type[Symmetry[Any, Any]], ...]:
        r"""Supported symmetries of the operator."""
        return (
            SymmetryN2S1,
            SymmetryN2S2,
            SymmetryN3S1,
            SymmetryN3S2Ket,
            SymmetryN4S1,
            SymmetryN4S4,
            SymmetryN4S8,
        )

    def _get_operator_matrix(
        self,
        exponent_a: array[float64],
        position_a: array[float64],
        angular_momentum_a: int,
        exponent_b: array[float64],
        position_b: array[float64],
        angular_momentum_b: int,
    ) -> OperatorMatrix:
        """Compute the Coulomb Hermite matrix for two Gaussian distributions."""
        max_order = angular_momentum_a + angular_momentum_b
        displacement = position_a.reshape(-1, 1, 3) - position_b.reshape(1, -1, 3)
        exponent_product = exponent_a.reshape(-1, 1) * exponent_b.reshape(1, -1)
        exponent_sum = exponent_a.reshape(-1, 1) + exponent_b.reshape(1, -1)
        exponent = exponent_product / exponent_sum
        argument = np.einsum("...,...x,...x->...", exponent, displacement, displacement, optimize=True)

        auxiliary = np.zeros((max_order + 1,) * 4 + (exponent_a.size, exponent_b.size), dtype=np.float64)

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
        sum_indices, signs = hermite_operator_matrix_indices(angular_momentum_a, angular_momentum_b)
        operator = auxiliary[*indices, 0][sum_indices]
        operator = operator.transpose(2, 3, 0, 1)  # p q ab cd -> ab cd p q
        operator *= signs  # cheaper to apply to E, but we want operators self-contained

        return operator  # type: ignore[no-any-return]

    def _get_operator_matrix_11(self, views: tuple[ShellView, ShellView]) -> OperatorMatrix:
        r"""Compute the (1, 1) Coulomb operator matrix in the Hermite basis."""
        shell_a, shell_b = views
        return self._get_operator_matrix(
            shell_a.exponent,
            shell_a.position,
            shell_a.angular_momentum,
            shell_b.exponent,
            shell_b.position,
            shell_b.angular_momentum,
        )

    def _get_operator_matrix_12(self, views: tuple[ShellView, ShellPairView]) -> OperatorMatrix:
        r"""Compute the (1, 2) Coulomb operator matrix in the Hermite basis."""
        shell_a, pair_bc = views
        return self._get_operator_matrix(
            shell_a.exponent,
            shell_a.position,
            shell_a.angular_momentum,
            pair_bc.exponent.ravel(),
            pair_bc.position.reshape(-1, 3),
            pair_bc.angular_momentum,
        )

    def _get_operator_matrix_22(self, views: tuple[ShellPairView, ShellPairView]) -> OperatorMatrix:
        r"""Compute the (2, 2) Coulomb operator matrix in the Hermite basis."""
        pair_ab, pair_cd = views
        return self._get_operator_matrix(
            pair_ab.exponent.ravel(),
            pair_ab.position.reshape(-1, 3),
            pair_ab.angular_momentum,
            pair_cd.exponent.ravel(),
            pair_cd.position.reshape(-1, 3),
            pair_cd.angular_momentum,
        )

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
        if check_views_rank(views, (1, 1)):
            return self._get_operator_matrix_11(views)
        elif check_views_rank(views, (1, 2)):
            return self._get_operator_matrix_12(views)
        elif check_views_rank(views, (2, 2)):
            return self._get_operator_matrix_22(views)
        return unreachable()
