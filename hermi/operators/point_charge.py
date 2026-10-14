"""Point-charge operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.integrals.boys import boys
from hermi.integrals.hermite import hermite_auxiliary_layout, hermite_indices, hermite_integral_recursion
from hermi.integrals.symmetry import SymmetryN1S1
from hermi.operators.operator import Operator, check_views_rank
from hermi.utils.linear_algebra import outer_powers
from hermi.utils.misc import unreachable

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array
    from hermi.integrals.symmetry import Symmetry
    from hermi.operators.operator import OperatorMatrix, View
    from hermi.systems.basis import ShellView


@dataclass(frozen=True)
class PointCharge(Operator):
    r"""Point-charge operator.

    The point-charge operator is defined as

    .. math::
        \hat{O} = \frac{1}{|r - r_x|}.
    """

    position: array[float64]

    def get_default_signature(
        self,
        signature: tuple[int, int] | None = None,
        ao_slices: tuple[slice, ...] | None = None,
    ) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (1, 0)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((1, 0),)

    @property
    def supported_symmetries(self) -> tuple[type[Symmetry[tuple[int], tuple[slice]]], ...]:
        r"""Supported symmetries of the operator."""
        return (SymmetryN1S1,)

    def _get_operator_matrix_10(self, views: tuple[ShellView, None]) -> OperatorMatrix:
        r"""Compute the (1, 0) point-charge operator matrix in the Hermite basis."""
        shell = views[0]
        max_order = shell.angular_momentum
        offsets, n_valid = hermite_auxiliary_layout(max_order)
        exponent = shell.exponent
        batched = self.position.ndim == 2
        auxiliary_shape: tuple[int, ...]
        if batched:
            displacement = shell.position[None, ...] - self.position[:, None, :]
            auxiliary_shape = (n_valid, self.position.shape[0], len(shell))
        else:
            displacement = shell.position - self.position
            auxiliary_shape = (n_valid, len(shell))
        argument = exponent * np.einsum("...x,...x->...", displacement, displacement, optimize=True)

        auxiliary = np.empty(auxiliary_shape, dtype=np.float64)

        # Fundamental value
        fundamental = np.moveaxis(outer_powers(-2 * exponent, max_order) * boys(argument, max_order), -1, 0)
        fundamental /= exponent
        fundamental *= 2 * np.pi
        auxiliary[: max_order + 1] = fundamental

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
        zero_order = np.moveaxis(auxiliary[offsets[*indices]], 0, -1)
        operator = np.expand_dims(zero_order, axis=(2, 4) if batched else (1, 3))

        return operator

    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the point-charge operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix with shape ``(n_primitive, 1, n_hermite, 1)`` or
            ``(n_positions, n_primitive, 1, n_hermite, 1)`` for batched positions.
        """
        if check_views_rank(views, (1, 0)):
            return self._get_operator_matrix_10(views)
        return unreachable()
