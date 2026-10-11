"""Overlap operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.hermite import hermite_indices, hermite_integral_recursion_zero_order, hermite_operator_matrix_indices
from hermi.operators.operator import Operator, check_views_rank
from hermi.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.util import unreachable

if TYPE_CHECKING:
    from hermi.basis import ShellPairView, ShellView
    from hermi.operators.operator import OperatorMatrix, View
    from hermi.symmetry import Symmetry


@dataclass(frozen=True)
class Overlap(Operator):
    r"""Overlap operator.

    The overlap operator is defined as

    .. math::
        \hat{O} = 1.
    """

    def get_default_signature(
        self,
        signature: tuple[int, int] | None = None,
        ao_slices: tuple[slice, ...] | None = None,
    ) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (2, 0)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((2, 0), (1, 1))

    @property
    def supported_symmetries(self) -> tuple[type[Symmetry[tuple[int, int], tuple[slice, slice]]], ...]:
        r"""Supported symmetries of the operator."""
        return (
            SymmetryN2S1,
            SymmetryN2S2,
        )

    def _get_operator_matrix_20(self, views: tuple[ShellPairView, None]) -> OperatorMatrix:
        r"""Compute the (2, 0) overlap operator matrix in the Hermite basis."""
        # TODO: implement rank-deficient contraction
        pair = views[0]
        max_order = pair.angular_momentum
        exponent = pair.exponent
        indices = hermite_indices(max_order)

        operator = np.zeros((len(views[0]), 1, len(indices), 1), dtype=np.float64)

        # Fundamental value
        operator[:, :, 0] = np.expand_dims(np.pi / exponent.ravel(), axis=(1, 2)) ** 1.5

        return operator

    def _get_operator_matrix_11(self, views: tuple[ShellView, ShellView]) -> OperatorMatrix:
        r"""Compute the (1, 1) overlap operator matrix in the Hermite basis."""
        pair = views[0].basis[views[0], views[1]]
        max_order = pair.angular_momentum
        displacement = pair.vector
        reduced_exponent = pair.reduced_exponent
        argument = np.einsum("...,...x,...x->...", reduced_exponent, displacement, displacement, optimize=True)
        scale = -2 * reduced_exponent

        auxiliary = np.zeros((max_order + 1,) * 3 + reduced_exponent.shape, dtype=np.float64)

        # Fundamental value
        fundamental = np.exp(-argument)
        fundamental /= pair.exponent**1.5
        fundamental *= np.pi**1.5
        auxiliary[0, 0, 0] = fundamental

        # Recursion for Hermite integrals
        hermite_integral_recursion_zero_order(
            auxiliary,
            max_order,
            displacement[..., 0],
            displacement[..., 1],
            displacement[..., 2],
            scale,
        )

        # Construct operator matrix
        indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
        sum_indices, signs = hermite_operator_matrix_indices(
            pair.shell_i.angular_momentum, pair.shell_j.angular_momentum
        )
        derivatives = np.moveaxis(auxiliary[*indices], 0, -1)
        operator = np.take(derivatives, sum_indices, axis=-1)
        operator *= signs

        return operator

    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the overlap operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix. Shape is ``(n_primitive_bra, n_primitive_ket, n_hermite_bra, n_hermite_ket)``,
            where ``n_primitive_bra`` and ``n_primitive_ket`` are the number of primitive Gaussians in the bra and ket
            shells, and ``n_hermite_bra`` and ``n_hermite_ket`` are the number of Hermite Gaussians in the bra and ket
            shells.
        """
        if check_views_rank(views, (2, 0)):
            return self._get_operator_matrix_20(views)
        elif check_views_rank(views, (1, 1)):
            return self._get_operator_matrix_11(views)
        return unreachable()
