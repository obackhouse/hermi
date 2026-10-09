"""Kinetic energy operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.hermite import hermite_indices, hermite_integral_recursion_zero_order, hermite_operator_matrix_indices
from hermi.operator import Operator, check_views_rank
from hermi.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.util import unreachable

if TYPE_CHECKING:
    from hermi.basis import ShellView
    from hermi.operator import OperatorMatrix, View
    from hermi.symmetry import Symmetry


@dataclass
class Kinetic(Operator):
    r"""Kinetic energy operator.

    The kinetic energy operator is defined as

    .. math::
        \hat{O} = -\frac{1}{2} \nabla^2.
    """

    @property
    def default_signature(self) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (1, 1)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((1, 1),)

    @property
    def default_symmetry(self) -> type[Symmetry[tuple[int, int], tuple[slice, slice]]]:
        r"""Default symmetry of the operator."""
        return SymmetryN2S2

    @property
    def supported_symmetries(self) -> tuple[type[Symmetry[tuple[int, int], tuple[slice, slice]]], ...]:
        r"""Supported symmetries of the operator."""
        return (
            SymmetryN2S1,
            SymmetryN2S2,
        )

    def _get_operator_matrix_11(self, views: tuple[ShellView, ShellView]) -> OperatorMatrix:
        r"""Compute the (1, 1) kinetic energy operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix. Shape is ``(n_primitive_bra, n_primitive_ket, n_hermite_bra, n_hermite_ket)``,
            where ``n_primitive_bra`` and ``n_primitive_ket`` are the number of primitive Gaussians in the bra and ket
            shells, and ``n_hermite_bra`` and ``n_hermite_ket`` are the number of Hermite Gaussians in the bra and ket
            shells.
        """
        pair = views[0].basis[views[0], views[1]]
        max_order = pair.angular_momentum + 2
        displacement = pair.vector
        reduced_exponent = pair.reduced_exponent
        argument = np.einsum("...,...x,...x->...", reduced_exponent, displacement, displacement, optimize=True)
        scale = -2 * reduced_exponent

        auxiliary = np.zeros((max_order + 1,) * 3 + reduced_exponent.shape, dtype=np.float64)

        # Fundamental value
        fundamental = np.exp(-argument)
        fundamental /= pair.exponent**1.5
        fundamental *= np.pi**1.5
        fundamental *= -0.5
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
            pair.shell_i.angular_momentum,
            pair.shell_j.angular_momentum,
            shifts=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        )
        derivatives = np.moveaxis(auxiliary[*indices], 0, -1)
        operator = np.sum(np.take(derivatives, sum_indices, axis=-1), axis=-3)
        operator *= signs

        return operator

    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the kinetic energy operator matrix in the Hermite basis.

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
        return unreachable()
