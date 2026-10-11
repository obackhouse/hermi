"""Kinetic energy operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.integrals.expansion import Expansion
from hermi.integrals.hermite import (
    hermite_indices,
    hermite_integral_recursion_zero_order,
    hermite_operator_matrix_indices,
)
from hermi.integrals.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.operators.operator import Operator, check_views_rank
from hermi.util import unreachable

if TYPE_CHECKING:
    from typing import Any

    from hermi.integrals.expansion import ExpansionMatrix
    from hermi.integrals.symmetry import Symmetry
    from hermi.operators.operator import OperatorMatrix, View
    from hermi.systems.basis import ShellPairView, ShellView


@dataclass(frozen=True)
class Kinetic(Operator):
    r"""Kinetic energy operator.

    The kinetic energy operator is defined as

    .. math::
        \hat{O} = -\frac{1}{2} \nabla^2.
    """

    def get_default_signature(
        self,
        signature: tuple[int, int] | None = None,
        ao_slices: tuple[slice, ...] | None = None,
    ) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (1, 1)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((1, 1), (2, 0))

    @property
    def supported_symmetries(self) -> tuple[type[Symmetry[tuple[int, int], tuple[slice, slice]]], ...]:
        r"""Supported symmetries of the operator."""
        return (
            SymmetryN2S1,
            SymmetryN2S2,
        )

    def _get_operator_matrix_11(self, views: tuple[ShellView, ShellView]) -> OperatorMatrix:
        r"""Compute the (1, 1) kinetic energy operator matrix in the Hermite basis."""
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

    def _get_operator_matrix_20(self, views: tuple[ShellPairView, None]) -> OperatorMatrix:
        r"""Compute the (2, 0) kinetic energy operator matrix in the Hermite basis."""
        pair = views[0]
        max_order = pair.angular_momentum + 2
        indices = hermite_indices(max_order)
        operator = np.zeros((len(pair), 1, len(indices), 1), dtype=np.float64)

        # Fundamental value
        operator[:, :, 0] = -0.5 * np.expand_dims(np.pi / pair.exponent.ravel(), axis=(1, 2)) ** 1.5

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
        elif check_views_rank(views, (2, 0)):
            return self._get_operator_matrix_20(views)
        return unreachable()

    def _get_expansion_matrices_20(
        self, views: tuple[ShellPairView, None], **kwargs: Any
    ) -> tuple[ExpansionMatrix, None]:
        r"""Get the Hermite expansion matrix for the shells for a (2, 0) operator."""
        pair = views[0]
        expansion = Expansion(pair)
        differentiated = [
            expansion.differentiate(axis, side="ket").differentiate(axis, side="ket") for axis in range(3)
        ]
        summed = differentiated[0] + differentiated[1] + differentiated[2]
        expansion_matrix = summed.compute(**kwargs)
        return expansion_matrix, None

    def _get_expansion_matrices_11(
        self, views: tuple[ShellView, ShellView], **kwargs: Any
    ) -> tuple[ExpansionMatrix, ExpansionMatrix]:
        r"""Get the Hermite expansion matrix for the shells for a (1, 1) operator."""
        pair = views[0].basis[views[0], views[1]]
        return Expansion(pair.shell_i).compute(**kwargs), Expansion(pair.shell_j).compute(**kwargs)

    def get_expansion_matrices(
        self, views: tuple[View, View], **kwargs: Any
    ) -> tuple[ExpansionMatrix | None, ExpansionMatrix | None]:
        r"""Get the Hermite expansion matrix for the shells.

        Args:
            views: Views of the shells to get the expansion matrix for.
            kwargs: Additional keyword arguments to pass to the expansion matrix function.

        Returns:
            Hermite expansion matrices for the shells. Shape is ``(n_primitive, n_hermite, n_cartesian)``, where
            ``n_primitive`` is the number of primitive Gaussians in the shell, ``n_hermite`` is the number of Hermite
            Gaussians in the shell, and ``n_cartesian`` is the number of Cartesian Gaussians in the shell.
        """
        if check_views_rank(views, (2, 0)):
            return self._get_expansion_matrices_20(views, **kwargs)
        elif check_views_rank(views, (1, 1)):
            return self._get_expansion_matrices_11(views, **kwargs)
        return unreachable()
