"""Nuclear attraction operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hermi.boys import boys
from hermi.hermite import hermite_indices, hermite_integral_recursion
from hermi.operator import Operator, check_views_rank
from hermi.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.util import outer_powers, unreachable

if TYPE_CHECKING:
    from hermi.basis import ShellPairView
    from hermi.molecule import Molecule
    from hermi.operator import OperatorMatrix, View
    from hermi.symmetry import Symmetry


@dataclass
class Nuclear(Operator):
    r"""Nuclear attraction operator.

    The nuclear attraction operator is defined as

    .. math::
        \hat{O} = -\sum_C \frac{Z_C}{|r - R_C|}.
    """

    molecule: Molecule

    @property
    def default_signature(self) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return (2, 0)

    @property
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return ((2, 0),)

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

    def _get_operator_matrix_20(self, views: tuple[ShellPairView, None]) -> OperatorMatrix:
        r"""Compute the (2, 0) nuclear attraction operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix. Shape is ``(n_primitive_bra, n_primitive_ket, n_hermite_bra, n_hermite_ket)``,
            where ``n_primitive_bra`` and ``n_primitive_ket`` are the number of primitive Gaussians in the bra and ket
            shells, and ``n_hermite_bra`` and ``n_hermite_ket`` are the number of Hermite Gaussians in the bra and ket
            shells.
        """
        pair = views[0]
        max_order = pair.angular_momentum
        exponent = np.expand_dims(pair.exponent.ravel(), axis=-1)
        displacement = pair.position.reshape(-1, 1, 3) - self.molecule.positions
        argument = exponent * np.einsum("...x,...x->...", displacement, displacement, optimize=True)
        charges = self.molecule.charges

        auxiliary = np.zeros((max_order + 1,) * 4 + (len(pair), len(self.molecule)), dtype=np.float64)

        # Fundamental value
        fundamental = np.moveaxis(outer_powers(-2 * exponent, max_order) * boys(argument, max_order), -1, 0)
        fundamental /= exponent
        fundamental *= 2 * np.pi
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
        zero_order = np.moveaxis(auxiliary[*indices, 0], 0, -1)
        operator = -np.expand_dims(np.einsum("...ah,a->...h", zero_order, charges), axis=(1, 3))

        return operator

    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the nuclear attraction operator matrix in the Hermite basis.

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
        return unreachable()
