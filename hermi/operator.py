"""Operator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from hermi.basis import ShellPairView, ShellView
from hermi.collect import collect_integral_11, collect_integral_20, collect_integral_22
from hermi.contract import contract_operator_11, contract_operator_20, contract_operator_22
from hermi.hermite import expansion_matrix, expansion_pair_matrix

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy import float64
    from typing_extensions import TypeIs

    from hermi._typing import array
    from hermi.basis import Basis
    from hermi.symmetry import Symmetry

    type OperatorMatrix = array[tuple[int, int, int, int], float64] | array[tuple[int, int, int, int, int], float64]
    type ExpansionMatrix = array[tuple[int, int, int], float64] | array[tuple[int, int, int, int], float64]
    type View = None | ShellView | ShellPairView


@overload
def _get_expansion_matrix(view: None, spherical: bool = False) -> None: ...
@overload
def _get_expansion_matrix(view: ShellView | ShellPairView, spherical: bool = False) -> ExpansionMatrix: ...


def _get_expansion_matrix(view: View, spherical: bool = False) -> ExpansionMatrix | None:
    """Get the Hermite expansion matrix for a view."""
    if isinstance(view, ShellView):
        return expansion_matrix(view, spherical=spherical)
    elif isinstance(view, ShellPairView):
        return expansion_pair_matrix(view, spherical=spherical)
    return None


@overload
def check_views_rank(views: tuple[View, View], rank: tuple[Literal[0], Literal[0]]) -> TypeIs[tuple[None, None]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[0], Literal[1]]
) -> TypeIs[tuple[None, ShellView]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[0], Literal[2]]
) -> TypeIs[tuple[None, ShellPairView]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[1], Literal[0]]
) -> TypeIs[tuple[ShellView, None]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[1], Literal[1]]
) -> TypeIs[tuple[ShellView, ShellView]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[1], Literal[2]]
) -> TypeIs[tuple[ShellView, ShellPairView]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[2], Literal[0]]
) -> TypeIs[tuple[ShellPairView, None]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[2], Literal[1]]
) -> TypeIs[tuple[ShellPairView, ShellView]]: ...
@overload
def check_views_rank(
    views: tuple[View, View], rank: tuple[Literal[2], Literal[2]]
) -> TypeIs[tuple[ShellPairView, ShellPairView]]: ...


def check_views_rank[NBra: int, NKet: int](views: tuple[View, View], rank: tuple[NBra, NKet]) -> bool:
    """Check that the views have the expected rank."""
    check = True
    for view, r in zip(views, rank):
        if r == 0:
            check &= view is None
        elif r == 1:
            check &= isinstance(view, ShellView)
        elif r == 2:
            check &= isinstance(view, ShellPairView)
    return check


@dataclass
class Operator(ABC):
    r"""Base class for operators."""

    @staticmethod
    def views_shape(views: tuple[View, View]) -> tuple[int, int]:
        """Get the number of shells in the bra and ket views."""
        return (0 if views[0] is None else views[0].rank, 0 if views[1] is None else views[1].rank)

    @staticmethod
    def views_rank(views: tuple[View, View]) -> int:
        """Get the total number of shells in the bra and ket views."""
        return (0 if views[0] is None else views[0].rank) + (0 if views[1] is None else views[1].rank)

    @property
    @abstractmethod
    def default_signature(self) -> tuple[int, int]:
        r"""Default signature of the operator."""
        pass

    @property
    @abstractmethod
    def supported_signatures(self) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        pass

    @property
    @abstractmethod
    def default_symmetry(self) -> type[Symmetry[Any, Any]]:
        r"""Default symmetry of the operator."""
        pass

    @property
    @abstractmethod
    def supported_symmetries(self) -> tuple[type[Symmetry[Any, Any]], ...]:
        r"""Supported symmetries of the operator."""
        pass

    @abstractmethod
    def get_operator_matrix(self, views: tuple[View, View]) -> OperatorMatrix:
        r"""Compute the operator matrix in the Hermite basis.

        Args:
            views: Views of the shells to compute the operator matrix for.

        Returns:
            Hermite operator matrix. Shape is ``(n_primitive_bra, n_primitive_ket, n_hermite_bra, n_hermite_ket)``,
            where ``n_primitive_bra`` and ``n_primitive_ket`` are the number of primitive Gaussians in the bra and ket
            shells, and ``n_hermite_bra`` and ``n_hermite_ket`` are the number of Hermite Gaussians in the bra and ket
            shells.
        """
        pass

    @overload
    def get_expansion_matrices(self, views: tuple[None, None], spherical: bool = False) -> tuple[None, None]: ...
    @overload
    def get_expansion_matrices(
        self, views: tuple[ShellView | ShellPairView, None], spherical: bool = False
    ) -> tuple[ExpansionMatrix, None]: ...
    @overload
    def get_expansion_matrices(
        self, views: tuple[None, ShellView | ShellPairView], spherical: bool = False
    ) -> tuple[None, ExpansionMatrix]: ...
    @overload
    def get_expansion_matrices(
        self, views: tuple[ShellView | ShellPairView, ShellView | ShellPairView], spherical: bool = False
    ) -> tuple[ExpansionMatrix, ExpansionMatrix]: ...
    @overload
    def get_expansion_matrices(
        self, views: tuple[View, View], spherical: bool = False
    ) -> tuple[ExpansionMatrix | None, ExpansionMatrix | None]: ...

    def get_expansion_matrices(
        self, views: tuple[View, View], spherical: bool = False
    ) -> tuple[ExpansionMatrix | None, ExpansionMatrix | None]:
        r"""Get the Hermite expansion matrices for the bra and ket shells.

        Args:
            views: Views of the shells to get the expansion matrices for.
            spherical: Return real spherical instead of Cartesian integrals.

        Returns:
            Hermite expansion matrices for the bra and ket shells. Shape is ``(n_primitive, n_hermite, n_cartesian)``,
            where ``n_primitive`` is the number of primitive Gaussians in the shell, ``n_hermite`` is the number of
            Hermite Gaussians in the shell, and ``n_cartesian`` is the number of Cartesian Gaussians in the shell.
        """
        return (
            _get_expansion_matrix(views[0], spherical=spherical),
            _get_expansion_matrix(views[1], spherical=spherical),
        )

    @overload
    def contract(
        self,
        views: tuple[ShellView, None],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[None, ShellView],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[ShellView, ShellView],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[ShellPairView, None],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[None, ShellPairView],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[ShellView, ShellPairView] | tuple[ShellPairView, ShellView],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int], float64]: ...
    @overload
    def contract(
        self,
        views: tuple[ShellPairView, ShellPairView],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int, int], float64]: ...

    def contract(
        self,
        views: tuple[View, View],
        operator: OperatorMatrix | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, ...], float64]:
        r"""Contract the operator matrix with the expansion coefficients and reduce primitives.

        Args:
            views: Views of the shells to compute the operator matrix for.
            operator: Operator matrix to contract.
            spherical: Return real spherical instead of Cartesian integrals.

        Returns:
            Contracted operator matrix.
        """
        if operator is None:
            operator = self.get_operator_matrix(views)
        expansion = self.get_expansion_matrices(views, spherical=spherical)

        if check_views_rank(views, (1, 1)):
            return contract_operator_11(operator, *views, *expansion, spherical=spherical)
        elif check_views_rank(views, (2, 2)):
            return contract_operator_22(operator, *views, *expansion, spherical=spherical)
        elif check_views_rank(views, (2, 0)):
            return contract_operator_20(operator, views[0], expansion[0], spherical=spherical)
        raise NotImplementedError(
            f"contraction for operator {self.__class__.__name__} with views {views} not implemented."
        )

    @overload
    def collect(
        self,
        basis: Basis,
        signature: tuple[Literal[0], Literal[1]] | tuple[Literal[1], Literal[0]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis,
        signature: tuple[Literal[0], Literal[2]] | tuple[Literal[2], Literal[0]] | tuple[Literal[1], Literal[1]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis,
        signature: tuple[Literal[1], Literal[2]] | tuple[Literal[2], Literal[1]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis,
        signature: tuple[Literal[2], Literal[2]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int, int], float64]: ...

    def collect(
        self,
        basis: Basis,
        signature: tuple[int, int] | None = None,
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, ...], float64]:
        r"""Collect contracted AO integrals for the operator.

        Args:
            basis: Basis set to collect integrals for.
            signature: Signature of the operator to collect integrals for.
            symmetry_type: Symmetry helper class for the integral matrix.
            spherical: Return real spherical instead of Cartesian integrals.

        Returns:
            Contracted Cartesian or real spherical AO integrals.
        """
        if signature is None:
            signature = self.default_signature
        if symmetry_type is None:
            symmetry_type = self.default_symmetry

        if signature not in self.supported_signatures:
            raise ValueError(f"operator signature {signature} not supported by operator {self.__class__.__name__}")
        if symmetry_type not in self.supported_symmetries:
            raise ValueError(
                f"operator symmetry {symmetry_type.__name__} not supported by operator {self.__class__.__name__}"
            )

        if signature == (1, 1):
            return collect_integral_11(basis, self, symmetry_type=symmetry_type, spherical=spherical)
        elif signature == (2, 2):
            return collect_integral_22(basis, self, symmetry_type=symmetry_type, spherical=spherical)
        elif signature == (2, 0):
            return collect_integral_20(basis, self, symmetry_type=symmetry_type, spherical=spherical)
        raise NotImplementedError(f"collection for operator signature {signature} not implemented.")
