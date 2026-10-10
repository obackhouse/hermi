"""Operator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, overload

import numpy as np

from hermi.basis import ShellPairView, ShellView
from hermi.collect import collect_integral_11, collect_integral_20, collect_integral_22
from hermi.contract import contract_operator_11, contract_operator_20, contract_operator_22
from hermi.expansion import Expansion

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy import float64
    from typing_extensions import TypeIs

    from hermi._typing import array
    from hermi.basis import Basis
    from hermi.expansion import ExpansionMatrix
    from hermi.symmetry import Symmetry

    type OperatorMatrix = array[tuple[int, int, int, int], float64] | array[tuple[int, int, int, int, int], float64]
    type View = None | ShellView | ShellPairView


@overload
def check_view_rank(view: None, rank: Literal[0]) -> TypeIs[None]: ...
@overload
def check_view_rank(view: ShellView, rank: Literal[1]) -> TypeIs[ShellView]: ...
@overload
def check_view_rank(view: ShellPairView, rank: Literal[2]) -> TypeIs[ShellPairView]: ...


def check_view_rank[N: int](view: View, rank: N) -> bool:
    """Check that the view has the expected rank."""
    if rank == 0:
        return view is None
    elif rank == 1:
        return isinstance(view, ShellView)
    elif rank == 2:
        return isinstance(view, ShellPairView)
    raise ValueError(f"invalid rank {rank}")


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


def _stack_operator_matrices(operator_matrices: list[OperatorMatrix]) -> OperatorMatrix:
    """Pad Hermite axes and stack operator matrices along a batch axis."""
    max_bra_order = max(operator_matrix.shape[-2] for operator_matrix in operator_matrices)
    max_ket_order = max(operator_matrix.shape[-1] for operator_matrix in operator_matrices)
    padded = [
        np.pad(
            operator_matrix,
            [(0, 0)] * (operator_matrix.ndim - 2)
            + [(0, max_bra_order - operator_matrix.shape[-2]), (0, max_ket_order - operator_matrix.shape[-1])],
        )
        for operator_matrix in operator_matrices
    ]
    return np.stack(padded, axis=0)


def _stack_expansion_matrices(
    expansion_matrices: tuple[ExpansionMatrix | None, ...],
) -> ExpansionMatrix | None:
    """Pad Hermite axes and stack different expansion matrices along a batch axis."""
    if all(expansion_matrix is None for expansion_matrix in expansion_matrices):
        return None
    if any(expansion_matrix is None for expansion_matrix in expansion_matrices):
        raise ValueError("operators with the same signature must use compatible expansion ranks")

    expansions = [expansion_matrix for expansion_matrix in expansion_matrices if expansion_matrix is not None]
    if all(expansion_matrix is expansions[0] for expansion_matrix in expansions):
        return np.expand_dims(expansions[0], axis=0)

    max_order = max(expansion_matrix.shape[-1] for expansion_matrix in expansions)
    padded = [
        np.pad(expansion_matrix, [(0, 0)] * (expansion_matrix.ndim - 1) + [(0, max_order - expansion_matrix.shape[-1])])
        for expansion_matrix in expansions
    ]
    return np.stack(padded, axis=0)


@dataclass(frozen=True)
class Operator(ABC):
    """Base class for operators."""

    @staticmethod
    def views_shape(views: tuple[View, View]) -> tuple[int, int]:
        """Get the number of shells in the bra and ket views."""
        return (0 if views[0] is None else views[0].rank, 0 if views[1] is None else views[1].rank)

    @staticmethod
    def views_rank(views: tuple[View, View]) -> int:
        """Get the total number of shells in the bra and ket views."""
        return (0 if views[0] is None else views[0].rank) + (0 if views[1] is None else views[1].rank)

    @abstractmethod
    def get_default_signature(self, split_basis: bool = False) -> tuple[int, int]:
        r"""Default signature of the operator."""
        pass

    @abstractmethod
    def get_supported_signatures(self, split_basis: bool = False) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        pass

    @abstractmethod
    def get_default_symmetry(self, split_basis: bool = False) -> type[Symmetry[Any, Any]]:
        r"""Default symmetry of the operator."""
        pass

    @abstractmethod
    def get_supported_symmetries(self, split_basis: bool = False) -> tuple[type[Symmetry[Any, Any]], ...]:
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
        expansion_a = Expansion(views[0]).compute(**kwargs) if views[0] is not None else None
        expansion_b = Expansion(views[1]).compute(**kwargs) if views[1] is not None else None
        return expansion_a, expansion_b

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
        expansion: tuple[ExpansionMatrix | None, ExpansionMatrix | None] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, ...], float64]:
        r"""Contract the operator matrix with the expansion coefficients and reduce primitives.

        Args:
            views: Views of the shells to compute the operator matrix for.
            operator: Operator matrix to contract.
            expansion: Expansion matrices to contract.
            spherical: Return real spherical instead of Cartesian integrals.

        Returns:
            Contracted operator matrix.
        """
        if operator is None:
            operator = self.get_operator_matrix(views)
        if expansion is None:
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
        basis: Basis | tuple[Basis, Basis],
        signature: tuple[Literal[0], Literal[1]] | tuple[Literal[1], Literal[0]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis | tuple[Basis, Basis],
        signature: tuple[Literal[0], Literal[2]] | tuple[Literal[2], Literal[0]] | tuple[Literal[1], Literal[1]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis | tuple[Basis, Basis],
        signature: tuple[Literal[1], Literal[2]] | tuple[Literal[2], Literal[1]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int], float64]: ...
    @overload
    def collect(
        self,
        basis: Basis | tuple[Basis, Basis],
        signature: tuple[Literal[2], Literal[2]],
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, int, int, int], float64]: ...

    def collect(
        self,
        basis: Basis | tuple[Basis, Basis],
        signature: tuple[int, int] | None = None,
        symmetry_type: type[Symmetry[Any, Any]] | None = None,
        *,
        spherical: bool = False,
    ) -> array[tuple[int, ...], float64]:
        r"""Collect contracted AO integrals for the operator.

        Args:
            basis: Basis set to collect integrals for. Can be a pair of basis sets in which case the
                first basis set is used for the bra and the second for the ket.
            signature: Signature of the operator to collect integrals for.
            symmetry_type: Symmetry helper class for the integral matrix.
            spherical: Return real spherical instead of Cartesian integrals.

        Returns:
            Contracted Cartesian or real spherical AO integrals.
        """
        split_basis = isinstance(basis, tuple) and basis[0] is not basis[1]
        if signature is None:
            signature = self.get_default_signature(split_basis=split_basis)
        if symmetry_type is None:
            symmetry_type = self.get_default_symmetry(split_basis=split_basis)

        if signature not in self.get_supported_signatures(split_basis=split_basis):
            raise ValueError(f"operator signature {signature} not supported by operator {self.__class__.__name__}")
        if symmetry_type not in self.get_supported_symmetries(split_basis=split_basis):
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

    def __add__(self, other: object) -> Operator:
        if not isinstance(other, Operator):
            return NotImplemented
        left = self.operators if isinstance(self, OperatorSum) else (self,)
        right = other.operators if isinstance(other, OperatorSum) else (other,)
        return OperatorSum(left + right)

    def __radd__(self, other: object) -> Operator:
        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Operator):
            return NotImplemented
        left = other.operators if isinstance(other, OperatorSum) else (other,)
        right = self.operators if isinstance(self, OperatorSum) else (self,)
        return OperatorSum(left + right)

    def __mul__(self, other: object) -> Operator:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return OperatorScaled(self, other)

    def __rmul__(self, other: object) -> Operator:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return OperatorScaled(self, other)


@dataclass(frozen=True)
class OperatorSum(Operator):
    """Operator sum."""

    operators: tuple[Operator, ...]

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError("operator sum must contain at least one operator")
        if len(self.get_supported_signatures()) == 0:
            raise ValueError("operator sum must contain operators with compatible signatures")

    def get_default_signature(self, split_basis: bool = False) -> tuple[int, int]:
        r"""Default signature of the operator."""
        default_signatures: dict[tuple[int, int], int] = {}
        supported_signatures = self.get_supported_signatures(split_basis=split_basis)
        for operator in self.operators:
            default_signature = operator.get_default_signature(split_basis=split_basis)
            if default_signature not in supported_signatures:
                continue
            default_signatures[default_signature] = default_signatures.get(default_signature, 0) + 1
        if len(default_signatures) == 0:
            if len(supported_signatures) == 0:
                raise ValueError("operator sum must contain operators with compatible signatures")
            return supported_signatures[0]
        return max(default_signatures, key=default_signatures.__getitem__)

    def get_supported_signatures(self, split_basis: bool = False) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return tuple(
            set.intersection(
                *(set(operator.get_supported_signatures(split_basis=split_basis)) for operator in self.operators)
            )
        )

    def get_default_symmetry(self, split_basis: bool = False) -> type[Symmetry[Any, Any]]:
        r"""Default symmetry of the operator."""
        default_symmetries: dict[type[Symmetry[Any, Any]], int] = {}
        supported_symmetries = self.get_supported_symmetries(split_basis=split_basis)
        for operator in self.operators:
            default_symmetry = operator.get_default_symmetry(split_basis=split_basis)
            if default_symmetry not in supported_symmetries:
                continue
            default_symmetries[default_symmetry] = default_symmetries.get(default_symmetry, 0) + 1
        if len(default_symmetries) == 0:
            if len(supported_symmetries) == 0:
                raise ValueError("operator sum must contain operators with compatible symmetries")
            return supported_symmetries[0]
        return max(default_symmetries, key=default_symmetries.__getitem__)

    def get_supported_symmetries(self, split_basis: bool = False) -> tuple[type[Symmetry[Any, Any]], ...]:
        r"""Supported symmetries of the operator."""
        return tuple(
            set.intersection(
                *(set(operator.get_supported_symmetries(split_basis=split_basis)) for operator in self.operators)
            )
        )

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
        operator_matrices = [operator.get_operator_matrix(views) for operator in self.operators]

        # See get_expansion_matrices comment. This *should* be cheap due to caching, but we should
        # make this more rigorous.
        expansion_matrices = [operator.get_expansion_matrices(views) for operator in self.operators]
        expansion_matrices_bra, expansion_matrices_ket = zip(*expansion_matrices)
        same_expansion = all(
            (
                *[expansion_matrix_bra is expansion_matrices_bra[0] for expansion_matrix_bra in expansion_matrices_bra],
                *[expansion_matrix_ket is expansion_matrices_ket[0] for expansion_matrix_ket in expansion_matrices_ket],
            )
        )
        if same_expansion:
            operator_matrix = reduce(np.add, operator_matrices)
        else:
            operator_matrix = _stack_operator_matrices(operator_matrices)

        return operator_matrix

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
        expansion_matrices = [operator.get_expansion_matrices(views, **kwargs) for operator in self.operators]
        expansion_matrices_bra, expansion_matrices_ket = zip(*expansion_matrices)

        # If all expansion matrices on one side are the same (e.g. they have the same ID due to
        # caching), we can return a single matrix with a leading dimension of 1. The contraction
        # code can either broadcast against the stacked operators, or sum the operators.
        expansion_matrix = (
            _stack_expansion_matrices(expansion_matrices_bra),
            _stack_expansion_matrices(expansion_matrices_ket),
        )

        return expansion_matrix


@dataclass(frozen=True)
class OperatorScaled(Operator):
    """Scaled operator."""

    operator: Operator
    scale: float

    def get_default_signature(self, split_basis: bool = False) -> tuple[int, int]:
        r"""Default signature of the operator."""
        return self.operator.get_default_signature(split_basis=split_basis)

    def get_supported_signatures(self, split_basis: bool = False) -> tuple[tuple[int, int], ...]:
        r"""Supported signatures of the operator."""
        return self.operator.get_supported_signatures(split_basis=split_basis)

    def get_default_symmetry(self, split_basis: bool = False) -> type[Symmetry[Any, Any]]:
        r"""Default symmetry of the operator."""
        return self.operator.get_default_symmetry(split_basis=split_basis)

    def get_supported_symmetries(self, split_basis: bool = False) -> tuple[type[Symmetry[Any, Any]], ...]:
        r"""Supported symmetries of the operator."""
        return self.operator.get_supported_symmetries(split_basis=split_basis)

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
        operator_matrix = self.operator.get_operator_matrix(views)
        return self.scale * operator_matrix

    def get_expansion_matrices(
        self, views: tuple[View, View], **kwargs: Any
    ) -> tuple[ExpansionMatrix | None, ExpansionMatrix | None]:
        """Get the expansion matrices used by the wrapped operator."""
        return self.operator.get_expansion_matrices(views, **kwargs)
