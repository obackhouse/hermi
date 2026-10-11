"""Hermite expansions."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from hermi.integrals.hermite import hermite_indices
from hermi.systems.basis import ShellPairView, ShellView
from hermi.util import cartesian_indices, cartesian_normalization, cartesian_to_spherical

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy import float64

    from hermi._typing import array

    type ExpansionMatrix = array[tuple[int, int, int], float64] | array[tuple[int, int, int, int], float64]
    type WeightArray = array[tuple[int, int, int], float64]
    type WeightMap = Mapping[tuple[Literal["bra", "ket"], int], WeightArray]


def _identity_weights(shell: ShellView) -> WeightArray:
    """Return weights mapping a shell axis to itself."""
    identity = np.eye(shell.angular_momentum + 1, dtype=np.float64)
    return np.broadcast_to(identity, (len(shell), *identity.shape))


def _differentiate_weights(weights: WeightArray, shell: ShellView) -> WeightArray:
    """Return weights after differentiating a Gaussian polynomial."""
    updated = np.zeros((*weights.shape[:-1], weights.shape[-1] + 1))
    factor = -2.0 * np.expand_dims(shell.exponent, axis=(-1, -2))
    if weights.shape[-1] > 1:
        powers = np.arange(1, weights.shape[-1])
        updated[..., :-2] += weights[..., 1:] * powers
    updated[..., 1:] += weights * factor
    return updated


def _multiply_coordinate_weights(
    weights: WeightArray,
    shell: ShellView,
    axis: int,
    origin: float,
) -> WeightArray:
    """Return weights after multiplying a Gaussian polynomial by a coordinate."""
    updated = np.zeros((*weights.shape[:-1], weights.shape[-1] + 1), dtype=np.float64)
    offset = np.expand_dims(shell.position[:, axis], axis=(-1, -2)) - origin
    updated[..., :-1] += weights * offset
    updated[..., 1:] += weights
    return updated


def _weight_shift(weights: WeightArray | None) -> int:
    """Return the angular-momentum shift encoded by a weight matrix."""
    if weights is None:
        return 0
    return weights.shape[-1] - weights.shape[-2]


def _plans_max_order(view: ShellView | ShellPairView, plans: tuple[WeightMap, ...]) -> int:
    """Return the maximum Hermite order required by a set of weight plans."""
    shifts = (
        sum(_weight_shift(plan.get((side, axis))) for side in ("bra", "ket") for axis in range(3)) for plan in plans
    )
    return view.angular_momentum + max(shifts, default=0)


def _shell_for_side(view: ShellView | ShellPairView, side: Literal["bra", "ket"]) -> ShellView:
    """Return the shell associated with a weight-map side."""
    if isinstance(view, ShellView):
        return view
    if side == "bra":
        return view.shell_i
    if side == "ket":
        return view.shell_j
    raise ValueError(f"invalid side: {side!r}")


def _weight_side(view: ShellView | ShellPairView, side: Literal["bra", "ket"]) -> Literal["bra", "ket"]:
    """Return the weight-map side for a shell or shell-pair view."""
    return "ket" if isinstance(view, ShellView) else side


class ExpansionTree[V: ShellView | ShellPairView](ABC):
    """Hermite expansion operator as a computational tree."""

    @property
    @abstractmethod
    def view(self) -> V:
        """Shell or shell-pair view used by the expansion."""
        pass

    def differentiate(
        self,
        axis: int,
        *,
        side: Literal["bra", "ket"] = "ket",
    ) -> ExpansionDerivative[V]:
        """Differentiate the expansion with respect to a Cartesian coordinate.

        Args:
            axis: Cartesian axis to differentiate along.
            side: Whether to differentiate the bra or ket side of the expansion.

        Returns:
            Differentiated expansion.
        """
        return ExpansionDerivative(self, axis, side)

    def multiply_coordinate(
        self,
        axis: int,
        origin: float = 0.0,
        *,
        side: Literal["bra", "ket"] = "ket",
    ) -> ExpansionCoordinateProduct[V]:
        """Multiply the expansion by a Cartesian coordinate.

        Args:
            axis: Cartesian axis to multiply by.
            origin: Origin of the coordinate to multiply by.
            side: Whether to multiply the bra or ket side of the expansion.

        Returns:
            Multiplied expansion.
        """
        return ExpansionCoordinateProduct(self, axis, origin, side)

    def get_weights(self, weights: WeightMap | None = None) -> tuple[WeightMap, ...]:
        """Return the per-term angular-momentum weights for this recipe.

        Args:
            weights: Initial weights keyed by side and Cartesian axis.

        Returns:
            One weight map for each expansion term.
        """
        return (dict(weights or {}),)

    @lru_cache(maxsize=None)
    def compute(
        self,
        *,
        max_order: int | None = None,
        spherical: bool = False,
    ) -> ExpansionMatrix:
        """Compute the Cartesian or real spherical Hermite expansion matrix.

        Args:
            max_order: Maximum Hermite order to compute. Defaults to the order required by the expansion.
            spherical: Whether to return real spherical functions instead of Cartesian functions.

        Returns:
            Hermite expansion matrix.
        """
        plans = self.get_weights()
        if max_order is None:
            max_order = _plans_max_order(self.view, plans)
        if isinstance(self.view, ShellView):
            return _compute_shell(self.view, plans, max_order, spherical)
        return _compute_shell_pair(self.view, plans, max_order, spherical)

    def __add__(self, other: object) -> ExpansionTree[V]:
        if not isinstance(other, ExpansionTree):
            return NotImplemented
        if type(self.view) is not type(other.view) or self.view != other.view:
            return NotImplemented
        left = self.children if isinstance(self, ExpansionSum) else (self,)
        right = other.children if isinstance(other, ExpansionSum) else (other,)
        return ExpansionSum[V](left + right)

    def __radd__(self, other: object) -> ExpansionTree[V]:
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, factor: float) -> ExpansionScaled[V]:
        if not isinstance(factor, (int, float, np.floating)):
            return NotImplemented
        return ExpansionScaled(self, factor)

    def __rmul__(self, factor: float) -> ExpansionScaled[V]:
        return self.__mul__(factor)

    def __hash__(self) -> int:
        return hash((self.__class__, self.view))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpansionTree):
            return NotImplemented
        return type(self.view) is type(other.view) and self.view == other.view


@dataclass(frozen=True, eq=True, slots=True)
class Expansion[V: ShellView | ShellPairView](ExpansionTree[V]):
    r"""Hermite expansion operator for a shell or shell pair.

    This class recursively computes either

    .. math::
        E_t^i = \frac{1}{2\alpha} E_{t-1}^{i-1} + t E_{t-1}^{i+1},

    or

    .. math::
        E_t^{ij} = \frac{1}{2\alpha} E_{t-1}^{i-1,j} + t E_{t-1}^{i+1,j} + \frac{1}{2\alpha} E_{t-1}^{i,j-1}
            + t E_{t-1}^{i,j+1},

    depending on whether the expansion is for a single shell or a shell pair.
    """

    shell: V

    @property
    def view(self) -> V:
        """Shell view used by the expansion."""
        return self.shell


@dataclass(frozen=True, eq=True, slots=True)
class ExpansionDerivative[V: ShellView | ShellPairView](ExpansionTree[V]):
    r"""Differentiated Hermite expansion operator for a shell or shell pair.

    This class computes either

    .. math::
        \frac{\partial E_t^i}{\partial x} = i E_t^{i-1} - 2\alpha E_t^{i+1},

    or

    .. math::
        \frac{\partial E_t^{ij}}{\partial x} =
        \begin{cases}
            i E_t^{i-1,j} - 2\alpha E_t^{i+1,j}, & \text{side = bra}, \\
            j E_t^{i,j-1} - 2\beta E_t^{i,j+1}, & \text{side = ket},
        \end{cases}

    For a shell pair, ``side`` selects the differentiated shell.
    """

    source: ExpansionTree[V]
    axis: int
    side: Literal["bra", "ket"] = "ket"

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"invalid axis: {self.axis!r}")

    @property
    def view(self) -> V:
        """Shell or shell-pair view used by the expansion."""
        return self.source.view

    def get_weights(self, weights: WeightMap | None = None) -> tuple[WeightMap, ...]:
        """Return weights after applying this derivative to each source plan.

        Args:
            weights: Initial weights keyed by side and Cartesian axis.

        Returns:
            One weight map for each expansion term.
        """
        plans = self.source.get_weights(weights)
        shell = _shell_for_side(self.source.view, self.side)
        side = _weight_side(self.source.view, self.side)
        result = []
        for plan in plans:
            updated = dict(plan)
            source_weights = updated.get((side, self.axis))
            if source_weights is None:
                source_weights = _identity_weights(shell)
            updated[side, self.axis] = _differentiate_weights(source_weights, shell)
            result.append(updated)
        return tuple(result)


@dataclass(frozen=True, eq=True, slots=True)
class ExpansionCoordinateProduct[V: ShellView | ShellPairView](ExpansionTree[V]):
    r"""Hermite expansion operator multiplied by a Cartesian coordinate for a shell or shell pair.

    This class computes either

    .. math::
        (x - x_0) E_t^i = (A - x_0) E_t^i + E_t^{i+1},

    or

    .. math::
        (x - x_0) E_t^{ij} =
        \begin{cases}
            (A - x_0) E_t^{ij} + E_t^{i+1,j}, & \text{side = bra}, \\
            (B - x_0) E_t^{ij} + E_t^{i,j+1}, & \text{side = ket},
        \end{cases}

    For a shell pair, ``side`` selects the multiplied shell.
    """

    source: ExpansionTree[V]
    axis: int
    origin: float = 0.0
    side: Literal["bra", "ket"] = "ket"

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"invalid axis: {self.axis!r}")

    @property
    def view(self) -> V:
        """Shell or shell-pair view used by the expansion."""
        return self.source.view

    def get_weights(self, weights: WeightMap | None = None) -> tuple[WeightMap, ...]:
        """Return weights after applying this coordinate product to each source plan.

        Args:
            weights: Initial weights keyed by side and Cartesian axis.

        Returns:
            One weight map for each expansion term.
        """
        plans = self.source.get_weights(weights)
        shell = _shell_for_side(self.source.view, self.side)
        side = _weight_side(self.source.view, self.side)
        result = []
        for plan in plans:
            updated = dict(plan)
            source_weights = updated.get((side, self.axis))
            if source_weights is None:
                source_weights = _identity_weights(shell)
            updated[side, self.axis] = _multiply_coordinate_weights(source_weights, shell, self.axis, self.origin)
            result.append(updated)
        return tuple(result)


@dataclass(frozen=True, eq=True, slots=True)
class ExpansionScaled[V: ShellView | ShellPairView](ExpansionTree[V]):
    """Scaled Hermite expansion operator for a shell or shell pair."""

    source: ExpansionTree[V]
    factor: float

    @property
    def view(self) -> V:
        """Shell or shell-pair view used by the expansion."""
        return self.source.view

    def get_weights(self, weights: WeightMap | None = None) -> tuple[WeightMap, ...]:
        """Return the weight plans scaled by this factor.

        Args:
            weights: Initial weights keyed by side and Cartesian axis.

        Returns:
            One weight map for each expansion term.
        """
        plans = self.source.get_weights(weights)
        shell = _shell_for_side(self.view, "ket")
        result = []
        for plan in plans:
            updated = dict(plan)
            source_weights = updated.get(("ket", 0))
            if source_weights is None:
                source_weights = _identity_weights(shell)
            updated["ket", 0] = source_weights * self.factor
            result.append(updated)
        return tuple(result)


@dataclass(frozen=True, eq=True, slots=True)
class ExpansionSum[V: ShellView | ShellPairView](ExpansionTree[V]):
    """Sum of Hermite expansion operators for a shell or shell pair."""

    children: tuple[ExpansionTree[V], ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("a sum must contain at least one expansion")
        if any(child.view != self.children[0].view for child in self.children[1:]):
            raise ValueError("summed expansions must use the same view")

    @property
    def view(self) -> V:
        """Shell or shell-pair view used by the expansions."""
        return self.children[0].view

    def get_weights(self, weights: WeightMap | None = None) -> tuple[WeightMap, ...]:
        """Return the weight plans for each child expansion.

        Args:
            weights: Initial weights keyed by side and Cartesian axis.

        Returns:
            One weight map for each expansion term.
        """
        return tuple(plan for child in self.children for plan in child.get_weights(weights))


def expansion_coefficients(
    shell: ShellView,
    *,
    angular_momentum: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion coefficients :math:`E_t^i`.

    Args:
        shell: Shell view.
        angular_momentum: Angular momentum of the shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion coefficients.
    """
    if angular_momentum is None:
        angular_momentum = shell.angular_momentum
    if max_order is None:
        max_order = shell.angular_momentum
    orders = np.arange(1, max_order + 1)

    # a -> a t
    half_reciprocal_exponent = np.expand_dims(0.5 / shell.exponent, axis=-1)

    def raise_coefficient(
        source: array[tuple[int, int, int], float64],
        target: array[tuple[int, int, int], float64],
    ) -> None:
        """Raise coefficients along the shell angular-momentum index.

        Args:
            source: Coefficients at the current shell angular momentum.
            target: Output coefficients at the raised angular momentum.
        """
        # a t, a -> a t
        target[..., 1:] += source[..., :-1] * half_reciprocal_exponent
        target[..., :-1] += orders * source[..., 1:]

    coefficients = np.zeros((len(shell), max_order + 1, angular_momentum + 1), dtype=np.float64)

    # E_0^0
    coefficients[..., 0, 0] = 1.0

    # E_t^i
    for i in range(angular_momentum):
        raise_coefficient(coefficients[..., :, i], coefficients[..., :, i + 1])

    return coefficients


def expansion_pair_coefficients(
    pair: ShellPairView,
    axis: int,
    *,
    angular_momentum_i: int | None = None,
    angular_momentum_j: int | None = None,
    max_order: int | None = None,
) -> array[tuple[int, int, int, int, int], float64]:
    r"""Compute one-dimensional Hermite expansion pair coefficients :math:`E_t^{ij}`.

    Args:
        pair: Shell pair view.
        axis: Cartesian axis.
        angular_momentum_i: Angular momentum of the first shell.
        angular_momentum_j: Angular momentum of the second shell.
        max_order: Maximum Hermite order to generate.

    Returns:
        Expansion pair coefficients.
    """
    if angular_momentum_i is None:
        angular_momentum_i = pair.shell_i.angular_momentum
    if angular_momentum_j is None:
        angular_momentum_j = pair.shell_j.angular_momentum
    if max_order is None:
        max_order = angular_momentum_i + angular_momentum_j
    orders = np.arange(1, max_order + 1)

    # a b -> a b t
    half_reciprocal_exponent = np.expand_dims(0.5 / pair.exponent, axis=-1)

    # a b, a -> a b
    offset_a = pair.position[..., axis] - pair.shell_i.position[..., axis][:, None]

    # a b, b -> a b
    offset_b = pair.position[..., axis] - pair.shell_j.position[..., axis][None, :]

    def raise_coefficient(
        source: array[tuple[int, int, int], float64],
        target: array[tuple[int, int, int], float64],
        offset: array[tuple[int, int], float64],
    ) -> None:
        """Raise coefficients along one shell angular-momentum index.

        Args:
            source: Coefficients at the current shell angular momentum.
            target: Output coefficients at the raised angular momentum.
            offset: Center offset for the shell being raised.
        """
        # a b t, a b -> a b t
        target[...] = source * np.expand_dims(offset, axis=-1)
        target[..., 1:] += source[..., :-1] * half_reciprocal_exponent
        target[..., :-1] += orders * source[..., 1:]

    coefficients = np.zeros(
        (
            len(pair.shell_i),
            len(pair.shell_j),
            max_order + 1,
            angular_momentum_i + 1,
            angular_momentum_j + 1,
        ),
        dtype=np.float64,
    )

    # E_0^00
    coefficients[..., 0, 0, 0] = pair.attenuation[..., axis]

    # E_t^i0
    for i in range(angular_momentum_i):
        raise_coefficient(coefficients[..., :, i, 0], coefficients[..., :, i + 1, 0], offset_a)

    # E_t^ij
    for i, j in itertools.product(
        range(angular_momentum_i + 1),
        range(angular_momentum_j),
    ):
        raise_coefficient(coefficients[..., :, i, j], coefficients[..., :, i, j + 1], offset_b)

    return coefficients


def _compute_shell(
    shell: ShellView,
    plans: tuple[WeightMap, ...],
    max_order: int,
    spherical: bool,
) -> ExpansionMatrix:
    """Compute a shell expansion from its weight plans."""
    angular_momentum = shell.angular_momentum
    cartesian = cartesian_indices(angular_momentum)
    indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
    coefficients = np.zeros((len(shell), len(indices[0]), len(cartesian)), dtype=np.float64)
    tables: dict[tuple[int, int], array[tuple[int, int, int], float64]] = {}

    # Compute the expansion coefficients along each Cartesian axis
    for plan in plans:
        term_coefficients = np.ones_like(coefficients)
        for axis in range(3):
            weights = plan.get(("ket", axis))
            shift = _weight_shift(weights)
            if (axis, shift) not in tables:
                tables[axis, shift] = expansion_coefficients(
                    shell,
                    angular_momentum=angular_momentum + shift,
                    max_order=max_order,
                )

            table = np.take(tables[axis, shift], indices[axis], axis=1)
            if weights is not None:
                table = (weights @ table.mT).mT
            table = np.take(table, cartesian[:, axis], axis=2)
            term_coefficients *= table.reshape(coefficients.shape)

        coefficients += term_coefficients

    # Apply Cartesian normalization factors
    coefficients *= cartesian_normalization(angular_momentum)  # a t i, i -> a t i

    if spherical:
        # Transform the Cartesian coefficients to real spherical coefficients
        coefficients = coefficients @ cartesian_to_spherical(angular_momentum)  # a t i, i p -> a t p

    # Reorder axes
    coefficients = coefficients.mT  # a t i -> a i t

    return coefficients


def _compute_shell_pair(
    pair: ShellPairView,
    plans: tuple[WeightMap, ...],
    max_order: int,
    spherical: bool,
) -> ExpansionMatrix:
    """Compute a shell-pair expansion from its weight plans."""
    angular_momentum_i = pair.shell_i.angular_momentum
    angular_momentum_j = pair.shell_j.angular_momentum
    cartesian_i = cartesian_indices(angular_momentum_i)
    cartesian_j = cartesian_indices(angular_momentum_j)
    indices = np.asarray(hermite_indices(max_order), dtype=np.int64).T
    coefficients = np.zeros((len(pair), len(indices[0]), len(cartesian_i) * len(cartesian_j)), dtype=np.float64)
    tables: dict[tuple[int, int, int], array[tuple[int, int, int, int, int], float64]] = {}

    # Compute the expansion coefficients along each Cartesian axis
    for plan in plans:
        term_coefficients = np.ones_like(coefficients)
        for axis in range(3):
            weights_i = plan.get(("bra", axis))
            weights_j = plan.get(("ket", axis))
            shift_i = _weight_shift(weights_i)
            shift_j = _weight_shift(weights_j)
            if (axis, shift_i, shift_j) not in tables:
                tables[axis, shift_i, shift_j] = expansion_pair_coefficients(
                    pair,
                    axis,
                    angular_momentum_i=angular_momentum_i + shift_i,
                    angular_momentum_j=angular_momentum_j + shift_j,
                    max_order=max_order,
                )

            table = np.take(tables[axis, shift_i, shift_j], indices[axis], axis=2)
            if weights_i is not None:
                table = np.expand_dims(weights_i, axis=(1, 2)) @ table
            if weights_j is not None:
                table = table @ np.expand_dims(weights_j, axis=(0, 2)).mT
            table = np.take(table, cartesian_i[:, axis], axis=3)
            table = np.take(table, cartesian_j[:, axis], axis=4)
            term_coefficients *= table.reshape(coefficients.shape)
        coefficients += term_coefficients

    # Apply Cartesian normalization factors
    normalization = np.multiply.outer(
        cartesian_normalization(angular_momentum_i), cartesian_normalization(angular_momentum_j)
    ).ravel()
    coefficients *= normalization  # ab t ij, ij -> ab t ij

    if spherical:
        # Transform the Cartesian coefficients to real spherical coefficients
        transform = np.multiply.outer(
            cartesian_to_spherical(angular_momentum_i),
            cartesian_to_spherical(angular_momentum_j),
        )
        transform = transform.transpose(0, 2, 1, 3).reshape(len(cartesian_i) * len(cartesian_j), -1)
        coefficients = coefficients @ transform  # ab t ij, ij pq -> ab t pq

    coefficients = coefficients.mT  # ab t ij -> ab ij t

    return coefficients
