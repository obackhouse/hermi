"""Gaussian basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np

from pymmd.screening import NoScreening
from pymmd.util import num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from typing import Iterable, Literal

    from numpy import bool_, float64, int64

    from pymmd._typing import array
    from pymmd.screening import Screening


class DefaultHash:
    """Base class for objects with the default ``object.__hash__`` and ``object.__eq__``."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultHash):
            return NotImplemented
        return object.__eq__(self, other)

    def __hash__(self) -> int:
        return object.__hash__(self)


@dataclass(frozen=True, eq=False)
class Shells(DefaultHash):
    """Primitive shell data with coefficients for unnormalized radial Gaussians."""

    position: array[tuple[int, int], float64]
    exponent: array[tuple[int], float64]
    coefficient: array[tuple[int], float64]
    angular_momentum: array[tuple[int], int64]
    shell_indices: array[tuple[int], int64]

    def check_contracted_slice(self, indices: slice) -> None:
        """Check that a slice selects one contracted shell and angular momentum.

        Args:
            indices: Slice into the primitive-shell arrays.

        Raises:
            ValueError: If the slice selects multiple shells or angular momenta.
        """
        if np.unique(self.shell_indices[indices]).size > 1:
            raise ValueError("slice must correspond to a single contracted function")
        if np.unique(self.angular_momentum[indices]).size > 1:
            raise ValueError("slice must correspond to a single angular momentum")

    def check_shell_group_slice(self, indices: slice) -> None:
        """Check that a slice selects complete contiguous shells with one angular momentum."""
        start = indices.start or 0
        stop = indices.stop if indices.stop is not None else self.shell_indices.size
        selected = self.shell_indices[indices]
        shell_ids = np.unique(selected)
        if shell_ids.size == 0:
            raise ValueError("slice must select at least one shell")
        if np.unique(self.angular_momentum[indices]).size > 1:
            raise ValueError("slice must contain shells with one angular momentum")
        for shell_id in shell_ids:
            positions = np.flatnonzero(self.shell_indices == shell_id)
            if positions[0] < start or positions[-1] >= stop or positions.size != positions[-1] - positions[0] + 1:
                raise ValueError("slice must contain complete contiguous shells")

    @cached_property
    def normalized_coefficient(self) -> array[tuple[int], float64]:
        """Return contraction coefficients with radial normalization applied."""
        coefficients = self.coefficient.copy()
        for shell_index in np.unique(self.shell_indices):
            selected = self.shell_indices == shell_index
            exponents = self.exponent[selected]
            angular_momentum = int(self.angular_momentum[selected][0])
            raw_coefficients = self.coefficient[selected]

            exponent_product = np.multiply.outer(exponents, exponents)
            exponent_sum = np.add.outer(exponents, exponents)
            normalized_primitive_overlap = (2 * np.sqrt(exponent_product) / exponent_sum) ** (angular_momentum + 1.5)
            contraction_norm_squared = raw_coefficients @ (normalized_primitive_overlap @ raw_coefficients)
            radial_normalization = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (angular_momentum / 2)
            coefficients[selected] = raw_coefficients * radial_normalization / np.sqrt(contraction_norm_squared)

        return coefficients

    def ordered_by_indices(self, order: array[tuple[int], int64]) -> Shells:
        """Return a copy of the shell data ordered by a given index array.

        Args:
            order: Array of indices to reorder the primitive arrays.

        Returns:
            A new ``Shells`` object with all primitive arrays reordered together.
        """
        return Shells(
            position=self.position[order].copy(),
            exponent=self.exponent[order].copy(),
            coefficient=self.coefficient[order].copy(),
            angular_momentum=self.angular_momentum[order].copy(),
            shell_indices=self.shell_indices[order].copy(),
        )


@dataclass(frozen=True, eq=False)
class ShellPairs(DefaultHash):
    """Basis data for primitive shell pairs."""

    shells: Shells
    position: array[tuple[int, int, int], float64]
    vector: array[tuple[int, int, int], float64]
    exponent: array[tuple[int, int], float64]
    reduced_exponent: array[tuple[int, int], float64]
    coefficient: array[tuple[int, int], float64]
    attenuation: array[tuple[int, int, int], float64]
    angular_momentum: array[tuple[int, int], int64]

    @classmethod
    def from_shells(cls, shells: Shells) -> ShellPairs:
        """Construct all primitive-pair data from shell arrays.

        Args:
            shells: Primitive shell data to combine pairwise.

        Returns:
            ``ShellPairs`` data with pairwise product centers and exponents.
        """
        vector = shells.position[:, None] - shells.position[None, :]
        exponent_sum = np.add.outer(shells.exponent, shells.exponent)
        exponent_product = np.multiply.outer(shells.exponent, shells.exponent)
        reduced_exponent = exponent_product / exponent_sum
        position_scaled = shells.exponent[:, None] * shells.position
        position = (position_scaled[:, None] + position_scaled[None, :]) / exponent_sum[..., None]
        coefficient = np.multiply.outer(shells.normalized_coefficient, shells.normalized_coefficient)
        attenuation = np.exp(-reduced_exponent[..., None] * vector**2)
        angular_momentum = np.add.outer(shells.angular_momentum, shells.angular_momentum)
        return cls(
            shells=shells,
            position=position,
            vector=vector,
            exponent=exponent_sum,
            reduced_exponent=reduced_exponent,
            coefficient=coefficient,
            attenuation=attenuation,
            angular_momentum=angular_momentum,
        )

    def ordered_by_indices(self, order_i: array[tuple[int], int64], order_j: array[tuple[int], int64]) -> ShellPairs:
        """Return a copy of the shell-pair data ordered by given index arrays.

        Args:
            order_i: Array of indices to reorder the first primitive-shell axis.
            order_j: Array of indices to reorder the second primitive-shell axis.

        Returns:
            A new ``ShellPairs`` object with all primitive-pair arrays reordered together.
        """
        return ShellPairs(
            shells=self.shells.ordered_by_indices(order_i),
            position=self.position[order_i][:, order_j].copy(),
            vector=self.vector[order_i][:, order_j].copy(),
            exponent=self.exponent[order_i][:, order_j].copy(),
            reduced_exponent=self.reduced_exponent[order_i][:, order_j].copy(),
            coefficient=self.coefficient[order_i][:, order_j].copy(),
            attenuation=self.attenuation[order_i][:, order_j].copy(),
            angular_momentum=self.angular_momentum[order_i][:, order_j].copy(),
        )


@dataclass(frozen=True, eq=False)
class Basis(DefaultHash):
    """Basis data."""

    shells: Shells
    shell_pairs: ShellPairs

    @classmethod
    def from_shells(cls, shells: Shells) -> Basis:
        """Construct basis and primitive-pair data from shells.

        Args:
            shells: Primitive shell data.

        Returns:
            A ``Basis`` with pair data derived from ``shells``.
        """
        shell_pairs = ShellPairs.from_shells(shells)
        return cls(shells=shells, shell_pairs=shell_pairs)

    @property
    def num_primitive_shells(self) -> int:
        """Number of primitive shells."""
        return self.shells.position.shape[0]

    @cached_property
    def num_contracted_shells(self) -> int:
        """Number of contracted shells."""
        return np.unique(self.shells.shell_indices).size

    @cached_property
    def num_primitive_cartesian_aos(self) -> int:
        """Number of primitive Cartesian atomic orbitals."""
        return int(np.sum(num_cartesian_functions(self.shells.angular_momentum)))

    @cached_property
    def num_contracted_cartesian_aos(self) -> int:
        """Number of contracted Cartesian atomic orbitals."""
        shell_starts = np.unique(self.shells.shell_indices, return_index=True)[1]
        return int(np.sum(num_cartesian_functions(self.shells.angular_momentum[shell_starts])))

    @cached_property
    def num_primitive_spherical_aos(self) -> int:
        """Number of primitive real spherical atomic orbitals."""
        return int(np.sum(num_spherical_functions(self.shells.angular_momentum)))

    @cached_property
    def num_contracted_spherical_aos(self) -> int:
        """Number of contracted real spherical atomic orbitals."""
        shell_starts = np.unique(self.shells.shell_indices, return_index=True)[1]
        return int(np.sum(num_spherical_functions(self.shells.angular_momentum[shell_starts])))

    @staticmethod
    def screen_shell_pair(
        shell_pair: ShellPairView,
        threshold: float = 1e-3,
        prefactor: float = 1.0,
    ) -> bool:
        """Determine whether a shell pair is significant based on its coefficient and attenuation.

        Args:
            shell_pair: Shell pair to screen.
            threshold: Minimum product of contraction coefficients and attenuation for significance.
            prefactor: Factor to scale the threshold for significance.

        Returns:
            True if the shell pair is significant, False otherwise.
        """
        normalized_coefficient = shell_pair.shell_pairs.shells.normalized_coefficient
        quantity = (
            np.einsum(
                "i,j,ijx->",
                normalized_coefficient[shell_pair.indices_i],
                normalized_coefficient[shell_pair.indices_j],
                shell_pair.attenuation,
            )
            * prefactor
        )
        return bool(np.max(np.abs(quantity)) >= threshold)

    def iter_shells(self) -> Iterable[ShellView]:
        """Iterate over contracted shell views."""
        for shell in np.unique(self.shells.shell_indices):
            shell_slice = slice(shell, shell + np.sum(self.shells.shell_indices == shell))
            yield ShellView(self.shells, indices=shell_slice)

    def iter_shell_pairs(
        self, unique: bool = False, screening: Screening[Literal[2]] = NoScreening()
    ) -> Iterable[ShellPairView]:
        """Iterate over contracted shell-pair views."""
        shells = list(self.iter_shells())
        for i, shell_i in enumerate(shells):
            for j, shell_j in enumerate(shells):
                if unique and j < i:
                    continue
                shell_pair = ShellPairView(self.shell_pairs, indices_i=shell_i.indices, indices_j=shell_j.indices)
                yield from screening(shell_pair)

    def iter_shell_groups(self) -> Iterable[ShellView]:
        """Iterate over contiguous shell groups with matching angular momentum."""
        shell_ids, first_indices = np.unique(self.shells.shell_indices, return_index=True)
        shell_ids = shell_ids[np.argsort(first_indices)]
        start = None
        stop = None
        angular_momentum = None
        for shell_id in shell_ids:
            positions = np.flatnonzero(self.shells.shell_indices == shell_id)
            shell_start = positions[0]
            shell_stop = positions[-1] + 1
            shell_angular_momentum = self.shells.angular_momentum[shell_start]
            if start is not None and shell_angular_momentum != angular_momentum:
                yield ShellView(self.shells, indices=slice(start, stop))
                start = None
            if start is None:
                start = shell_start
            stop = shell_stop
            angular_momentum = shell_angular_momentum
        if start is not None:
            yield ShellView(self.shells, indices=slice(start, stop))

    def iter_shell_pair_groups(
        self, unique: bool = False, screening: Screening[Literal[2]] = NoScreening()
    ) -> Iterable[ShellPairView]:
        """Iterate over shell-pair rectangles with matching angular momenta on each side.

        If screened, each rectangle contains only shell pairs that pass screening.
        """
        shell_groups = list(self.iter_shell_groups())
        for i, shell_i in enumerate(shell_groups):
            for j, shell_j in enumerate(shell_groups):
                if unique and j < i:
                    continue
                shell_pair = ShellPairView(self.shell_pairs, indices_i=shell_i.indices, indices_j=shell_j.indices)
                yield from screening(shell_pair)

    def ordered_by_angular_momentum(self, spherical: bool = False) -> tuple[Basis, array[tuple[int], int64]]:
        """Return a copy of the basis ordered by angular momentum.

        Args:
            spherical: Build the AO permutation for real spherical functions.

        Returns:
            A new ``Basis`` object with contracted shells ordered by angular momentum, and AO indices
            that restore the original basis order.
        """
        shell_ids, first_indices = np.unique(self.shells.shell_indices, return_index=True)
        shell_ids = shell_ids[np.argsort(first_indices)]
        shell_groups = [np.flatnonzero(self.shells.shell_indices == shell_id) for shell_id in shell_ids]
        shell_order = np.argsort([self.shells.angular_momentum[group[0]] for group in shell_groups], kind="stable")
        ordered_groups = [shell_groups[index] for index in shell_order]
        order = np.concatenate(ordered_groups)

        ordered_shells = self.shells.ordered_by_indices(order)
        shell_indices = np.empty_like(ordered_shells.shell_indices)
        start = 0
        for group in ordered_groups:
            stop = start + len(group)
            shell_indices[start:stop] = start
            start = stop
        ordered_shells = Shells(
            position=ordered_shells.position,
            exponent=ordered_shells.exponent,
            coefficient=ordered_shells.coefficient,
            angular_momentum=ordered_shells.angular_momentum,
            shell_indices=shell_indices,
        )

        num_functions = num_spherical_functions if spherical else num_cartesian_functions
        ao_starts = np.cumsum([0, *(num_functions(self.shells.angular_momentum[group[0]]) for group in shell_groups)])
        ao_slices = {
            shell_id: np.arange(start, stop) for shell_id, start, stop in zip(shell_ids, ao_starts[:-1], ao_starts[1:])
        }
        indices = np.argsort(np.concatenate([ao_slices[shell_ids[index]] for index in shell_order]), kind="stable")

        shell_pairs = self.shell_pairs.ordered_by_indices(order, order)
        shell_pairs = ShellPairs(
            shells=ordered_shells,
            position=shell_pairs.position,
            vector=shell_pairs.vector,
            exponent=shell_pairs.exponent,
            reduced_exponent=shell_pairs.reduced_exponent,
            coefficient=shell_pairs.coefficient,
            attenuation=shell_pairs.attenuation,
            angular_momentum=shell_pairs.angular_momentum,
        )

        reordered = Basis(shells=ordered_shells, shell_pairs=shell_pairs)

        return reordered, indices

    @overload
    def __getitem__(self, key: slice) -> ShellView: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ShellPairView: ...
    @overload
    def __getitem__(self, key: tuple[ShellView, ShellView]) -> ShellPairView: ...

    def __getitem__(self, key: slice | tuple[slice, slice] | tuple[ShellView, ShellView]) -> ShellView | ShellPairView:
        if isinstance(key, slice):
            self.shells.check_contracted_slice(key)
            return ShellView(self.shells, indices=key)
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, slice) for k in key):
            assert isinstance(key[0], slice) and isinstance(key[1], slice)  # for mypy
            self.shells.check_contracted_slice(key[0])
            self.shells.check_contracted_slice(key[1])
            return ShellPairView(self.shell_pairs, indices_i=key[0], indices_j=key[1])
        elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, ShellView) for k in key):
            assert isinstance(key[0], ShellView) and isinstance(key[1], ShellView)  # for mypy
            return ShellPairView(self.shell_pairs, indices_i=key[0].indices, indices_j=key[1].indices)
        raise TypeError(f"invalid key type: {type(key)}")


@dataclass(frozen=True)
class ShellView:
    """View of a shell."""

    shells: Shells
    indices: slice

    def __post_init__(self) -> None:
        self.shells.check_shell_group_slice(self.indices)

    @property
    def position(self) -> array[tuple[int, int], float64]:
        """Primitive centers for the selected shell."""
        return self.shells.position[self.indices]

    @property
    def exponent(self) -> array[tuple[int], float64]:
        """Primitive exponents for the selected shell."""
        return self.shells.exponent[self.indices]

    @property
    def coefficient(self) -> array[tuple[int], float64]:
        """Radially normalized contraction coefficients for the selected shell."""
        return self.shells.normalized_coefficient[self.indices]

    @property
    def angular_momentum(self) -> int:
        """Angular momentum of the selected shell."""
        return int(self.shells.angular_momentum[self.indices][0])

    @cached_property
    def cartesian_ao_slice(self) -> slice:
        """Slice of contracted Cartesian atomic orbitals for the selected shell."""
        _, first_indices = np.unique(self.shells.shell_indices[: self.indices.start or 0], return_index=True)
        start = int(np.sum(num_cartesian_functions(self.shells.angular_momentum[first_indices])))
        stop = start + num_cartesian_functions(self.angular_momentum) * self.num_shells
        return slice(start, stop)

    @cached_property
    def spherical_ao_slice(self) -> slice:
        """Slice of contracted real spherical atomic orbitals for the selected shell."""
        _, first_indices = np.unique(self.shells.shell_indices[: self.indices.start or 0], return_index=True)
        start = int(np.sum(num_spherical_functions(self.shells.angular_momentum[first_indices])))
        stop = start + num_spherical_functions(self.angular_momentum) * self.num_shells
        return slice(start, stop)

    @cached_property
    def num_shells(self) -> int:
        """Number of contracted shells in the view."""
        return np.unique(self.shells.shell_indices[self.indices]).size

    def __len__(self) -> int:
        start = self.indices.start or 0
        stop = self.indices.stop or self.shells.position.shape[0]
        return int(stop - start)

    def iter_masked_views(self, mask: array[tuple[int], bool_]) -> Iterable[ShellView]:
        """Yield contiguous shell views selected by a one-dimensional mask."""
        if mask.shape != (self.num_shells,):
            raise ValueError(f"mask must have shape ({self.num_shells},)")

        shell_slices = _shell_slices(self)
        for start, stop in _true_runs(mask):
            yield ShellView(
                self.shells,
                indices=slice(shell_slices[start].start, shell_slices[stop - 1].stop),
            )


def _shell_slices(view: ShellView) -> list[slice]:
    """Return primitive slices for each contracted shell in a view."""
    start = view.indices.start or 0
    stop = view.indices.stop if view.indices.stop is not None else view.shells.shell_indices.size
    _, offsets = np.unique(view.shells.shell_indices[view.indices], return_index=True)
    boundaries = [start + int(offset) for offset in np.sort(offsets)]
    boundaries.append(stop)
    return [slice(first, last) for first, last in zip(boundaries[:-1], boundaries[1:])]


def _individual_shell_views(view: ShellView) -> Iterable[ShellView]:
    """Yield one view for each contracted shell in a group."""
    for shell_slice in _shell_slices(view):
        yield ShellView(view.shells, indices=shell_slice)


def _true_runs(mask: array[tuple[int], bool_]) -> list[tuple[int, int]]:
    """Return half-open intervals of consecutive true values."""
    runs = []
    start = None
    for index, selected in enumerate(mask):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


@dataclass(frozen=True)
class ShellPairView:
    """View of a shell pair."""

    shell_pairs: ShellPairs
    indices_i: slice
    indices_j: slice

    def __post_init__(self) -> None:
        self.shell_pairs.shells.check_shell_group_slice(self.indices_i)
        self.shell_pairs.shells.check_shell_group_slice(self.indices_j)

    @cached_property
    def shell_i(self) -> ShellView:
        """View of the first shell."""
        return ShellView(self.shell_pairs.shells, self.indices_i)

    @cached_property
    def shell_j(self) -> ShellView:
        """View of the second shell."""
        return ShellView(self.shell_pairs.shells, self.indices_j)

    @property
    def position(self) -> array[tuple[int, int, int], float64]:
        """Product centers for the selected primitive pairs."""
        return self.shell_pairs.position[self.indices_i, self.indices_j]

    @property
    def vector(self) -> array[tuple[int, int, int], float64]:
        """Center differences for the selected primitive pairs."""
        return self.shell_pairs.vector[self.indices_i, self.indices_j]

    @property
    def exponent(self) -> array[tuple[int, int], float64]:
        """Combined exponents for the selected primitive pairs."""
        return self.shell_pairs.exponent[self.indices_i, self.indices_j]

    @property
    def reduced_exponent(self) -> array[tuple[int, int], float64]:
        """Reduced exponents for the selected primitive pairs."""
        return self.shell_pairs.reduced_exponent[self.indices_i, self.indices_j]

    @property
    def coefficient(self) -> array[tuple[int, int], float64]:
        """Products of primitive coefficients for the selected pairs."""
        return self.shell_pairs.coefficient[self.indices_i, self.indices_j]

    @property
    def attenuation(self) -> array[tuple[int, int, int], float64]:
        """Axis-separated pair attenuation for the selected pairs."""
        return self.shell_pairs.attenuation[self.indices_i, self.indices_j]

    @property
    def angular_momentum(self) -> int:
        """Total angular momentum of the selected shell pair."""
        return int(self.shell_pairs.angular_momentum[self.indices_i, self.indices_j][0, 0])

    @property
    def cartesian_ao_slices(self) -> tuple[slice, slice]:
        """Slices of contracted Cartesian atomic orbitals for the selected shell pair."""
        return self.shell_i.cartesian_ao_slice, self.shell_j.cartesian_ao_slice

    @property
    def spherical_ao_slices(self) -> tuple[slice, slice]:
        """Slices of contracted real spherical atomic orbitals for the selected shell pair."""
        return self.shell_i.spherical_ao_slice, self.shell_j.spherical_ao_slice

    def __len__(self) -> int:
        start_i = self.indices_i.start or 0
        stop_i = self.indices_i.stop or self.shell_pairs.position.shape[0]
        start_j = self.indices_j.start or 0
        stop_j = self.indices_j.stop or self.shell_pairs.position.shape[1]
        return int((stop_i - start_i) * (stop_j - start_j))

    def iter_masked_views(self, mask: array[tuple[int, int], bool_]) -> Iterable[ShellPairView]:
        """Yield rectangular shell-pair views containing only true mask entries."""
        shape = (self.shell_i.num_shells, self.shell_j.num_shells)
        if mask.shape != shape:
            raise ValueError(f"mask must have shape {shape}")

        shells_i = _shell_slices(self.shell_i)
        shells_j = _shell_slices(self.shell_j)
        active: dict[tuple[int, int], int] = {}

        def append_rectangle(row_start: int, row_stop: int, interval: tuple[int, int]) -> ShellPairView:
            column_start, column_stop = interval
            return ShellPairView(
                self.shell_pairs,
                indices_i=slice(shells_i[row_start].start, shells_i[row_stop - 1].stop),
                indices_j=slice(shells_j[column_start].start, shells_j[column_stop - 1].stop),
            )

        for row, row_mask in enumerate(mask):
            intervals = _true_runs(row_mask)
            current = set(intervals)
            for interval in sorted(active.keys() - current):
                yield append_rectangle(active.pop(interval), row, interval)
            for interval in intervals:
                active.setdefault(interval, row)

        for interval, row_start in sorted(active.items()):
            yield append_rectangle(row_start, shape[0], interval)
