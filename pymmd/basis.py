"""Gaussian basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np

from pymmd.util import num_cartesian_indices

if TYPE_CHECKING:
    from typing import Iterable

    from numpy import float64, int64

    from pymmd._typing import array


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

    def order_by_indices(self, order: array[tuple[int], int64]) -> Shells:
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

    def order_by_indices(self, order_i: array[tuple[int], int64], order_j: array[tuple[int], int64]) -> ShellPairs:
        """Return a copy of the shell-pair data ordered by given index arrays.

        Args:
            order_i: Array of indices to reorder the first primitive-shell axis.
            order_j: Array of indices to reorder the second primitive-shell axis.

        Returns:
            A new ``ShellPairs`` object with all primitive-pair arrays reordered together.
        """
        return ShellPairs(
            shells=self.shells.order_by_indices(order_i),
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
    def num_primitive_aos(self) -> int:
        """Number of primitive atomic orbitals."""
        return int(np.sum(num_cartesian_indices(self.shells.angular_momentum)))

    @cached_property
    def num_contracted_aos(self) -> int:
        """Number of contracted atomic orbitals."""
        shell_starts = np.unique(self.shells.shell_indices, return_index=True)[1]
        return int(np.sum(num_cartesian_indices(self.shells.angular_momentum[shell_starts])))

    def iter_shells(self) -> Iterable[ShellView]:
        """Iterate over contracted shell views."""
        for shell in np.unique(self.shells.shell_indices):
            shell_slice = slice(shell, shell + np.sum(self.shells.shell_indices == shell))
            yield ShellView(self.shells, indices=shell_slice)

    def iter_shell_pairs(self, unique: bool = False) -> Iterable[ShellPairView]:
        """Iterate over contracted shell-pair views."""
        for shell_i in np.unique(self.shells.shell_indices):
            for shell_j in np.unique(self.shells.shell_indices):
                if unique and shell_j < shell_i:
                    continue
                shell_slice_i = slice(shell_i, shell_i + np.sum(self.shells.shell_indices == shell_i))
                shell_slice_j = slice(shell_j, shell_j + np.sum(self.shells.shell_indices == shell_j))
                yield ShellPairView(self.shell_pairs, indices_i=shell_slice_i, indices_j=shell_slice_j)

    def iter_shell_groups(self) -> Iterable[ShellView]:
        """Iterate over contracted contiguous shell view groups with the same angular momentum."""
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

    def iter_shell_pair_groups(self, unique: bool = False) -> Iterable[ShellPairView]:
        """Iterate over rectangular shell-pair groups with matching angular momenta on each side."""
        shell_groups = list(self.iter_shell_groups())
        for i, shell_i in enumerate(shell_groups):
            for j, shell_j in enumerate(shell_groups):
                if unique and j < i:
                    continue
                yield ShellPairView(self.shell_pairs, indices_i=shell_i.indices, indices_j=shell_j.indices)

    def order_by_angular_momentum(self) -> tuple[Basis, array[tuple[int], int64]]:
        """Return a copy of the basis ordered by angular momentum.

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

        ordered_shells = self.shells.order_by_indices(order)
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

        ao_starts = np.cumsum(
            [0, *(num_cartesian_indices(self.shells.angular_momentum[group[0]]) for group in shell_groups)]
        )
        ao_slices = {
            shell_id: np.arange(start, stop) for shell_id, start, stop in zip(shell_ids, ao_starts[:-1], ao_starts[1:])
        }
        indices = np.argsort(np.concatenate([ao_slices[shell_ids[index]] for index in shell_order]), kind="stable")

        shell_pairs = self.shell_pairs.order_by_indices(order, order)
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
    def ao(self) -> slice:
        """Slice of contracted atomic orbitals for the selected shell."""
        # TODO: Easy hidden bottleneck
        _, first_indices = np.unique(self.shells.shell_indices[: self.indices.start or 0], return_index=True)
        start = int(np.sum(num_cartesian_indices(self.shells.angular_momentum[first_indices])))
        stop = start + num_cartesian_indices(self.angular_momentum) * self.num_shells
        return slice(start, stop)

    @cached_property
    def num_shells(self) -> int:
        """Number of contracted shells in the view."""
        return np.unique(self.shells.shell_indices[self.indices]).size

    def __len__(self) -> int:
        start = self.indices.start or 0
        stop = self.indices.stop or self.shells.position.shape[0]
        return int(stop - start)


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
    def ao(self) -> tuple[slice, slice]:
        """Slices of contracted atomic orbitals for the selected shell pair."""
        return self.shell_i.ao, self.shell_j.ao

    def __len__(self) -> int:
        start_i = self.indices_i.start or 0
        stop_i = self.indices_i.stop or self.shell_pairs.position.shape[0]
        start_j = self.indices_j.start or 0
        stop_j = self.indices_j.stop or self.shell_pairs.position.shape[1]
        return int((stop_i - start_i) * (stop_j - start_j))
