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

    def order_by_angular_momentum(self) -> tuple[Shells, array[tuple[int], int64]]:
        """Return a copy of the shell data ordered by angular momentum.

        Returns:
            A new ``Shells`` object with all primitive arrays reordered together, and an array of indices that can be
            used to invert the ordering.
        """
        indices = np.argsort(self.angular_momentum)
        inverse = np.empty_like(indices)
        inverse[indices] = np.arange(len(indices))
        return self.order_by_indices(indices), inverse


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
        self.shells.check_contracted_slice(self.indices)

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
        start = 0
        for shell in np.unique(self.shells.shell_indices[: self.indices.start or 0]):
            start += num_cartesian_indices(self.shells.angular_momentum[shell])
        stop = start + num_cartesian_indices(self.angular_momentum)
        return slice(start, stop)

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
        self.shell_pairs.shells.check_contracted_slice(self.indices_i)
        self.shell_pairs.shells.check_contracted_slice(self.indices_j)

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
