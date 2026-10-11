"""Symmetry."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from typing import Iterable, Sequence

    from numpy import number

    from hermi._typing import Comparable, array


def _ravel_symmetric_indices(i: int, j: int) -> int:
    """Ravel symmetric indices to the upper triangle of a matrix."""
    if i > j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def _unravel_symmetric_indices(index: int) -> tuple[int, int]:
    """Unravel symmetric indices from the upper triangle of a matrix."""
    i = int((np.sqrt(8 * index + 1) - 1) / 2)
    j = index - _ravel_symmetric_indices(i, 0)
    return i, j


def _equal_elements[T: (Comparable, array[tuple[int], number])](*elements: T) -> bool:
    """Check if all elements are equal."""
    first = elements[0]
    if any(isinstance(element, np.ndarray) for element in elements):
        if not all(isinstance(element, np.ndarray) for element in elements):
            return False
        return all(np.array_equal(first, element) for element in elements[1:])  # type: ignore[arg-type]
    return all(first == element for element in elements[1:])


@dataclass(frozen=True)
class Symmetry[S: tuple[int, ...], M: tuple[slice, ...]](ABC):
    """Abstract base class for symmetry."""

    shape: S

    def __post_init__(self) -> None:
        self.assert_symmetric_sequence(self.shape)

    @overload
    def set_value[T: number](self, array: array[S, T], indices: S, value: T) -> None: ...
    @overload
    def set_value[T: number](self, array: array[S, T], indices: M, value: array[S, T]) -> None: ...

    @abstractmethod
    def set_value[T: number](self, array: array[S, T], indices: S | M, value: T | array[S, T]) -> None:
        """Set an array according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        pass

    @abstractmethod
    def ravel_index(self, index: S) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        pass

    @abstractmethod
    def unravel_index(self, index: int) -> S:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        pass

    def ravel_asymmetric_index(self, index: S) -> int:
        """Ravel an index as if the array were asymmetric.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        return int(np.ravel_multi_index(index, self.shape))

    def unravel_asymmetric_index(self, index: int) -> S:
        """Unravel an index as if the array were asymmetric.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        return tuple(map(int, np.unravel_index(index, self.shape)))  # type: ignore[return-value]

    @property
    def unravelled_size(self) -> int:
        """Total size of the unravelled array."""
        return prod(self.shape)

    @property
    @abstractmethod
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        pass

    @abstractmethod
    def canonical_indices(self) -> Iterable[tuple[S, bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        pass

    @abstractmethod
    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        pass


@dataclass(frozen=True)
class SymmetryN2S1(Symmetry[tuple[int, int], tuple[slice, slice]]):
    """Symmetry for a 2D array with no symmetry (S1)."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[slice, slice],
        value: array[tuple[int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int] | tuple[slice, slice],
        value: T | array[tuple[int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        array[indices] = value

    def ravel_index(self, index: tuple[int, int]) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        return self.ravel_asymmetric_index(index)

    def unravel_index(self, index: int) -> tuple[int, int]:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        return self.unravel_asymmetric_index(index)

    @property
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        return prod(self.shape)

    def canonical_indices(self) -> Iterable[tuple[tuple[int, int], bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        for indices in itertools.product(range(self.shape[0]), range(self.shape[1])):
            yield indices, False

    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        if len(objects) != 2:
            raise ValueError("objects must have length 2")


@dataclass(frozen=True)
class SymmetryN2S2(Symmetry[tuple[int, int], tuple[slice, slice]]):
    """Symmetry for a 2D array with S2 symmetry."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[slice, slice],
        value: array[tuple[int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int] | tuple[slice, slice],
        value: T | array[tuple[int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        i, j = indices
        array[i, j] = value
        if i != j:
            array[j, i] = np.transpose(value, (1, 0))

    def ravel_index(self, index: tuple[int, int]) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        return _ravel_symmetric_indices(*index)

    def unravel_index(self, index: int) -> tuple[int, int]:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        return _unravel_symmetric_indices(index)

    @property
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        return _ravel_symmetric_indices(self.shape[0], 0)

    def canonical_indices(self) -> Iterable[tuple[tuple[int, int], bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        for indices in itertools.combinations_with_replacement(range(self.shape[0]), 2):
            yield indices, False

    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        if len(objects) != 2:
            raise ValueError("objects must have length 2")
        if not _equal_elements(*objects):
            raise ValueError("objects must be equal at index 0 and 1")


@dataclass(frozen=True)
class SymmetryN2A1(SymmetryN2S1):
    """Symmetry for a 2D array with A1 symmetry."""

    pass


@dataclass(frozen=True)
class SymmetryN2A2(SymmetryN2S2):
    """Symmetry for a 2D array with A2 symmetry."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[slice, slice],
        value: array[tuple[int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int], T],
        indices: tuple[int, int] | tuple[slice, slice],
        value: T | array[tuple[int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        i, j = indices
        array[i, j] = value
        if i != j:
            array[j, i] = -value


@dataclass(frozen=True)
class SymmetryN4S1(Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]):
    """Symmetry for a 4D array with no symmetry (S1)."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[slice, slice, slice, slice],
        value: array[tuple[int, int, int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int] | tuple[slice, slice, slice, slice],
        value: T | array[tuple[int, int, int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        array[indices] = value

    def ravel_index(self, index: tuple[int, int, int, int]) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        return self.ravel_asymmetric_index(index)

    def unravel_index(self, index: int) -> tuple[int, int, int, int]:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        return self.unravel_asymmetric_index(index)

    @property
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        return prod(self.shape)

    def canonical_indices(self) -> Iterable[tuple[tuple[int, int, int, int], bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        for indices in itertools.product(
            range(self.shape[0]), range(self.shape[1]), range(self.shape[2]), range(self.shape[3])
        ):
            yield indices, False

    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        if len(objects) != 4:
            raise ValueError("objects must have length 4")


@dataclass(frozen=True)
class SymmetryN4S4(Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]):
    """Symmetry for a 4D array with S4 symmetry."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[slice, slice, slice, slice],
        value: array[tuple[int, int, int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int] | tuple[slice, slice, slice, slice],
        value: T | array[tuple[int, int, int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        i, j, k, l = indices
        seen = set()
        for indices_perm, perm in [
            ((i, j, k, l), (0, 1, 2, 3)),
            ((j, i, k, l), (1, 0, 2, 3)),
            ((i, j, l, k), (0, 1, 3, 2)),
            ((j, i, l, k), (1, 0, 3, 2)),
        ]:
            if indices_perm not in seen:
                array[indices_perm] = np.transpose(value, perm)
                seen.add(indices_perm)

    def ravel_index(self, index: tuple[int, int, int, int]) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        ij = _ravel_symmetric_indices(index[0], index[1])
        kl = _ravel_symmetric_indices(index[2], index[3])
        return ij * self.shape[2] * self.shape[3] + kl

    def unravel_index(self, index: int) -> tuple[int, int, int, int]:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        ij, kl = divmod(index, self.shape[2] * self.shape[3])
        i, j = _unravel_symmetric_indices(ij)
        k, l = _unravel_symmetric_indices(kl)
        return i, j, k, l

    @property
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        return _ravel_symmetric_indices(self.shape[0], 0) * _ravel_symmetric_indices(self.shape[2], 0)

    def canonical_indices(self) -> Iterable[tuple[tuple[int, int, int, int], bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        for indices_ab in itertools.combinations_with_replacement(range(self.shape[0]), 2):
            for indices_cd in itertools.combinations_with_replacement(range(self.shape[2]), 2):
                yield indices_ab + indices_cd, False

    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        if len(objects) != 4:
            raise ValueError("objects must have length 4")
        if not _equal_elements(objects[0], objects[1]):
            raise ValueError("objects must be equal at index 0 and 1")
        if not _equal_elements(objects[2], objects[3]):
            raise ValueError("objects must be equal at index 2 and 3")


@dataclass(frozen=True)
class SymmetryN4S8(Symmetry[tuple[int, int, int, int], tuple[slice, slice, slice, slice]]):
    """Symmetry for a 4D array with S8 symmetry."""

    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int],
        value: T,
    ) -> None: ...
    @overload
    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[slice, slice, slice, slice],
        value: array[tuple[int, int, int, int], T],
    ) -> None: ...

    def set_value[T: number](
        self,
        array: array[tuple[int, int, int, int], T],
        indices: tuple[int, int, int, int] | tuple[slice, slice, slice, slice],
        value: T | array[tuple[int, int, int, int], T],
    ) -> None:
        """Set an array value according to the symmetry.

        Args:
            array: Array to set values in.
            indices: Indices to set values at.
            value: Value to set.
        """
        i, j, k, l = indices
        seen = set()
        for indices_perm, perm in [
            ((i, j, k, l), (0, 1, 2, 3)),
            ((j, i, k, l), (1, 0, 2, 3)),
            ((i, j, l, k), (0, 1, 3, 2)),
            ((j, i, l, k), (1, 0, 3, 2)),
            ((k, l, i, j), (2, 3, 0, 1)),
            ((l, k, i, j), (3, 2, 0, 1)),
            ((k, l, j, i), (2, 3, 1, 0)),
            ((l, k, j, i), (3, 2, 1, 0)),
        ]:
            if indices_perm not in seen:
                array[indices_perm] = np.transpose(value, perm)
                seen.add(indices_perm)

    def ravel_index(self, index: tuple[int, int, int, int]) -> int:
        """Ravel an index according to the symmetry.

        Args:
            index: Index to ravel.

        Returns:
            Raveled index.
        """
        return _ravel_symmetric_indices(
            _ravel_symmetric_indices(index[0], index[1]),
            _ravel_symmetric_indices(index[2], index[3]),
        )

    def unravel_index(self, index: int) -> tuple[int, int, int, int]:
        """Unravel an index according to the symmetry.

        Args:
            index: Index to unravel.

        Returns:
            Unraveled index.
        """
        ij, kl = _unravel_symmetric_indices(index)
        i, j = _unravel_symmetric_indices(ij)
        k, l = _unravel_symmetric_indices(kl)
        return i, j, k, l

    @property
    def ravelled_size(self) -> int:
        """Total size of the ravelled array."""
        return _ravel_symmetric_indices(_ravel_symmetric_indices(self.shape[0], 0), 0)

    def canonical_indices(self) -> Iterable[tuple[tuple[int, int, int, int], bool]]:
        """Yield canonical indices for the symmetry.

        Yields:
            Canonical indices for the symmetry and whether their sign flips under the permutation.
        """
        for indices in itertools.combinations_with_replacement(
            itertools.combinations_with_replacement(range(self.shape[0]), 2), 2
        ):
            yield indices[0] + indices[1], False

    def assert_symmetric_sequence(self, objects: Sequence[Comparable] | array[tuple[int], number]) -> None:
        """Check that a sequence of objects is valid under the symmetry.

        Args:
            objects: Sequence of objects to check.
        """
        if len(objects) != 4:
            raise ValueError("objects must have length 4")
        if not _equal_elements(*objects):
            raise ValueError("objects must be equal at index 0, 1, 2, and 3")
