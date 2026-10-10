"""Gaussian basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, overload

import basis_set_exchange as bse
import numpy as np

from hermi.screening import NoScreening
from hermi.symmetry import SymmetryN2S1, SymmetryN2S2
from hermi.util import DefaultHash, num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from typing import Iterable, Literal

    from numpy import bool_, float64, int64

    from hermi._typing import array
    from hermi.molecule import Molecule
    from hermi.screening import Screening


@dataclass(frozen=True, eq=False, slots=True)
class Shells(DefaultHash):
    """Primitive shell data with coefficients for unnormalized radial Gaussians."""

    position: array[tuple[int, int], float64]
    exponent: array[tuple[int], float64]
    coefficient: array[tuple[int], float64]
    angular_momentum: array[tuple[int], int64]
    shell_indices: array[tuple[int], int64]

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


@dataclass(frozen=True, eq=False, slots=True)
class ShellPairs(DefaultHash):
    """Basis data for primitive shell pairs."""

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
        coefficient = np.multiply.outer(shells.coefficient, shells.coefficient)
        attenuation = np.exp(-reduced_exponent[..., None] * vector**2)
        angular_momentum = np.add.outer(shells.angular_momentum, shells.angular_momentum)
        return cls(
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
            position=self.position[order_i][:, order_j].copy(),
            vector=self.vector[order_i][:, order_j].copy(),
            exponent=self.exponent[order_i][:, order_j].copy(),
            reduced_exponent=self.reduced_exponent[order_i][:, order_j].copy(),
            coefficient=self.coefficient[order_i][:, order_j].copy(),
            attenuation=self.attenuation[order_i][:, order_j].copy(),
            angular_momentum=self.angular_momentum[order_i][:, order_j].copy(),
        )


@dataclass(frozen=True, eq=False, slots=True)
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

    @classmethod
    def from_molecule(cls, molecule: Molecule, basis_set: str) -> Basis:
        """Construct a basis from a molecule and a basis set name.

        Args:
            molecule: Molecule to build the basis for.
            basis_set: Name of the basis set to use.

        Returns:
            A ``Basis`` with shells and shell pairs derived from the molecule and basis set.
        """
        if not molecule.elements:
            shells = Shells(
                position=np.empty((0, 3), dtype=np.float64),
                exponent=np.empty(0, dtype=np.float64),
                coefficient=np.empty(0, dtype=np.float64),
                angular_momentum=np.empty(0, dtype=np.int64),
                shell_indices=np.empty(0, dtype=np.int64),
            )
            return cls.from_shells(shells)

        atomic_charges = list(dict.fromkeys(element.atomic_charge for element in molecule.elements))
        basis_data = bse.get_basis(basis_set, elements=atomic_charges)

        positions: list[array[tuple[int], float64]] = []
        exponents: list[float] = []
        coefficients: list[float] = []
        angular_momenta: list[int] = []
        shell_indices: list[int] = []

        for element, position in zip(molecule.elements, molecule.positions, strict=True):
            element_data = basis_data["elements"].get(str(element.atomic_charge))
            if element_data is None:
                raise ValueError(f"basis set {basis_set!r} has no data for {element.symbol}")

            for electron_shell in element_data["electron_shells"]:
                shell_exponents = np.asarray(electron_shell["exponents"], dtype=np.float64)
                angular_momentum = electron_shell["angular_momentum"]
                shell_coefficients = electron_shell["coefficients"]

                for momentum, contraction_coefficients in zip(angular_momentum, shell_coefficients, strict=True):
                    contraction_coefficients = np.asarray(contraction_coefficients, dtype=np.float64)
                    if contraction_coefficients.ndim == 1:
                        contraction_coefficients = contraction_coefficients[None, :]
                    if contraction_coefficients.shape[1] != shell_exponents.size:
                        raise ValueError("basis-set coefficient count does not match exponent count")

                    for contraction in contraction_coefficients:
                        start = len(exponents)
                        nprim = shell_exponents.size
                        positions.extend(np.tile(position, (nprim, 1)))
                        exponents.extend(shell_exponents)
                        coefficients.extend(contraction)
                        angular_momenta.extend([momentum] * nprim)
                        shell_indices.extend([start] * nprim)

        shells = Shells(
            position=np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
            exponent=np.asarray(exponents, dtype=np.float64),
            coefficient=np.asarray(coefficients, dtype=np.float64),
            angular_momentum=np.asarray(angular_momenta, dtype=np.int64),
            shell_indices=np.asarray(shell_indices, dtype=np.int64),
        )
        return cls.from_shells(shells).normalize()

    def normalize(self) -> Basis:
        """Return a basis with radial normalization applied to each contraction.

        Returns:
            A new basis with pair data rebuilt from the normalized coefficients.
        """
        coefficients = self.shells.coefficient.copy()
        starts = self.contracted_shell_starts(slice(None))
        stops = np.append(starts[1:], self.num_primitive_shells)
        for first, last in zip(starts, stops):
            selected = slice(int(first), int(last))
            exponents = self.shells.exponent[selected]
            angular_momentum = int(self.shells.angular_momentum[selected][0])
            raw_coefficients = self.shells.coefficient[selected]

            exponent_product = np.multiply.outer(exponents, exponents)
            exponent_sum = np.add.outer(exponents, exponents)
            normalized_primitive_overlap = (2 * np.sqrt(exponent_product) / exponent_sum) ** (angular_momentum + 1.5)
            contraction_norm_squared = raw_coefficients @ (normalized_primitive_overlap @ raw_coefficients)
            radial_normalization = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (angular_momentum / 2)
            coefficients[selected] = raw_coefficients * radial_normalization / np.sqrt(contraction_norm_squared)

        shells = Shells(
            position=self.shells.position.copy(),
            exponent=self.shells.exponent.copy(),
            coefficient=coefficients,
            angular_momentum=self.shells.angular_momentum.copy(),
            shell_indices=self.shells.shell_indices.copy(),
        )
        return self.from_shells(shells)

    @property
    def num_primitive_shells(self) -> int:
        """Number of primitive shells."""
        return self.shells.position.shape[0]

    @property
    def num_contracted_shells(self) -> int:
        """Number of contracted shells."""
        return self.contracted_shell_starts(slice(None)).size

    @property
    def num_primitive_cartesian_aos(self) -> int:
        """Number of primitive Cartesian atomic orbitals."""
        return int(np.sum(num_cartesian_functions(self.shells.angular_momentum)))

    @property
    def num_contracted_cartesian_aos(self) -> int:
        """Number of contracted Cartesian atomic orbitals."""
        shell_starts = self.contracted_shell_starts(slice(None))
        return int(np.sum(num_cartesian_functions(self.shells.angular_momentum[shell_starts])))

    @property
    def num_primitive_spherical_aos(self) -> int:
        """Number of primitive real spherical atomic orbitals."""
        return int(np.sum(num_spherical_functions(self.shells.angular_momentum)))

    @property
    def num_contracted_spherical_aos(self) -> int:
        """Number of contracted real spherical atomic orbitals."""
        shell_starts = self.contracted_shell_starts(slice(None))
        return int(np.sum(num_spherical_functions(self.shells.angular_momentum[shell_starts])))

    def contracted_shell_starts(self, indices: slice) -> array[tuple[int], int64]:
        """Return local primitive offsets where contracted shells begin."""
        _, starts = np.unique(self.shells.shell_indices[indices], return_index=True)
        return np.sort(starts)

    def num_contracted_shells_in(self, indices: slice) -> int:
        """Return the number of contracted shells selected by a primitive-shell slice."""
        return int(self.contracted_shell_starts(indices).size)

    def cartesian_ao_slice(self, indices: slice) -> slice:
        """Return the contracted Cartesian AO slice for a primitive-shell slice."""
        return self._contracted_ao_slice(indices, spherical=False)

    def spherical_ao_slice(self, indices: slice) -> slice:
        """Return the contracted spherical AO slice for a primitive-shell slice."""
        return self._contracted_ao_slice(indices, spherical=True)

    def _contracted_ao_slice(self, indices: slice, *, spherical: bool) -> slice:
        """Return an AO slice for selected complete, contiguous contracted shells."""
        start, stop, step = indices.indices(self.num_primitive_shells)
        if step != 1:
            raise ValueError("shell slices must be contiguous")

        preceding_starts = self.contracted_shell_starts(slice(0, start))
        selected_starts = self.contracted_shell_starts(slice(start, stop)) + start
        angular_momentum = self.shells.angular_momentum
        if spherical:
            start_ao = int(np.sum(num_spherical_functions(angular_momentum[preceding_starts])))
            size = int(np.sum(num_spherical_functions(angular_momentum[selected_starts])))
        else:
            start_ao = int(np.sum(num_cartesian_functions(angular_momentum[preceding_starts])))
            size = int(np.sum(num_cartesian_functions(angular_momentum[selected_starts])))
        return slice(start_ao, start_ao + size)

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
        coefficient = shell_pair.basis.shells.coefficient
        quantity = (
            np.einsum(
                "i,j,ijx->",
                coefficient[shell_pair.indices_i],
                coefficient[shell_pair.indices_j],
                shell_pair.attenuation,
            )
            * prefactor
        )
        return bool(np.max(np.abs(quantity)) >= threshold)

    def iter_shells(self) -> Iterable[ShellView]:
        """Iterate over contracted shell views."""
        starts = self.contracted_shell_starts(slice(None))
        stops = np.append(starts[1:], self.num_primitive_shells)
        for start, stop in zip(starts, stops):
            shell_slice = slice(int(start), int(stop))
            yield self[shell_slice]

    def iter_shell_pairs(
        self, unique: bool = False, screening: Screening[Literal[2]] = NoScreening()
    ) -> Iterable[ShellPairView]:
        """Iterate over contracted shell-pair views."""
        shells = list(self.iter_shells())
        symmetry = (SymmetryN2S2 if unique else SymmetryN2S1)((len(shells), len(shells)))
        for (i, j), _ in symmetry.canonical_indices():
            shell_pair = self[shells[i], shells[j]]
            assert isinstance(shell_pair, ShellPairView)
            yield from screening(shell_pair)

    def iter_shell_groups(self) -> Iterable[ShellView]:
        """Iterate over contiguous shell groups with matching angular momentum."""
        first_indices = self.contracted_shell_starts(slice(None))
        stops = np.append(first_indices[1:], self.num_primitive_shells)
        start = None
        stop = None
        angular_momentum = None
        for first_index, last_index in zip(first_indices, stops):
            shell_start = int(first_index)
            shell_stop = int(last_index)
            shell_angular_momentum = self.shells.angular_momentum[shell_start]
            if start is not None and shell_angular_momentum != angular_momentum:
                yield self[slice(start, stop)]
                start = None
            if start is None:
                start = shell_start
            stop = shell_stop
            angular_momentum = shell_angular_momentum
        if start is not None:
            yield self[slice(start, stop)]

    def iter_shell_pair_groups(
        self, unique: bool = False, screening: Screening[Literal[2]] = NoScreening()
    ) -> Iterable[ShellPairView]:
        """Iterate over shell-pair rectangles with matching angular momenta on each side.

        If screened, each rectangle contains only shell pairs that pass screening.
        """
        shell_groups = list(self.iter_shell_groups())
        symmetry = (SymmetryN2S2 if unique else SymmetryN2S1)((len(shell_groups), len(shell_groups)))
        for (i, j), _ in symmetry.canonical_indices():
            shell_pair = self[shell_groups[i], shell_groups[j]]
            assert isinstance(shell_pair, ShellPairView)
            yield from screening(shell_pair)

    @lru_cache(maxsize=None)
    def ordered_by_angular_momentum(self, spherical: bool = False) -> tuple[Basis, array[tuple[int], int64]]:
        """Return a copy of the basis ordered by angular momentum.

        Args:
            spherical: Build the AO permutation for real spherical functions.

        Returns:
            A new ``Basis`` object with contracted shells ordered by angular momentum, and AO indices
            that restore the original basis order.

        Note:
            This method is cached on the parent ``Basis`` object, so repeated calls will return the
            same object to admit caching of derived quantities. With the simple ``lru_cache``, this
            does not support caching within nested calls.
        """
        first_indices = self.contracted_shell_starts(slice(None))
        shell_ids = self.shells.shell_indices[first_indices]
        stops = np.append(first_indices[1:], self.num_primitive_shells)
        shell_groups = [np.arange(start, stop) for start, stop in zip(first_indices, stops)]
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
            return ShellView(self, key)
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, slice) for k in key):
            assert isinstance(key[0], slice) and isinstance(key[1], slice)  # for mypy
            shell_i, shell_j = self[key[0]], self[key[1]]
            assert isinstance(shell_i, ShellView) and isinstance(shell_j, ShellView)
            return ShellPairView(shell_i, shell_j)
        elif isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, ShellView) for k in key):
            assert isinstance(key[0], ShellView) and isinstance(key[1], ShellView)  # for mypy
            if key[0].basis is not key[1].basis or key[0].basis is not self:
                raise ValueError("shell views must belong to this basis")
            return ShellPairView(key[0], key[1])
        raise TypeError(f"invalid key type: {type(key)}")

    def __len__(self) -> int:
        return self.num_primitive_shells


@dataclass(frozen=True, eq=False, slots=True)
class ShellView:
    """View of a shell."""

    basis: Basis
    indices: slice

    @property
    def position(self) -> array[tuple[int, int], float64]:
        """Primitive centers for the selected shell."""
        return self.basis.shells.position[self.indices]

    @property
    def exponent(self) -> array[tuple[int], float64]:
        """Primitive exponents for the selected shell."""
        return self.basis.shells.exponent[self.indices]

    @property
    def coefficient(self) -> array[tuple[int], float64]:
        """Contraction coefficients for the selected primitive shells."""
        return self.basis.shells.coefficient[self.indices]

    @property
    def angular_momentum(self) -> int:
        """Angular momentum of the selected shell."""
        return int(self.basis.shells.angular_momentum[self.indices][0])

    @property
    def cartesian_ao_slice(self) -> slice:
        """Slice of contracted Cartesian atomic orbitals for the selected shell."""
        return self.basis.cartesian_ao_slice(self.indices)

    @property
    def spherical_ao_slice(self) -> slice:
        """Slice of contracted real spherical atomic orbitals for the selected shell."""
        return self.basis.spherical_ao_slice(self.indices)

    @property
    def num_shells(self) -> int:
        """Number of contracted shells in the view."""
        return self.basis.num_contracted_shells_in(self.indices)

    def iter_masked_views(self, mask: array[tuple[int], bool_]) -> Iterable[ShellView]:
        """Yield contiguous shell views selected by a one-dimensional mask."""
        if mask.shape != (self.num_shells,):
            raise ValueError(f"mask must have shape ({self.num_shells},)")

        shell_slices = _shell_slices(self)
        for start, stop in _true_runs(mask):
            yield self.basis[slice(shell_slices[start].start, shell_slices[stop - 1].stop)]

    @property
    def rank(self) -> int:
        """Rank of the view (always 1 for a shell)."""
        return 1

    def __len__(self) -> int:
        start = self.indices.start or 0
        stop = self.indices.stop or self.basis.num_primitive_shells
        return int(stop - start)

    def __hash__(self) -> int:
        return hash((self.basis, self.indices))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShellView):
            return NotImplemented
        return self.basis is other.basis and self.indices == other.indices


def _shell_slices(view: ShellView) -> list[slice]:
    """Return primitive slices for each contracted shell in a view."""
    start, stop, step = view.indices.indices(view.basis.num_primitive_shells)
    if step != 1:
        raise ValueError("shell slices must be contiguous")
    starts = view.basis.contracted_shell_starts(view.indices) + start
    boundaries = [*(int(offset) for offset in starts), stop]
    return [slice(first, last) for first, last in zip(boundaries[:-1], boundaries[1:])]


def _individual_shell_views(view: ShellView) -> Iterable[ShellView]:
    """Yield one view for each contracted shell in a group."""
    for shell_slice in _shell_slices(view):
        yield view.basis[shell_slice]


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


@dataclass(frozen=True, eq=False, slots=True)
class ShellPairView:
    """View of a shell pair."""

    shell_i: ShellView
    shell_j: ShellView

    def __post_init__(self) -> None:
        if self.shell_i.basis is not self.shell_j.basis:
            raise ValueError("shell views must belong to the same basis")

    @property
    def basis(self) -> Basis:
        """Basis containing both shell views."""
        return self.shell_i.basis

    @property
    def shell_pairs(self) -> ShellPairs:
        """Primitive pair data associated with the views."""
        return self.basis.shell_pairs

    @property
    def indices_i(self) -> slice:
        """Primitive-shell slice for the first view."""
        return self.shell_i.indices

    @property
    def indices_j(self) -> slice:
        """Primitive-shell slice for the second view."""
        return self.shell_j.indices

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
            shell_i = self.basis[slice(shells_i[row_start].start, shells_i[row_stop - 1].stop)]
            shell_j = self.basis[slice(shells_j[column_start].start, shells_j[column_stop - 1].stop)]
            assert isinstance(shell_i, ShellView) and isinstance(shell_j, ShellView)
            return ShellPairView(shell_i, shell_j)

        for row, row_mask in enumerate(mask):
            intervals = _true_runs(row_mask)
            current = set(intervals)
            for interval in sorted(active.keys() - current):
                yield append_rectangle(active.pop(interval), row, interval)
            for interval in intervals:
                active.setdefault(interval, row)

        for interval, row_start in sorted(active.items()):
            yield append_rectangle(row_start, shape[0], interval)

    @property
    def rank(self) -> int:
        """Rank of the view (always 2 for a shell pair)."""
        return 2

    def __len__(self) -> int:
        start_i = self.indices_i.start or 0
        stop_i = self.indices_i.stop or self.shell_pairs.position.shape[0]
        start_j = self.indices_j.start or 0
        stop_j = self.indices_j.stop or self.shell_pairs.position.shape[1]
        return int((stop_i - start_i) * (stop_j - start_j))

    def __hash__(self) -> int:
        return hash((self.shell_i, self.shell_j))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShellPairView):
            return NotImplemented
        return self.shell_i == other.shell_i and self.shell_j == other.shell_j
