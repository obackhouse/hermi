"""Molecular structure."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants

from hermi.util import DefaultHash

if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy import float64, int64

    from hermi._typing import array


@dataclass(frozen=True, eq=False)
class Element(DefaultHash):
    """An element."""

    symbol: str
    atomic_charge: int
    name: str
    shells: list[int]


@dataclass(frozen=True, eq=False)
class PeriodicTable(DefaultHash):
    """Periodic table."""

    entries: list[Element]

    @classmethod
    @lru_cache(maxsize=1)
    def from_json(cls, json_file: str) -> PeriodicTable:
        """Load periodic table from JSON file."""
        with open(json_file, "r") as f:
            data = json.load(f)
        entries = [Element(**entry) for entry in data]
        return cls(entries=entries)

    def find(self, **kwargs: Any) -> Element:
        """Find an element by its attributes."""
        for entry in self.entries:
            if all(getattr(entry, k) == v for k, v in kwargs.items()):
                return entry
        raise ValueError(f"no element found with attributes {kwargs}")


@dataclass(frozen=True, eq=False)
class Molecule(DefaultHash):
    """Molecular structure."""

    #: Element for each atom
    elements: list[Element]

    #: Positions of the atoms
    positions: array[tuple[int, int], float64]

    def __post_init__(self) -> None:
        if len(self.elements) != self.positions.shape[0]:
            raise ValueError(
                f"number of elements ({len(self.elements)}) does not match number of coordinates "
                f"({self.positions.shape[0]})"
            )
        if self.positions.shape[1] != 3:
            raise ValueError(f"coordinates must have shape (N, 3), but got {self.positions.shape}")

    @classmethod
    def from_xyz(self, content: list[str] | str, units: Literal["angstrom", "bohr"] = "angstrom") -> Molecule:
        """Load molecule from XYZ file."""
        path = os.path.join(os.path.dirname(__file__), "periodic_table.json")
        periodic_table = PeriodicTable.from_json(path)
        if isinstance(content, str):
            content = content.splitlines()
        n_atoms = int(content[0])
        elements = []
        coordinates = []
        for line in content[2 : 2 + n_atoms]:
            parts = line.split()
            elements.append(periodic_table.find(symbol=parts[0]))
            coordinates.append(map(float, parts[1:4]))
        positions: array[tuple[int, int], float64] = np.array(coordinates, dtype=np.float64)
        if units == "angstrom":
            positions *= constants.value("Bohr radius") / constants.value("Angstrom")
        return Molecule(elements=elements, positions=positions)

    @cached_property
    def symbols(self) -> list[str]:
        """Element symbols."""
        return [element.symbol for element in self.elements]

    @cached_property
    def charges(self) -> array[tuple[int], int64]:
        """Atomic charges."""
        return np.array([element.atomic_charge for element in self.elements], dtype=np.int64)

    @cached_property
    def names(self) -> list[str]:
        """Element names."""
        return [element.name for element in self.elements]

    @cached_property
    def shells(self) -> list[list[int]]:
        """Electron shells."""
        return [element.shells for element in self.elements]

    def __len__(self) -> int:
        return len(self.elements)
