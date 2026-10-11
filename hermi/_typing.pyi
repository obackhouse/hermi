"""Typing."""

from __future__ import annotations

from typing import Protocol, Self

from numpy import ndarray, float64, int64, dtype, generic


type array[S: tuple[int, ...], T: generic] = ndarray[S, dtype[T]]


class Comparable(Protocol):
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: Self) -> bool: ...
