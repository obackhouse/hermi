"""Typing."""

from __future__ import annotations

from numpy import ndarray, float64, int64, dtype, generic


type array[S: tuple[int, ...], T: generic] = ndarray[S, dtype[T]]
