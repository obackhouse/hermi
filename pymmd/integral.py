"""Integral utilities."""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import TYPE_CHECKING, overload

import numpy as np

from pymmd.hermite import hermite_indices
from pymmd.util import add_reduceat, num_cartesian_functions, num_spherical_functions

if TYPE_CHECKING:
    from typing import Literal

    from numpy import float64, int64

    from pymmd._typing import array
    from pymmd.basis import ShellPairView, ShellView


@overload
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: None = None,
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[array[tuple[int, int], int64], array[tuple[int], float64]]: ...


@overload
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: tuple[tuple[int, int, int], ...],
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[array[tuple[int, int, int], int64], array[tuple[int], float64]]: ...


@lru_cache(maxsize=None)
def hermite_operator_matrix_indices(
    angular_momentum_i: int,
    angular_momentum_j: int,
    *,
    shifts: tuple[tuple[int, int, int], ...] | None = None,
    sign_side: Literal["bra", "ket"] = "ket",
) -> tuple[
    array[tuple[int, int], int64] | array[tuple[int, int, int], int64],
    array[tuple[int], float64],
]:
    r"""Map pairs of Hermite indices to operator indices and return parity signs.

    Args:
        angular_momentum_i: Maximum Hermite order for the first expansion.
        angular_momentum_j: Maximum Hermite order for the second expansion.
        shifts: Hermite index shifts, or ``None`` for the unshifted mapping.
        sign_side: Expansion side whose Hermite indices determine the signs.

    Returns:
        A single index map for ``None``, or one map per shift, and parity signs.
    """
    single_shift = shifts is None
    if shifts is None:
        shifts = ((0, 0, 0),)
    if not shifts:
        raise ValueError("at least one Hermite index shift is required")
    if any(component < 0 for shift in shifts for component in shift):
        raise ValueError("Hermite index shifts must be nonnegative")
    if sign_side not in ("bra", "ket"):
        raise ValueError(f"invalid sign side: {sign_side!r}")

    max_order = angular_momentum_i + angular_momentum_j + max(map(sum, shifts))
    hermite_i = np.asarray(hermite_indices(angular_momentum_i), dtype=np.int64)
    hermite_j = np.asarray(hermite_indices(angular_momentum_j), dtype=np.int64)
    hermite_all = np.asarray(hermite_indices(max_order), dtype=np.int64)

    hermite_map = np.full((max_order + 1,) * 3, -1, dtype=np.int64)
    hermite_map[tuple(hermite_all.T)] = np.arange(len(hermite_all), dtype=np.int64)

    hermite_sums = (
        hermite_i[:, None, None, :] + hermite_j[None, :, None, :] + np.asarray(shifts, dtype=np.int64)[None, None, :, :]
    )
    indices = hermite_map[tuple(hermite_sums.transpose(3, 2, 0, 1))]
    if single_shift:
        indices = indices[0]

    signed_indices = hermite_i if sign_side == "bra" else hermite_j
    signs = np.where(np.sum(signed_indices, axis=-1) % 2 == 0, 1.0, -1.0)

    return indices, signs


def _reduce_primitive[S: tuple[int, ...]](values: array[S, float64], shell: ShellView, axis: int) -> array[S, float64]:
    """Reduce a primitive shell axis via summation to contracted shells."""
    expanded = values.reshape(
        *values.shape[:axis],
        len(shell),
        *values.shape[axis + 1 :],
    )
    expanded = add_reduceat(expanded, shell.basis.contracted_shell_starts(shell.indices), axis=axis)
    expanded = expanded.reshape(
        *values.shape[:axis],
        shell.num_shells,
        *values.shape[axis + 1 :],
    )
    return expanded  # type: ignore[return-value]


def _reduce_primitive_pair[S: tuple[int, ...]](
    values: array[S, float64], pair: ShellPairView, axis: int
) -> array[S, float64]:
    """Reduce a primitive shell-pair axis via summation to contracted shells."""
    expanded = values.reshape(
        *values.shape[:axis],
        len(pair.shell_i),
        len(pair.shell_j),
        *values.shape[axis + 1 :],
    )
    expanded = add_reduceat(expanded, pair.basis.contracted_shell_starts(pair.shell_j.indices), axis=axis + 1)
    expanded = add_reduceat(expanded, pair.basis.contracted_shell_starts(pair.shell_i.indices), axis=axis)
    expanded = expanded.reshape(
        *values.shape[:axis],
        pair.shell_i.num_shells * pair.shell_j.num_shells,
        *values.shape[axis + 1 :],
    )
    return expanded  # type: ignore[return-value]


def _unpack_from_shell_pair_major[S: tuple[int, ...]](
    values: array[S, float64], *shells: ShellView, spherical: bool = False
) -> array[S, float64]:
    """Unpack a shell-pair-major array to an interleaved array of angular momentum functions."""
    num_functions = num_spherical_functions if spherical else num_cartesian_functions
    num_ang = tuple(num_functions(shell.angular_momentum) for shell in shells)
    num_shells = tuple(shell.num_shells for shell in shells)
    transpose = sum(((i, i + len(shells)) for i in range(len(shells))), ())
    expanded = values.reshape(*num_shells, *num_ang)
    expanded = expanded.transpose(transpose)
    return expanded.reshape(tuple(map(prod, zip(num_shells, num_ang))))  # type: ignore[return-value]
