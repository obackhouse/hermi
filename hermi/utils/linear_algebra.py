"""Array operations and linear algebra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy import float64

    from hermi._typing import array


def outer_powers(
    base: array[tuple[int, ...], float64],
    max_order: int,
) -> array[tuple[int, ...], float64]:
    """Compute consecutive integer powers of each base value, optionally scaled.

    Args:
        base: Values whose powers to compute.
        max_order: Largest nonnegative integer power.

    Returns:
        Consecutive powers from zero through ``max_order``.
    """
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    elif max_order == 0:
        return np.ones((*base.shape, 1), dtype=np.float64)
    elif max_order == 1:
        return np.stack((np.ones_like(base), base), axis=-1)
    powers = np.ones((*base.shape, max_order + 1), dtype=np.float64)
    powers[..., 1] = base
    for order in range(2, max_order + 1):
        powers[..., order] = powers[..., order - 1] * base
    return powers
