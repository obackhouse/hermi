"""Slice utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator


def rebase_slice(source: slice, target: slice) -> slice:
    """Rebase a source slice to a target slice.

    Args:
        source: Slice to rebase.
        target: Slice to rebase onto.

    Returns:
        Rebasing of the source slice onto the target slice.
    """
    start = source.start - target.start if source.start is not None else None
    stop = source.stop - target.start if source.stop is not None else None
    return slice(start, stop, source.step)


def iter_slices(start: int, stop: int, step: int) -> Generator[slice]:
    """Iterate over slices of a given step size.

    Args:
        start: Start index.
        stop: Stop index.
        step: Step size.

    Yields:
        Tuples of (start, stop) indices for each slice.
    """
    for i in range(start, stop, step):
        yield slice(i, min(i + step, stop))
