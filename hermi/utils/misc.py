"""Miscellaneous functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Never


class DefaultHash:
    """Base class for objects with the default ``object.__hash__`` and ``object.__eq__``."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultHash):
            return NotImplemented
        return object.__eq__(self, other)

    def __hash__(self) -> int:
        return object.__hash__(self)


def update_if_missing(d: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Update a dictionary with new key-value pairs if the keys are not already present.

    Args:
        d: Dictionary to update.
        kwargs: Key-value pairs to add to the dictionary.

    Returns:
        Updated dictionary with new key-value pairs added if the keys were not already present.
    """
    out = d.copy()
    for key, value in kwargs.items():
        if key not in out:
            out[key] = value
    return out


def unreachable() -> Never:
    """Raise an error indicating that the code is unreachable."""
    raise AssertionError("unreachable code, this should never happen")
