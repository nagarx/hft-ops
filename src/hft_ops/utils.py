"""
Shared utility functions for hft-ops.

Small, stateless helpers used by multiple modules.
"""

from __future__ import annotations

from typing import Any, Dict


def get_nested(data: Dict[str, Any], dotted_key: str) -> Any:
    """Walk a nested dict by dotted key path.

    Args:
        data: Nested dictionary.
        dotted_key: Key path with dots (e.g., "stages.training.config").

    Returns:
        The value at the path, or None if any segment is missing.

    Example:
        >>> get_nested({"a": {"b": 1}}, "a.b")
        1
        >>> get_nested({"a": {"b": 1}}, "a.c") is None
        True
    """
    parts = dotted_key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def set_nested(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict by dotted key path, creating intermediates.

    Args:
        data: Mutable nested dictionary.
        dotted_key: Key path with dots (e.g., "model.hidden_dim").
        value: Value to set.

    Example:
        >>> d = {}
        >>> set_nested(d, "a.b.c", 42)
        >>> d
        {'a': {'b': {'c': 42}}}
    """
    parts = dotted_key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
