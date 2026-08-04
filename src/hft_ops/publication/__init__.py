"""Immutable publication authority and generation-control primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hft_ops.publication.development_authority import (
        AuthorityIssueError,
        IssuedDevelopmentAuthority,
        issue_schema4_development_authority,
    )

__all__ = [
    "AuthorityIssueError",
    "IssuedDevelopmentAuthority",
    "issue_schema4_development_authority",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from hft_ops.publication import development_authority

    return getattr(development_authority, name)
