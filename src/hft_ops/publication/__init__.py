"""Immutable publication authority and generation-control primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hft_ops.publication.development_authority import (
        AuthorityIssueError,
        IssuedDevelopmentAuthority,
        issue_schema4_development_authority,
    )
    from hft_ops.publication.feature_carrier_admission import (
        FeatureCarrierAdmissionIssueError,
        IssuedFeatureCarrierAdmissionV1,
        issue_feature_carrier_admission,
    )

__all__ = [
    "AuthorityIssueError",
    "IssuedDevelopmentAuthority",
    "issue_schema4_development_authority",
    "FeatureCarrierAdmissionIssueError",
    "IssuedFeatureCarrierAdmissionV1",
    "issue_feature_carrier_admission",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "FeatureCarrierAdmissionIssueError",
        "IssuedFeatureCarrierAdmissionV1",
        "issue_feature_carrier_admission",
    }:
        from hft_ops.publication import feature_carrier_admission

        return getattr(feature_carrier_admission, name)
    from hft_ops.publication import development_authority

    return getattr(development_authority, name)
