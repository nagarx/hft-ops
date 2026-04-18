"""
FeatureSet schema re-export shim (Phase 6 6B.3, 2026-04-17).

The authoritative source lives in ``hft_contracts.feature_sets.schema``.
This shim preserves backward compatibility for pre-6B.3 imports:

    from hft_ops.feature_sets.schema import FeatureSet           # still works
    from hft_ops.feature_sets.schema import FeatureSetRef        # still works
    from hft_ops.feature_sets.schema import FeatureSetAppliesTo  # still works
    ...

New code should import from hft_contracts:

    from hft_contracts.feature_sets import FeatureSet, FeatureSetRef
    # or from the specific submodule:
    from hft_contracts.feature_sets.schema import FeatureSet

See PIPELINE_ARCHITECTURE.md §17.3 + Phase 6 plan §6B.3.
"""

from __future__ import annotations

from hft_contracts.feature_sets.schema import (
    FEATURE_SET_SCHEMA_VERSION,
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetRef,
    FeatureSetValidationError,
    validate_feature_set_dict,
)

__all__ = [
    "FEATURE_SET_SCHEMA_VERSION",
    "FeatureSet",
    "FeatureSetAppliesTo",
    "FeatureSetIntegrityError",
    "FeatureSetProducedBy",
    "FeatureSetRef",
    "FeatureSetValidationError",
    "validate_feature_set_dict",
]
