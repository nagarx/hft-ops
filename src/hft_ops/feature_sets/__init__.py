"""
hft-ops FeatureSet registry module (Phase 4).

Public re-exports for the six submodules. Import layout is deliberately
flat so callers can write ``from hft_ops.feature_sets import FeatureSet``
rather than chasing the submodule split.

Submodule responsibilities:
    - ``hashing``: content-addressed SHA-256 over (indices, source_feature_count, contract_version).
    - ``schema``:  FeatureSet / FeatureSetRef / FeatureSetAppliesTo /
                   FeatureSetProducedBy dataclasses + validators.
    - ``writer``:  atomic JSON writes with refuse-overwrite + idempotent-on-match.
    - ``registry``: read-side queries (list, get, verify).
    - ``producer``: evaluator → FeatureSet orchestration.
"""

from hft_ops.feature_sets.hashing import (
    compute_feature_set_hash,
    _sanitize_for_hash,
)
from hft_ops.feature_sets.producer import (
    EvaluatorNotInstalled,
    NoFeaturesSelectedError,
    produce_feature_set,
)
from hft_ops.feature_sets.registry import (
    FeatureSetNotFound,
    FeatureSetRegistry,
)
from hft_ops.feature_sets.schema import (
    FEATURE_SET_SCHEMA_VERSION,
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetRef,
    FeatureSetValidationError,
    validate_feature_set_dict,
)
from hft_ops.feature_sets.writer import (
    AtomicWriteError,
    FeatureSetExists,
    atomic_write_json,
    write_feature_set,
)

__all__ = [
    # hashing
    "compute_feature_set_hash",
    "_sanitize_for_hash",
    # schema
    "FEATURE_SET_SCHEMA_VERSION",
    "FeatureSet",
    "FeatureSetAppliesTo",
    "FeatureSetIntegrityError",
    "FeatureSetProducedBy",
    "FeatureSetRef",
    "FeatureSetValidationError",
    "validate_feature_set_dict",
    # writer
    "AtomicWriteError",
    "FeatureSetExists",
    "atomic_write_json",
    "write_feature_set",
    # registry
    "FeatureSetNotFound",
    "FeatureSetRegistry",
    # producer
    "EvaluatorNotInstalled",
    "NoFeaturesSelectedError",
    "produce_feature_set",
]
