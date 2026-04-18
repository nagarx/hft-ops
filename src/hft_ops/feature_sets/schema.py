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

Phase 6 post-validation (2026-04-18): lazy ``__getattr__`` emits
DeprecationWarning once per symbol access. This shim is scheduled for
removal in version 0.4.0 — migrate call sites before then.

See PIPELINE_ARCHITECTURE.md §14.8 + §17.3.
"""

from __future__ import annotations

import warnings as _warnings

_CANONICAL_MODULE = "hft_contracts.feature_sets.schema"
_REMOVAL_VERSION = "0.4.0"
_PUBLIC_NAMES = frozenset({
    "FEATURE_SET_SCHEMA_VERSION",
    "FeatureSet",
    "FeatureSetAppliesTo",
    "FeatureSetIntegrityError",
    "FeatureSetProducedBy",
    "FeatureSetRef",
    "FeatureSetValidationError",
    "validate_feature_set_dict",
})
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol."""
    if name in _PUBLIC_NAMES:
        if name not in _WARNED:
            _WARNED.add(name)
            _warnings.warn(
                f"`hft_ops.feature_sets.schema.{name}` is a Phase 6 6B.3 "
                f"re-export shim. Migrate to "
                f"`from {_CANONICAL_MODULE} import {name}` (or the "
                f"convenience path `from hft_contracts.feature_sets import "
                f"{name}`) before the {_REMOVAL_VERSION} removal. "
                f"(This warning fires once per symbol per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        return getattr(importlib.import_module(_CANONICAL_MODULE), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = sorted(_PUBLIC_NAMES)
