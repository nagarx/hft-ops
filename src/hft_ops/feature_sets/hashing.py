"""
Content-addressed hashing re-export shim (Phase 6 6B.3, 2026-04-17).

The authoritative source lives in ``hft_contracts.feature_sets.hashing``.
This shim preserves backward compatibility for pre-6B.3 imports:

    from hft_ops.feature_sets.hashing import compute_feature_set_hash  # still works
    from hft_ops.feature_sets.hashing import _sanitize_for_hash        # still works

New code should import from hft_contracts:

    from hft_contracts.feature_sets.hashing import compute_feature_set_hash
    # or the convenience path:
    from hft_contracts.feature_sets import compute_feature_set_hash
    # and for the canonical-form primitive:
    from hft_contracts.canonical_hash import sanitize_for_hash

Phase 6 post-validation (2026-04-18): lazy ``__getattr__`` emits
DeprecationWarning once per symbol access. Scheduled for removal in
on ``_REMOVAL_DATE`` (``2026-10-31``).

See PIPELINE_ARCHITECTURE.md §17.3.
"""

from __future__ import annotations

import warnings as _warnings

# Calendar-driven shim deadline (Phase 7 post-validation, 2026-04-19).
# See provenance/lineage.py for rationale — 6 months from Phase 6 6B.3.
_REMOVAL_DATE = "2026-10-31"
_SYMBOL_HOMES = {
    "compute_feature_set_hash": "hft_contracts.feature_sets.hashing",
    # `_sanitize_for_hash` was a re-export alias of
    # `hft_contracts.canonical_hash.sanitize_for_hash`. Preserved here
    # for pre-6B.3 importers; new code should import directly.
    "_sanitize_for_hash": "hft_contracts.canonical_hash",
}
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol."""
    if name in _SYMBOL_HOMES:
        canonical_module = _SYMBOL_HOMES[name]
        if name not in _WARNED:
            _WARNED.add(name)
            # Resolve the canonical attribute name (strip leading underscore
            # for `_sanitize_for_hash` which is re-exported as
            # `sanitize_for_hash` in hft_contracts.canonical_hash).
            canonical_attr = (
                "sanitize_for_hash" if name == "_sanitize_for_hash" else name
            )
            _warnings.warn(
                f"`hft_ops.feature_sets.hashing.{name}` is a Phase 6 6B.3 "
                f"re-export shim. Migrate to "
                f"`from {canonical_module} import {canonical_attr}` "
                f"before the {_REMOVAL_DATE} removal deadline. "
                f"(This warning fires once per symbol per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        module = importlib.import_module(canonical_module)
        canonical_attr = (
            "sanitize_for_hash" if name == "_sanitize_for_hash" else name
        )
        return getattr(module, canonical_attr)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = sorted(_SYMBOL_HOMES)
