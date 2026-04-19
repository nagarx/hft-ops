"""
Provenance capture re-export shim (Phase 6 6B.4, 2026-04-17).

The authoritative source lives in ``hft_contracts.provenance``. This shim
preserves backward compatibility for pre-6B.4 imports:

    from hft_ops.provenance.lineage import Provenance       # still works
    from hft_ops.provenance.lineage import build_provenance  # still works

New code should import directly from hft_contracts:

    from hft_contracts.provenance import Provenance, build_provenance

Phase 6 post-validation (2026-04-18): lazy ``__getattr__`` emits
DeprecationWarning once per symbol access to prevent shim ossification.
Python's default filter hides DeprecationWarning outside ``__main__``;
pytest surfaces it; explicit `-W error::DeprecationWarning` errors. This
shim is scheduled for removal in version 0.4.0 — migrate call sites to
``hft_contracts.provenance`` imports before then.

See PIPELINE_ARCHITECTURE.md §17.3 and the Phase 7 migration guide at
``docs/phase7_roadmap.md`` (if present).
"""

from __future__ import annotations

import warnings as _warnings

_CANONICAL_MODULE = "hft_contracts.provenance"
# Calendar-driven shim deadline (Phase 7 post-validation, 2026-04-19):
# matches the `feature_preset` deprecation pattern (calendar date, NOT
# version). Version-driven deadlines ("0.4.0") were ambiguous because
# hft-ops has no fixed release cadence. A calendar date gives consumers
# a concrete migration window. 6-month grace period from Phase 6 6B.4
# (2026-04-17) → removal deadline 2026-10-31 (end-of-month alignment).
_REMOVAL_DATE = "2026-10-31"
_PUBLIC_NAMES = frozenset({
    "GitInfo",
    "NOT_GIT_TRACKED_SENTINEL",
    "PROVENANCE_SCHEMA_VERSION",
    "Provenance",
    "build_provenance",
    "capture_git_info",
    "hash_config_dict",
    "hash_directory_manifest",
    "hash_file",
})
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol.

    Python calls ``__getattr__`` only for names NOT in ``module.__dict__``,
    so symbols resolve through the canonical module on every direct access.
    The class identity is preserved — ``hft_ops.provenance.lineage.Provenance
    is hft_contracts.provenance.Provenance`` remains ``True``.
    """
    if name in _PUBLIC_NAMES:
        if name not in _WARNED:
            _WARNED.add(name)
            _warnings.warn(
                f"`hft_ops.provenance.lineage.{name}` is a Phase 6 6B.4 "
                f"re-export shim. Migrate to "
                f"`from {_CANONICAL_MODULE} import {name}` before the "
                f"{_REMOVAL_DATE} removal deadline. "
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
