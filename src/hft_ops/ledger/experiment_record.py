"""
Experiment record re-export shim (Phase 6 6B.1a, 2026-04-17).

The authoritative source lives in ``hft_contracts.experiment_record``.
This shim preserves backward compatibility for every existing importer
(cli.py, dashboard, ledger writer, dedup, tests).

    from hft_ops.ledger.experiment_record import ExperimentRecord  # still works
    from hft_ops.ledger.experiment_record import RecordType        # still works
    from hft_ops.ledger.experiment_record import Provenance        # still works (re-export)

New code should import from hft_contracts:

    from hft_contracts.experiment_record import ExperimentRecord, RecordType
    from hft_contracts.provenance import Provenance

Phase 6 post-validation (2026-04-18): lazy ``__getattr__`` emits
DeprecationWarning once per symbol access. This shim is scheduled for
removal in version 0.4.0 — migrate call sites before then.

Phase 6B.1a = dataclass-only narrow move. Phase 7 6B.1b retires
``lobtrainer.experiments.ExperimentRegistry`` entirely and ships the
migration CLI rewriting trainer-local ``experiments/*.json`` into
``ExperimentRecord`` shapes. See Phase 6 plan §Final Validation
Corrections BLOCKER 5 for the LOC revision (~900 LOC).

See PIPELINE_ARCHITECTURE.md §17.3.
"""

from __future__ import annotations

import warnings as _warnings

_REMOVAL_VERSION = "0.4.0"
# Map symbol name → (canonical module path) so re-exports remain explicit.
_SYMBOL_HOMES = {
    "ExperimentRecord": "hft_contracts.experiment_record",
    "RecordType": "hft_contracts.experiment_record",
    # Provenance is re-exported here because hft_ops/ledger/__init__.py
    # imports it from this module (back-compat path pre-6B.4). New code
    # should import from hft_contracts.provenance directly.
    "Provenance": "hft_contracts.provenance",
}
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol."""
    if name in _SYMBOL_HOMES:
        canonical = _SYMBOL_HOMES[name]
        if name not in _WARNED:
            _WARNED.add(name)
            _warnings.warn(
                f"`hft_ops.ledger.experiment_record.{name}` is a "
                f"Phase 6 6B.1a re-export shim. Migrate to "
                f"`from {canonical} import {name}` before the "
                f"{_REMOVAL_VERSION} removal. "
                f"(This warning fires once per symbol per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        return getattr(importlib.import_module(canonical), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = sorted(_SYMBOL_HOMES)
