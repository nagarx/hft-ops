"""
Experiment record re-export shim (Phase 6 6B.1a, 2026-04-17).

The authoritative source lives in `hft_contracts.experiment_record`. This
shim preserves backward compatibility for every existing importer
(cli.py, dashboard script, ledger writer, dedup, multiple tests).

    from hft_ops.ledger.experiment_record import ExperimentRecord  # still works
    from hft_ops.ledger.experiment_record import RecordType        # still works

New code should import from hft_contracts:

    from hft_contracts.experiment_record import ExperimentRecord, RecordType

Phase 6B.1a (narrow move) is the dataclass-only phase. Phase 6B.1b
(deferred to Phase 7) retires `lobtrainer.experiments.ExperimentRegistry`
and ships the migration CLI that rewrites trainer-local experiment JSON
into ExperimentRecord shapes — see Phase 6 plan §Final Validation
Corrections BLOCKER 5 for the LOC revision (~900 LOC, not ~400).

See PIPELINE_ARCHITECTURE.md §17.3 for the producer→consumer matrix.
"""

from __future__ import annotations

from hft_contracts.experiment_record import ExperimentRecord, RecordType
# Re-export Provenance for the back-compat path `hft_ops.ledger.__init__.py`
# uses (and any downstream consumer that imported Provenance via the ledger
# module pre-6B.1a).
from hft_contracts.provenance import Provenance

__all__ = [
    "ExperimentRecord",
    "Provenance",
    "RecordType",
]
