"""Experiment ledger: append-only storage, dedup, and comparison."""

# M8 (2026-06-01): canonical hft_contracts homes, NOT the local Phase-6
# `experiment_record` shim, so `import hft_ops.ledger` is DeprecationWarning-free
# at import time. The shim still warns for direct legacy importers. Same objects.
from hft_contracts.experiment_record import ExperimentRecord
from hft_contracts.provenance import Provenance
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.ledger.dedup import compute_fingerprint, check_duplicate
from hft_ops.ledger.comparator import compare_experiments, diff_experiments

__all__ = [
    "ExperimentRecord",
    "Provenance",
    "ExperimentLedger",
    "compute_fingerprint",
    "check_duplicate",
    "compare_experiments",
    "diff_experiments",
]
