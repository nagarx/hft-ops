"""
Provenance capture re-export shim (Phase 6 6B.4, 2026-04-17).

The authoritative source lives in `hft_contracts.provenance` so the
`Provenance` dataclass can be embedded in `ExperimentRecord` (6B.1) and
imported cross-module without a hard dependency on hft-ops. This module
exists solely for backward compatibility with pre-6B.4 imports:

    from hft_ops.provenance.lineage import Provenance       # still works
    from hft_ops.provenance.lineage import build_provenance  # still works

New code should import directly from hft_contracts:

    from hft_contracts.provenance import Provenance, build_provenance

See PIPELINE_ARCHITECTURE.md §17.3 for the producer→consumer matrix and
the rationale for placing Provenance in the contract plane.
"""

from __future__ import annotations

# Re-export the full public surface. The star import + __all__ contract
# locks the names so sibling modules (experiment_record.py, cli.py,
# feature_sets/producer.py, ledger/dedup.py) keep working.
from hft_contracts.provenance import (
    GitInfo,
    NOT_GIT_TRACKED_SENTINEL,
    PROVENANCE_SCHEMA_VERSION,
    Provenance,
    build_provenance,
    capture_git_info,
    hash_config_dict,
    hash_directory_manifest,
    hash_file,
)

__all__ = [
    "GitInfo",
    "NOT_GIT_TRACKED_SENTINEL",
    "PROVENANCE_SCHEMA_VERSION",
    "Provenance",
    "build_provenance",
    "capture_git_info",
    "hash_config_dict",
    "hash_directory_manifest",
    "hash_file",
]
