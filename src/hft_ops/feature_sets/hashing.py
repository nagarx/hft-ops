"""
Content-addressed hashing re-export shim (Phase 6 6B.3, 2026-04-17).

The authoritative source lives in ``hft_contracts.feature_sets.hashing``.
This shim exists for backward compatibility with pre-6B.3 imports:

    from hft_ops.feature_sets.hashing import compute_feature_set_hash  # still works

New code should import from hft_contracts:

    from hft_contracts.feature_sets.hashing import compute_feature_set_hash
    # or from the package __init__:
    from hft_contracts.feature_sets import compute_feature_set_hash

The ``_sanitize_for_hash`` name was a module-level backward-compat
re-export of the canonical-hash primitive; new code should import it
from ``hft_contracts.canonical_hash`` directly.

See PIPELINE_ARCHITECTURE.md §17.3 + Phase 6 plan §6B.3.
"""

from __future__ import annotations

# Primary re-export (hft-contracts plane).
from hft_contracts.feature_sets.hashing import compute_feature_set_hash

# Back-compat: `_sanitize_for_hash` was previously re-exported from this
# module for older tests + consumers. Restore via a top-level import so
# `from hft_ops.feature_sets.hashing import _sanitize_for_hash` still works.
from hft_contracts.canonical_hash import sanitize_for_hash as _sanitize_for_hash

__all__ = [
    "compute_feature_set_hash",
    "_sanitize_for_hash",
]
