"""
Content-addressed hashing for Phase 4 FeatureSet artifacts.

The hash is a fingerprint of the PRODUCT — the selected feature indices,
source feature width, and pipeline contract version. It is NOT a hash of
the recipe that produced the selection (criteria, tool_version, assets,
horizons). Two evaluator runs with different criteria that arrive at
identical indices produce identical content hashes; recipe fields are
stored as provenance metadata in the FeatureSet JSON but EXCLUDED from
the content hash by design.

Design rationale (Phase 4 R1, validated 2026-04-15 via adversarial
design audit): recipe-inclusive hashing would explode the registry with
N-criteria × 1-selection entries during criteria iteration. Product-only
hashing matches git-blob semantics — address by content, not by command.

Canonical form matches the monorepo convention (locked by tests):
    json.dumps(obj, sort_keys=True, default=str) → SHA-256 hex

See:
- hft-feature-evaluator/src/hft_evaluator/pipeline.py::compute_profile_hash
- hft-ops/src/hft_ops/ledger/dedup.py::compute_fingerprint
- hft-ops/src/hft_ops/provenance/lineage.py::hash_config_dict

Portability caveat: stable across CPython versions + platforms, but NOT
byte-portable to other languages' default JSON serializers (Rust
``serde_json`` emits ``","`` where Python default emits ``", "``).
Consumers needing polyglot reproducibility must mirror Python's
whitespace convention.
"""

from __future__ import annotations

from typing import Iterable

# Phase 4 Batch 4c hardening (2026-04-15): canonical-form primitives moved
# to ``hft_contracts.canonical_hash``. The ``_sanitize_for_hash`` name is
# preserved as a backward-compat re-export for tests that import it
# directly from this module. Canonical form byte-parity with the
# pre-extraction implementation is locked by
# ``hft-contracts/tests/test_canonical_hash.py``.
from hft_contracts.canonical_hash import (
    canonical_json_blob,
    sanitize_for_hash as _sanitize_for_hash,
    sha256_hex,
)


def compute_feature_set_hash(
    feature_indices: Iterable[int],
    source_feature_count: int,
    contract_version: str,
) -> str:
    """Compute a deterministic content hash over FeatureSet core fields.

    Hashes ONLY the PRODUCT: sorted-unique feature indices + source
    feature width + pipeline contract version. Recipe fields (criteria,
    tool_version, applies_to, description, created_at, created_by) are
    EXCLUDED — see module docstring for the full rationale.

    Canonical form (locked by tests):

    .. code-block:: python

        canonical = {
            "feature_indices": sorted(set(int(i) for i in feature_indices)),
            "source_feature_count": int(source_feature_count),
            "contract_version": str(contract_version),
        }
        blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
        hash = hashlib.sha256(blob).hexdigest()

    Args:
        feature_indices: Sequence of feature indices. Order and duplicates
            are normalized by ``sorted(set(...))`` before hashing, so
            callers don't need to pre-sort.
        source_feature_count: Width of the source feature axis (e.g.,
            98, 128). Prevents index-list collisions across different
            source widths: the list ``[0, 5, 10]`` derived from a
            98-feature export MUST hash differently from the same list
            derived from a 128-feature export.
        contract_version: Pipeline schema version string (e.g., ``"2.2"``).
            Feature index semantics are contract-version-bound; bumping
            the contract version means the same index may refer to a
            different feature.

    Returns:
        64-char lowercase hex SHA-256 digest. NO ``sha256:`` prefix
        (matches ``ExperimentRecord.fingerprint`` + ``hash_config_dict``
        convention; the prefix is reserved for external identifiers
        like databento manifests).

    Raises:
        ValueError: If ``feature_indices`` is empty, or if any index is
            negative, or if ``source_feature_count`` is non-positive.
            (Hash computation itself is pure; these guards exist to catch
            obviously-invalid inputs at the hashing boundary rather than
            let them propagate to consumers.)
    """
    indices = sorted(set(int(i) for i in feature_indices))
    if not indices:
        raise ValueError(
            "feature_indices must be non-empty for hashing"
        )
    if indices[0] < 0:
        raise ValueError(
            f"feature_indices must be non-negative, got min={indices[0]}"
        )
    if int(source_feature_count) <= 0:
        raise ValueError(
            f"source_feature_count must be positive, got {source_feature_count}"
        )

    canonical = {
        "feature_indices": indices,
        "source_feature_count": int(source_feature_count),
        "contract_version": str(contract_version),
    }
    return sha256_hex(canonical_json_blob(canonical))
