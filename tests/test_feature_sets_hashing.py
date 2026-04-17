"""Tests for hft_ops.feature_sets.hashing (Phase 4 Batch 4b)."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from hft_ops.feature_sets.hashing import (
    _sanitize_for_hash,
    compute_feature_set_hash,
)


# ---------------------------------------------------------------------------
# Determinism (product-only hash must be stable across runs + orderings)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_hash_across_calls(self):
        hashes = {
            compute_feature_set_hash([0, 5, 12], 98, "2.2")
            for _ in range(100)
        }
        assert len(hashes) == 1

    def test_input_order_does_not_matter(self):
        # sorted(set(...)) normalizes input
        h1 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h2 = compute_feature_set_hash([12, 0, 5], 98, "2.2")
        h3 = compute_feature_set_hash([5, 12, 0], 98, "2.2")
        assert h1 == h2 == h3

    def test_duplicates_are_normalized_away(self):
        # Duplicates in input produce the same hash as deduped input
        h1 = compute_feature_set_hash([0, 5, 5, 12], 98, "2.2")
        h2 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert h1 == h2


# ---------------------------------------------------------------------------
# Sensitivity (any core-field change MUST change hash)
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_adding_index_changes_hash(self):
        h1 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h2 = compute_feature_set_hash([0, 5, 12, 42], 98, "2.2")
        assert h1 != h2

    def test_removing_index_changes_hash(self):
        h1 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h2 = compute_feature_set_hash([0, 12], 98, "2.2")
        assert h1 != h2

    def test_source_feature_count_change_changes_hash(self):
        # Same indices, different source width → distinct product
        h98 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h128 = compute_feature_set_hash([0, 5, 12], 128, "2.2")
        assert h98 != h128

    def test_contract_version_change_changes_hash(self):
        h22 = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        h23 = compute_feature_set_hash([0, 5, 12], 98, "2.3")
        assert h22 != h23


# ---------------------------------------------------------------------------
# Hash format (locked by tests — consumers depend on the shape)
# ---------------------------------------------------------------------------


class TestHashFormat:
    def test_64_lowercase_hex(self):
        h = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
        assert h == h.lower()

    def test_no_sha256_prefix(self):
        # Matches ExperimentRecord.fingerprint + hash_config_dict convention.
        h = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert not h.startswith("sha256:")
        assert not h.startswith("SHA256:")


# ---------------------------------------------------------------------------
# Validation (bad inputs must fail loudly at the hashing boundary)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_feature_set_hash([], 98, "2.2")

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_feature_set_hash([-1, 5], 98, "2.2")

    def test_zero_source_feature_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_feature_set_hash([0, 5], 0, "2.2")

    def test_negative_source_feature_count_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_feature_set_hash([0, 5], -1, "2.2")


# ---------------------------------------------------------------------------
# Canonical form alignment (locks monorepo convention)
# ---------------------------------------------------------------------------


class TestCanonicalFormAlignment:
    """Lock the canonical form (sort_keys=True + default=str + raw hex
    SHA-256) so a future refactor cannot silently drift from the
    hft-ops dedup.py / lineage.py / evaluator compute_profile_hash
    convention without this test failing."""

    def test_hash_matches_explicit_canonical_form(self):
        indices = [0, 5, 12]
        sfc = 98
        cv = "2.2"
        expected_canonical = {
            "feature_indices": sorted(set(indices)),
            "source_feature_count": sfc,
            "contract_version": cv,
        }
        expected_blob = json.dumps(
            expected_canonical, sort_keys=True, default=str
        ).encode("utf-8")
        expected_hash = hashlib.sha256(expected_blob).hexdigest()
        assert compute_feature_set_hash(indices, sfc, cv) == expected_hash

    def test_behavioral_parity_with_evaluator_sanitize_for_hash(self):
        # Lock that hft-ops's _sanitize_for_hash matches the evaluator's
        # on the inputs that matter (dict/list/tuple + NaN/Inf). If
        # the two implementations ever drift, FeatureSet hashes will
        # drift silently.
        assert _sanitize_for_hash(float("nan")) is None
        assert _sanitize_for_hash(float("inf")) is None
        assert _sanitize_for_hash(float("-inf")) is None
        assert _sanitize_for_hash(1.5) == 1.5
        assert _sanitize_for_hash(0) == 0
        assert _sanitize_for_hash("hello") == "hello"
        assert _sanitize_for_hash(True) is True
        assert _sanitize_for_hash(None) is None
        assert _sanitize_for_hash([1.0, float("nan"), 2.0]) == [1.0, None, 2.0]
        assert _sanitize_for_hash((3.0, float("inf"))) == [3.0, None]
        assert _sanitize_for_hash({"x": float("nan")}) == {"x": None}
        assert _sanitize_for_hash({"nested": [float("-inf")]}) == {
            "nested": [None]
        }
