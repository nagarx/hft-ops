"""Tests for hft_ops.feature_sets.schema (Phase 4 Batch 4b)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from hft_ops.feature_sets.schema import (
    FEATURE_SET_SCHEMA_VERSION,
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetRef,
    FeatureSetValidationError,
    validate_feature_set_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _applies_to() -> FeatureSetAppliesTo:
    return FeatureSetAppliesTo(assets=("NVDA",), horizons=(10, 60))


def _produced_by() -> FeatureSetProducedBy:
    return FeatureSetProducedBy(
        tool="hft-feature-evaluator",
        tool_version="0.3.0",
        config_path="hft-feature-evaluator/configs/test.yaml",
        config_hash="abc" * 21 + "d",  # 64-char hex
        source_profile_hash="def" * 21 + "0",
        data_export="data/exports/test_export",
        data_dir_hash="012" * 21 + "3",
    )


def _build(
    name: str = "test_v1",
    indices: list[int] = [0, 5, 12],
    source_feature_count: int = 98,
    contract_version: str = "2.2",
) -> FeatureSet:
    return FeatureSet.build(
        name=name,
        feature_indices=indices,
        feature_names=[f"feature_{i}" for i in indices],
        source_feature_count=source_feature_count,
        contract_version=contract_version,
        applies_to=_applies_to(),
        produced_by=_produced_by(),
        criteria={"name": "default", "min_passing_paths": 1},
        criteria_schema_version="1.0",
        description="Test FeatureSet",
        notes="",
        created_at="2026-04-15T12:00:00+00:00",
        created_by="test",
    )


# ---------------------------------------------------------------------------
# FeatureSet.build (auto-hash)
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_produces_correct_hash(self):
        fs = _build()
        # Verify hash independently via compute_feature_set_hash
        from hft_ops.feature_sets.hashing import compute_feature_set_hash
        expected = compute_feature_set_hash([0, 5, 12], 98, "2.2")
        assert fs.content_hash == expected

    def test_build_normalizes_indices_to_sorted_tuple(self):
        fs = _build(indices=[12, 0, 5, 5, 0])  # unsorted + duplicates
        assert fs.feature_indices == (0, 5, 12)

    def test_build_schema_version_is_current(self):
        fs = _build()
        assert fs.schema_version == FEATURE_SET_SCHEMA_VERSION

    def test_frozen_dataclass(self):
        fs = _build()
        with pytest.raises(FrozenInstanceError):
            fs.name = "other"  # type: ignore

    def test_build_requires_keyword_args(self):
        # build() uses keyword-only arguments to guard against
        # positional drift as the schema evolves.
        with pytest.raises(TypeError):
            FeatureSet.build(  # type: ignore[misc]
                "x", [0], ["f0"], 98, "2.2",
                _applies_to(), _produced_by(), {}, "1.0",
            )


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class TestVerifyIntegrity:
    def test_built_feature_set_passes_integrity(self):
        fs = _build()
        fs.verify_integrity()  # Should not raise

    def test_tampered_indices_fail_integrity(self):
        fs = _build()
        tampered = replace(fs, feature_indices=(0, 5, 12, 42))
        with pytest.raises(FeatureSetIntegrityError, match="integrity check failed"):
            tampered.verify_integrity()

    def test_tampered_source_feature_count_fails_integrity(self):
        fs = _build()
        tampered = replace(fs, source_feature_count=128)
        with pytest.raises(FeatureSetIntegrityError):
            tampered.verify_integrity()

    def test_tampered_contract_version_fails_integrity(self):
        fs = _build()
        tampered = replace(fs, contract_version="2.3")
        with pytest.raises(FeatureSetIntegrityError):
            tampered.verify_integrity()

    def test_metadata_edit_does_not_break_integrity(self):
        # PRODUCT-only hash: description/notes/created_at changes are
        # invisible to integrity verification. This is by design.
        fs = _build()
        edited = replace(
            fs,
            description="Completely different description",
            notes="Added notes after the fact",
            created_by="someone_else",
        )
        edited.verify_integrity()  # Should NOT raise

    def test_criteria_edit_does_not_break_integrity(self):
        # Recipe is not hashed.
        fs = _build()
        edited = replace(fs, criteria={"name": "different"})
        edited.verify_integrity()  # Should NOT raise

    def test_applies_to_edit_does_not_break_integrity(self):
        fs = _build()
        edited = replace(
            fs,
            applies_to=FeatureSetAppliesTo(assets=("MSFT",), horizons=(300,)),
        )
        edited.verify_integrity()  # Should NOT raise


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_dict_round_trip_preserves_all_fields(self):
        fs = _build()
        d = fs.to_dict()
        fs2 = FeatureSet.from_dict(d)
        assert fs == fs2

    def test_to_dict_produces_lists_not_tuples(self):
        fs = _build()
        d = fs.to_dict()
        assert isinstance(d["feature_indices"], list)
        assert isinstance(d["feature_names"], list)
        assert isinstance(d["applies_to"]["assets"], list)
        assert isinstance(d["applies_to"]["horizons"], list)

    def test_to_dict_contains_all_required_keys(self):
        fs = _build()
        d = fs.to_dict()
        required = {
            "schema_version", "name", "content_hash",
            "contract_version", "source_feature_count",
            "applies_to", "feature_indices", "feature_names",
            "produced_by", "criteria", "criteria_schema_version",
            "description", "notes", "created_at", "created_by",
        }
        assert required <= set(d.keys())

    def test_from_dict_with_tampered_hash_raises(self):
        fs = _build()
        d = fs.to_dict()
        # Tamper: flip a hash digit
        d["content_hash"] = "f" + d["content_hash"][1:]
        with pytest.raises(FeatureSetIntegrityError):
            FeatureSet.from_dict(d)

    def test_from_dict_with_verify_false_accepts_tampered(self):
        # Inspection mode — load even tampered files for diff tools.
        fs = _build()
        d = fs.to_dict()
        d["content_hash"] = "f" + d["content_hash"][1:]
        fs2 = FeatureSet.from_dict(d, verify=False)
        assert fs2.content_hash != fs.content_hash


# ---------------------------------------------------------------------------
# validate_feature_set_dict
# ---------------------------------------------------------------------------


class TestValidator:
    def test_valid_dict_passes(self):
        validate_feature_set_dict(_build().to_dict())

    def test_missing_required_key_raises(self):
        d = _build().to_dict()
        del d["name"]
        with pytest.raises(FeatureSetValidationError, match="missing required keys"):
            validate_feature_set_dict(d)

    def test_schema_version_mismatch_raises(self):
        d = _build().to_dict()
        d["schema_version"] = "99.0"
        with pytest.raises(FeatureSetValidationError, match="Unsupported"):
            validate_feature_set_dict(d)

    def test_bad_hash_format_raises(self):
        d = _build().to_dict()
        d["content_hash"] = "sha256:abc"
        with pytest.raises(FeatureSetValidationError, match="64-char"):
            validate_feature_set_dict(d)

    def test_uppercase_hash_rejected(self):
        d = _build().to_dict()
        d["content_hash"] = d["content_hash"].upper()
        with pytest.raises(FeatureSetValidationError, match="lowercase"):
            validate_feature_set_dict(d)

    def test_empty_indices_raises(self):
        d = _build().to_dict()
        d["feature_indices"] = []
        with pytest.raises(FeatureSetValidationError, match="non-empty"):
            validate_feature_set_dict(d)

    def test_negative_index_raises(self):
        d = _build().to_dict()
        d["feature_indices"] = [-1, 0]
        with pytest.raises(FeatureSetValidationError, match="non-negative"):
            validate_feature_set_dict(d)

    def test_duplicate_indices_raises(self):
        d = _build().to_dict()
        d["feature_indices"] = [0, 5, 5, 12]
        with pytest.raises(FeatureSetValidationError, match="unique"):
            validate_feature_set_dict(d)

    def test_index_out_of_range_raises(self):
        d = _build().to_dict()
        d["feature_indices"] = [0, 5, 99]  # but source_feature_count = 98
        with pytest.raises(FeatureSetValidationError, match="< source_feature_count"):
            validate_feature_set_dict(d)

    def test_bool_in_indices_rejected(self):
        # In Python, bool is a subclass of int. True would silently pass
        # a naive check. Validator must filter.
        d = _build().to_dict()
        d["feature_indices"] = [0, True, 5]
        with pytest.raises(FeatureSetValidationError):
            validate_feature_set_dict(d)

    def test_applies_to_missing_assets_raises(self):
        d = _build().to_dict()
        del d["applies_to"]["assets"]
        with pytest.raises(FeatureSetValidationError, match="assets"):
            validate_feature_set_dict(d)

    def test_applies_to_non_string_asset_raises(self):
        d = _build().to_dict()
        d["applies_to"]["assets"] = [42]
        with pytest.raises(FeatureSetValidationError, match="str"):
            validate_feature_set_dict(d)

    def test_produced_by_missing_key_raises(self):
        d = _build().to_dict()
        del d["produced_by"]["tool_version"]
        with pytest.raises(FeatureSetValidationError, match="produced_by missing"):
            validate_feature_set_dict(d)


# ---------------------------------------------------------------------------
# FeatureSetRef
# ---------------------------------------------------------------------------


class TestFeatureSetRef:
    def test_ref_fields_match_parent(self):
        fs = _build()
        r = fs.ref()
        assert r.name == fs.name
        assert r.content_hash == fs.content_hash

    def test_ref_is_frozen(self):
        r = FeatureSetRef(name="x", content_hash="a" * 64)
        with pytest.raises(FrozenInstanceError):
            r.name = "y"  # type: ignore
