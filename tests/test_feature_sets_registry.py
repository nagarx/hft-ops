"""Tests for hft_ops.feature_sets.registry (Phase 4 Batch 4b)."""

from __future__ import annotations

import json

import pytest

from hft_ops.feature_sets.registry import (
    FeatureSetNotFound,
    FeatureSetRegistry,
)
from hft_ops.feature_sets.schema import (
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
    FeatureSetValidationError,
)
from hft_ops.feature_sets.writer import write_feature_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _applies_to() -> FeatureSetAppliesTo:
    return FeatureSetAppliesTo(assets=("NVDA",), horizons=(10,))


def _produced_by() -> FeatureSetProducedBy:
    return FeatureSetProducedBy(
        tool="hft-feature-evaluator",
        tool_version="0.3.0",
        config_path="x.yaml",
        config_hash="a" * 64,
        source_profile_hash="b" * 64,
        data_export="data/exports/x",
        data_dir_hash="c" * 64,
    )


def _build(name: str, indices: list[int]) -> FeatureSet:
    return FeatureSet.build(
        name=name,
        feature_indices=indices,
        feature_names=[f"feature_{i}" for i in indices],
        source_feature_count=98,
        contract_version="2.2",
        applies_to=_applies_to(),
        produced_by=_produced_by(),
        criteria={"name": "default"},
        criteria_schema_version="1.0",
        description="",
        created_at="2026-04-15T12:00:00+00:00",
        created_by="test",
    )


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


class TestRegistryConstruction:
    def test_missing_root_raises_by_default(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FeatureSetRegistry(tmp_path / "nonexistent")

    def test_missing_root_allowed_with_flag(self, tmp_path):
        # Don't raise — downstream writers may create it.
        reg = FeatureSetRegistry(tmp_path / "nonexistent", allow_missing=True)
        assert reg.list_refs() == []
        assert reg.names() == []


# ---------------------------------------------------------------------------
# list_refs / names / exists
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_empty_registry(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path)
        assert reg.list_refs() == []
        assert reg.names() == []

    def test_single_entry(self, tmp_path):
        fs = _build("alpha_v1", [0, 5, 12])
        write_feature_set(tmp_path / "alpha_v1.json", fs)

        reg = FeatureSetRegistry(tmp_path)
        refs = reg.list_refs()
        assert len(refs) == 1
        assert refs[0].name == "alpha_v1"
        assert refs[0].content_hash == fs.content_hash

    def test_multiple_entries_sorted_by_name(self, tmp_path):
        write_feature_set(tmp_path / "gamma_v1.json", _build("gamma_v1", [0]))
        write_feature_set(tmp_path / "alpha_v1.json", _build("alpha_v1", [1]))
        write_feature_set(tmp_path / "beta_v1.json", _build("beta_v1", [2]))

        reg = FeatureSetRegistry(tmp_path)
        names = reg.names()
        assert names == ["alpha_v1", "beta_v1", "gamma_v1"]

    def test_exists_true_for_written_entry(self, tmp_path):
        write_feature_set(tmp_path / "x_v1.json", _build("x_v1", [0]))
        reg = FeatureSetRegistry(tmp_path)
        assert reg.exists("x_v1")

    def test_exists_false_for_absent_entry(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path)
        assert not reg.exists("missing")

    def test_malformed_file_skipped_in_listing(self, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json")
        reg = FeatureSetRegistry(tmp_path)
        # Listing succeeds (quietly skips bad file)
        assert reg.list_refs() == []

    def test_non_json_files_ignored(self, tmp_path):
        (tmp_path / "README.md").write_text("# Not a FeatureSet")
        (tmp_path / "noise.yaml").write_text("stuff")
        reg = FeatureSetRegistry(tmp_path)
        assert reg.names() == []


# ---------------------------------------------------------------------------
# get — full load + verify
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_returns_full_feature_set(self, tmp_path):
        fs = _build("test_v1", [0, 5, 12])
        write_feature_set(tmp_path / "test_v1.json", fs)

        reg = FeatureSetRegistry(tmp_path)
        loaded = reg.get("test_v1")
        assert loaded == fs

    def test_get_missing_raises_featureset_not_found(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path)
        with pytest.raises(FeatureSetNotFound, match="not found"):
            reg.get("missing")

    def test_get_malformed_json_raises_validation_error(self, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json")
        reg = FeatureSetRegistry(tmp_path)
        with pytest.raises(FeatureSetValidationError, match="not valid JSON"):
            reg.get("bad")

    def test_get_tampered_file_raises_integrity_error_by_default(self, tmp_path):
        fs = _build("test_v1", [0, 5, 12])
        path = tmp_path / "test_v1.json"
        write_feature_set(path, fs)

        # Manually tamper: flip a feature_index without updating hash
        data = json.loads(path.read_text())
        data["feature_indices"] = [0, 5, 12, 42]  # added without re-hashing
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

        reg = FeatureSetRegistry(tmp_path)
        with pytest.raises(FeatureSetIntegrityError, match="integrity check failed"):
            reg.get("test_v1")

    def test_get_tampered_file_loads_with_verify_false(self, tmp_path):
        fs = _build("test_v1", [0, 5, 12])
        path = tmp_path / "test_v1.json"
        write_feature_set(path, fs)

        data = json.loads(path.read_text())
        data["feature_indices"] = [0, 5, 12, 42]
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

        reg = FeatureSetRegistry(tmp_path)
        loaded = reg.get("test_v1", verify=False)
        # Loaded despite hash mismatch
        assert loaded.feature_indices == (0, 5, 12, 42)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_path_separator_in_name_rejected(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path, allow_missing=True)
        with pytest.raises(ValueError, match="path separators"):
            reg.path_for("foo/bar")

    def test_backslash_in_name_rejected(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path, allow_missing=True)
        with pytest.raises(ValueError, match="path separators"):
            reg.path_for("foo\\bar")

    def test_dotfile_name_rejected(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path, allow_missing=True)
        with pytest.raises(ValueError, match="start with"):
            reg.path_for(".hidden")

    def test_ordinary_name_accepted(self, tmp_path):
        reg = FeatureSetRegistry(tmp_path, allow_missing=True)
        p = reg.path_for("momentum_hft_v1")
        assert p == tmp_path / "momentum_hft_v1.json"
