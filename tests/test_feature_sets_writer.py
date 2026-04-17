"""Tests for hft_ops.feature_sets.writer (Phase 4 Batch 4b)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hft_ops.feature_sets.schema import (
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetIntegrityError,
    FeatureSetProducedBy,
)
from hft_ops.feature_sets.writer import (
    AtomicWriteError,
    FeatureSetExists,
    atomic_write_json,
    write_feature_set,
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
        config_path="x/y.yaml",
        config_hash="a" * 64,
        source_profile_hash="b" * 64,
        data_export="data/exports/x",
        data_dir_hash="c" * 64,
    )


def _build(name: str = "test_v1", indices: list[int] = [0, 5, 12]) -> FeatureSet:
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
        description="Test",
        created_at="2026-04-15T12:00:00+00:00",
        created_by="test",
    )


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    def test_writes_object_as_json(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1, "b": [2, 3]})
        loaded = json.loads(path.read_text())
        assert loaded == {"a": 1, "b": [2, 3]}

    def test_sort_keys_enforced(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"b": 1, "a": 2, "c": 3})
        text = path.read_text()
        # Sort-keys means "a" appears first despite insertion order
        assert text.index('"a"') < text.index('"b"') < text.index('"c"')

    def test_trailing_newline(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"k": "v"})
        assert path.read_text().endswith("\n")

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "out.json"
        atomic_write_json(path, {"k": "v"})
        assert path.exists()

    def test_no_leftover_tmp_on_success(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"k": "v"})
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert tmp_files == []

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text('{"old": true}\n')
        atomic_write_json(path, {"new": True})
        assert json.loads(path.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# write_feature_set — refuse-overwrite + idempotent-on-match
# ---------------------------------------------------------------------------


class TestWriteFeatureSet:
    def test_first_write_succeeds(self, tmp_path):
        fs = _build()
        path = tmp_path / "test_v1.json"
        returned = write_feature_set(path, fs)
        assert returned == path
        assert path.exists()

    def test_idempotent_rewrite_with_same_content(self, tmp_path):
        fs = _build()
        path = tmp_path / "test_v1.json"
        write_feature_set(path, fs)
        mtime_1 = path.stat().st_mtime_ns

        # Rewrite with identical content — must NOT rewrite (preserves mtime).
        write_feature_set(path, fs)
        mtime_2 = path.stat().st_mtime_ns
        assert mtime_1 == mtime_2

    def test_different_content_without_force_raises(self, tmp_path):
        fs1 = _build(indices=[0, 5, 12])
        fs2 = _build(indices=[0, 5, 12, 42])  # different indices → different hash
        path = tmp_path / "test_v1.json"
        write_feature_set(path, fs1)

        with pytest.raises(FeatureSetExists, match="DIFFERENT content_hash"):
            write_feature_set(path, fs2)

    def test_different_content_with_force_overwrites(self, tmp_path):
        fs1 = _build(indices=[0, 5, 12])
        fs2 = _build(indices=[0, 5, 12, 42])
        path = tmp_path / "test_v1.json"
        write_feature_set(path, fs1)
        write_feature_set(path, fs2, force=True)

        loaded = json.loads(path.read_text())
        assert loaded["content_hash"] == fs2.content_hash

    def test_error_message_suggests_version_bump(self, tmp_path):
        fs1 = _build(name="momentum_v1", indices=[0, 5])
        fs2 = _build(name="momentum_v1", indices=[0, 5, 12])
        path = tmp_path / "momentum_v1.json"
        write_feature_set(path, fs1)

        with pytest.raises(FeatureSetExists) as excinfo:
            write_feature_set(path, fs2)
        assert "momentum_v2" in str(excinfo.value)

    def test_tampered_feature_set_refused_before_write(self, tmp_path):
        fs = _build()
        tampered = replace(fs, feature_indices=(0, 5, 12, 999))  # hash mismatch
        path = tmp_path / "tampered.json"
        with pytest.raises(FeatureSetIntegrityError):
            write_feature_set(path, tampered)
        # Must NOT create the file
        assert not path.exists()

    def test_corrupt_existing_file_refused_without_force(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{this is not valid json")

        fs = _build(name="corrupt")
        with pytest.raises(FeatureSetExists):
            write_feature_set(path, fs)

    def test_corrupt_existing_file_overwritten_with_force(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json")

        fs = _build(name="corrupt")
        write_feature_set(path, fs, force=True)
        # Now it's valid
        loaded = json.loads(path.read_text())
        assert loaded["content_hash"] == fs.content_hash
