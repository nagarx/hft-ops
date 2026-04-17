"""Tests for provenance capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_ops.provenance.lineage import (
    GitInfo,
    Provenance,
    build_provenance,
    capture_git_info,
    hash_config_dict,
    hash_directory_manifest,
    hash_file,
)


class TestHashFile:
    def test_existing_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = hash_file(f)
        assert len(h) == 64  # SHA-256 hex
        assert h == hash_file(f)  # deterministic

    def test_different_content(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "b.txt"
        f2.write_text("world")
        assert hash_file(f1) != hash_file(f2)

    def test_missing_file(self, tmp_path: Path):
        assert hash_file(tmp_path / "nonexistent.txt") == ""


class TestHashConfigDict:
    def test_deterministic(self):
        cfg = {"a": 1, "b": {"c": 3}}
        h1 = hash_config_dict(cfg)
        h2 = hash_config_dict(cfg)
        assert h1 == h2
        assert len(h1) == 64

    def test_order_independent(self):
        h1 = hash_config_dict({"a": 1, "b": 2})
        h2 = hash_config_dict({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_values(self):
        h1 = hash_config_dict({"a": 1})
        h2 = hash_config_dict({"a": 2})
        assert h1 != h2


class TestHashDirectoryManifest:
    def test_empty_dir(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        h = hash_directory_manifest(d)
        assert len(h) == 64

    def test_with_files(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.npy").write_bytes(b"x" * 100)
        (d / "b.npy").write_bytes(b"y" * 200)

        h = hash_directory_manifest(d)
        assert len(h) == 64

    def test_deterministic(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "file.txt").write_text("content")

        h1 = hash_directory_manifest(d)
        h2 = hash_directory_manifest(d)
        assert h1 == h2

    def test_detects_file_addition(self, tmp_path: Path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("content")
        h1 = hash_directory_manifest(d)

        (d / "b.txt").write_text("new file")
        h2 = hash_directory_manifest(d)
        assert h1 != h2

    def test_missing_dir(self, tmp_path: Path):
        assert hash_directory_manifest(tmp_path / "nonexistent") == ""


class TestGitInfo:
    def test_roundtrip(self):
        info = GitInfo(commit_hash="abc123", branch="main", dirty=True, short_hash="abc1")
        d = info.to_dict()
        restored = GitInfo.from_dict(d)
        assert restored.commit_hash == "abc123"
        assert restored.branch == "main"
        assert restored.dirty is True

    def test_empty_defaults(self):
        info = GitInfo()
        assert info.commit_hash == ""
        assert info.dirty is False


class TestCaptureGitInfo:
    def test_non_repo_dir(self, tmp_path: Path):
        """Non-repo dir returns the NOT_GIT_TRACKED_SENTINEL.

        Rationale: an explicit sentinel is distinguishable from a git-unavailable
        environment (empty string) and from a malformed record. Retroactive
        records and monorepo-root-without-git setups use this sentinel so
        downstream tools can filter / detect untracked records.
        """
        from hft_ops.provenance.lineage import NOT_GIT_TRACKED_SENTINEL
        info = capture_git_info(tmp_path)
        assert info.commit_hash == NOT_GIT_TRACKED_SENTINEL
        assert info.short_hash == NOT_GIT_TRACKED_SENTINEL[:8]


class TestProvenanceRetroactiveAndSchema:
    """Phase 0.3: Provenance gains `retroactive` and `schema_version` fields."""

    def test_default_not_retroactive(self):
        prov = Provenance()
        assert prov.retroactive is False

    def test_default_schema_version(self):
        from hft_ops.provenance.lineage import PROVENANCE_SCHEMA_VERSION
        prov = Provenance()
        assert prov.schema_version == PROVENANCE_SCHEMA_VERSION
        assert prov.schema_version == "1.0"

    def test_retroactive_true_roundtrip(self):
        prov = Provenance(
            git=GitInfo(commit_hash="not_git_tracked", short_hash="not_git_"),
            contract_version="2.2",
            retroactive=True,
        )
        d = prov.to_dict()
        assert d["retroactive"] is True
        assert d["schema_version"] == "1.0"
        restored = Provenance.from_dict(d)
        assert restored.retroactive is True
        assert restored.schema_version == "1.0"

    def test_old_records_default_to_schema_1_0(self):
        """Old records without schema_version default to '1.0' for backward compat."""
        old_data = {
            "git": {"commit_hash": "abc", "branch": "main"},
            "config_hashes": {},
            "contract_version": "2.2",
            "timestamp_utc": "2026-03-01T00:00:00+00:00",
            # no retroactive, no schema_version
        }
        restored = Provenance.from_dict(old_data)
        assert restored.schema_version == "1.0"
        assert restored.retroactive is False


class TestProvenance:
    def test_roundtrip(self):
        prov = Provenance(
            git=GitInfo(commit_hash="abc", branch="main", dirty=False, short_hash="ab"),
            config_hashes={"manifest": "hash1", "extractor": "hash2"},
            data_dir_hash="datahash",
            contract_version="2.2",
            timestamp_utc="2026-03-05T12:00:00+00:00",
        )
        d = prov.to_dict()
        restored = Provenance.from_dict(d)
        assert restored.git.commit_hash == "abc"
        assert restored.config_hashes["manifest"] == "hash1"
        assert restored.contract_version == "2.2"

    def test_build_provenance(self, tmp_path: Path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("experiment:\n  name: test\n")

        prov = build_provenance(
            tmp_path,
            manifest_path=manifest,
            contract_version="2.2",
        )
        assert prov.config_hashes["manifest"] != ""
        assert prov.contract_version == "2.2"
        assert prov.timestamp_utc != ""

    def test_build_provenance_inline_trainer_config_dict(self, tmp_path: Path):
        """Phase 6 6A.3 regression guard — inline `trainer_config:` (Phase 1
        wrapper-less) must produce `config_hashes["trainer"]` via canonical
        hash of the dict. Prior code silently left this None for every
        unified manifest.
        """
        trainer_cfg = {
            "name": "E5_60s_huber_cvml",
            "model": {"model_type": "tlob", "input_size": 98},
            "train": {"batch_size": 128, "seed": 42},
        }
        prov = build_provenance(
            tmp_path,
            trainer_config_dict=trainer_cfg,
            contract_version="2.2",
        )
        assert "trainer" in prov.config_hashes, (
            "Inline trainer_config must populate config_hashes['trainer']"
        )
        assert len(prov.config_hashes["trainer"]) == 64, (
            "SHA-256 hex digest is 64 chars"
        )
        # Deterministic hash across call sites (SSoT via hft_contracts.canonical_hash).
        prov2 = build_provenance(
            tmp_path,
            trainer_config_dict=dict(trainer_cfg),  # fresh copy
            contract_version="2.2",
        )
        assert prov.config_hashes["trainer"] == prov2.config_hashes["trainer"], (
            "Deterministic canonical hash: same dict → same digest"
        )

    def test_build_provenance_trainer_config_path_and_dict_mutually_exclusive(
        self, tmp_path: Path
    ):
        """Phase 6 6A.3 — `trainer_config_path` and `trainer_config_dict` are
        mutually exclusive; caller must not supply both."""
        trainer_yaml = tmp_path / "trainer.yaml"
        trainer_yaml.write_text("model:\n  model_type: tlob\n")
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_provenance(
                tmp_path,
                trainer_config_path=trainer_yaml,
                trainer_config_dict={"model": {"model_type": "tlob"}},
                contract_version="2.2",
            )
