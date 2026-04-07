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
        info = capture_git_info(tmp_path)
        assert info.commit_hash == ""


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
