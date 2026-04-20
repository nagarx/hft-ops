"""Tests for fingerprint computation and deduplication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_ops.ledger.dedup import (
    _extract_fingerprint_fields,
    check_duplicate,
    compute_fingerprint,
)
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


class TestExtractFingerprintFields:
    def test_strips_metadata(self):
        cfg = {
            "name": "should_be_stripped",
            "description": "also stripped",
            "tags": ["stripped"],
            "output_dir": "stripped",
            "data": {"feature_count": 98, "window_size": 100},
        }
        result = _extract_fingerprint_fields(cfg)
        assert "name" not in result
        assert "description" not in result
        assert "tags" not in result
        assert "output_dir" not in result
        assert result["data"]["feature_count"] == 98

    def test_preserves_numerical_config(self):
        cfg = {
            "model": {"hidden_size": 64, "dropout": 0.1},
            "train": {"learning_rate": 0.0001, "epochs": 50},
        }
        result = _extract_fingerprint_fields(cfg)
        assert result["model"]["hidden_size"] == 64
        assert result["train"]["epochs"] == 50


class TestComputeFingerprint:
    def test_deterministic(self, sample_manifest_yaml: Path, tmp_pipeline: Path):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)

        fp1 = compute_fingerprint(manifest, paths)
        fp2 = compute_fingerprint(manifest, paths)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_different_horizon_different_fingerprint(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        m1 = load_manifest(sample_manifest_yaml)
        m1.stages.training.horizon_value = 50

        m2 = load_manifest(sample_manifest_yaml)
        m2.stages.training.horizon_value = 200

        fp1 = compute_fingerprint(m1, paths)
        fp2 = compute_fingerprint(m2, paths)
        assert fp1 != fp2

    def test_different_overrides_different_fingerprint(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        m1 = load_manifest(sample_manifest_yaml)
        m1.stages.training.overrides["data.data_dir"] = "path_a"

        m2 = load_manifest(sample_manifest_yaml)
        m2.stages.training.overrides["data.data_dir"] = "path_b"

        fp1 = compute_fingerprint(m1, paths)
        fp2 = compute_fingerprint(m2, paths)
        assert fp1 != fp2


class TestCheckDuplicate:
    def test_no_ledger(self, tmp_path: Path):
        result = check_duplicate("abc123", tmp_path / "nonexistent")
        assert result is None

    def test_no_match(self, tmp_path: Path):
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        index = [{"experiment_id": "exp1", "fingerprint": "aaa"}]
        (ledger_dir / "index.json").write_text(json.dumps(index))

        result = check_duplicate("bbb", ledger_dir)
        assert result is None

    def test_match_found(self, tmp_path: Path):
        # Phase 8B MUST-FIX (2026-04-20): `check_duplicate` now routes through
        # `ExperimentLedger._load_index`, which treats `records/*.json` as
        # authoritative and re-projects the index envelope on legacy-bare-list
        # detection. Pre-Phase-8B test wrote ONLY `index.json` with no matching
        # record file; that scenario now correctly yields an empty envelope
        # (the legacy bare-list entries for records-that-don't-exist are
        # phantoms — dropping them is a feature, not a bug). Updated to
        # register via ExperimentLedger so both records/ and the envelope
        # reflect the same ground truth.
        from hft_ops.ledger.experiment_record import ExperimentRecord
        from hft_ops.ledger.ledger import ExperimentLedger
        from hft_ops.provenance.lineage import GitInfo, Provenance

        ledger_dir = tmp_path / "ledger"
        ledger = ExperimentLedger(ledger_dir)
        record = ExperimentRecord(
            experiment_id="exp1_20260420T000000_aaaaaaaa",
            name="exp1",
            fingerprint="aaa",
            contract_version="2.2",
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
            provenance=Provenance(
                git=GitInfo(commit_hash="x", branch="main", dirty=False),
                contract_version="2.2",
            ),
        )
        ledger.register(record)

        result = check_duplicate("aaa", ledger_dir)
        assert result is not None
        assert result["fingerprint"] == "aaa"
