"""Tests for fingerprint computation and deduplication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

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

    def test_importance_field_excluded_from_fingerprint(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        """Phase 8C-α post-audit round-2 (architect-Q7 trainer wire-in):
        lock the fingerprint-vs-importance invariant. ``importance`` on
        ``ExperimentConfig`` is an OBSERVATION (post-training permutation
        importance audit), NOT a treatment.

        Round-3 post-audit Agent-3 H2 fix: previous version called
        ``compute_fingerprint_explain`` on a fixture manifest that had
        NO ``importance`` field — the test passed vacuously (deleting
        the blacklist entry would NOT have failed it). Corrected test
        calls ``_extract_fingerprint_fields`` DIRECTLY on a dict that
        explicitly CONTAINS ``importance`` at multiple nest levels.
        Now: deleting the blacklist entry DOES fail the test.
        """
        from hft_ops.ledger.dedup import _extract_fingerprint_fields

        # Inject importance at multiple nest levels to exercise the
        # recursive strip logic. If the blacklist entry for "importance"
        # is removed, any of these will leak into the result.
        config_with_importance = {
            "name": "test_experiment",  # stripped (metadata)
            "data": {"batch_size": 128},  # NOT stripped (treatment)
            "importance": {  # MUST be stripped (observation)
                "enabled": True,
                "n_permutations": 500,
                "seed": 42,
            },
            "model": {
                "type": "tlob",
                "importance": {"nested_variant": True},  # deeply-nested
            },
            "train": {
                "epochs": 30,
                "importance": "also_stripped",  # mid-level
            },
        }

        stripped = _extract_fingerprint_fields(config_with_importance)

        _assert_no_field_in_canonical_input(
            stripped,
            field_name="importance",
            rationale=(
                "importance.{enabled,n_permutations,n_seeds,seed,...} are "
                "observations (post-training permutation audit), not "
                "treatments. Fingerprint-including breaks dedup (same "
                "trained model → different fingerprints depending on "
                "observation flags). Round-3 non-vacuous test: deleting "
                "'importance' from exclude_keys MUST fail this test."
            ),
        )
        # Positive checks: non-importance fields still present
        assert "data" in stripped, "Non-importance treatment keys must survive strip"
        assert "epochs" in stripped.get("train", {}), (
            "Sibling keys under stripped-parent paths must survive"
        )

    def test_artifacts_field_excluded_from_fingerprint(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        """Phase 8C-α post-audit (arch-S1, 2026-04-20): lock the
        fingerprint-vs-artifacts invariant. ``ExperimentRecord.artifacts``
        is an OBSERVATION (post-training importance artifacts) — NOT a
        TREATMENT.

        Note: the current `_extract_fingerprint_fields` blacklist does
        NOT contain 'artifacts' — that field lives on `ExperimentRecord`
        (output), NOT on `ExperimentConfig` / manifest (input to
        `compute_fingerprint`). Structurally, `artifacts` cannot enter
        the fingerprint because the fingerprint reads only the
        manifest config tree, never the record. This test locks that
        structural invariant by asserting a direct call to
        `_extract_fingerprint_fields` with an `artifacts`-containing
        dict does NOT leak it — even though it would NEVER be called
        with such a dict in production.

        Round-3 post-audit Agent-3 H2 fix: previous version was
        vacuous (fixture had no `artifacts` key → strip never
        exercised). Corrected to directly exercise the strip path.
        """
        from hft_ops.ledger.dedup import _extract_fingerprint_fields

        # If `artifacts` were EVER accidentally added to the manifest
        # config tree (e.g., via a future refactor that serializes
        # ExperimentRecord into the fingerprint input), this test
        # locks the blacklist to strip it.
        config_with_artifacts = {
            "data": {"batch_size": 128},
            "artifacts": [  # hypothetical future leak
                {"kind": "feature_importance", "sha256": "abc123"},
            ],
            "model": {
                "type": "tlob",
                "artifacts": [{"deeply": "nested"}],
            },
        }

        stripped = _extract_fingerprint_fields(config_with_artifacts)

        _assert_no_field_in_canonical_input(
            stripped,
            field_name="artifacts",
            rationale=(
                "artifacts[] is an observation (post-training importance), "
                "not a treatment. Including it in the fingerprint breaks "
                "dedup (Phase-3-§3.3b-class ledger conflation). Round-3 "
                "non-vacuous test: adding `artifacts` to the fingerprint "
                "manifest tree MUST be stripped."
            ),
        )
        # Positive check: non-artifacts treatment keys still present
        assert "data" in stripped, "Non-artifacts treatment keys must survive strip"

    def test_fingerprint_invariants_via_manifest_with_importance(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        """Round-3 integration-lens lock: with a MANIFEST that carries
        importance via trainer_config inline overrides, the resulting
        fingerprint components must still not contain 'importance'.
        Orthogonal test to the direct _extract_fingerprint_fields call —
        covers the manifest → compute_fingerprint_explain code path.
        """
        from hft_ops.ledger.dedup import compute_fingerprint_explain

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)
        # Inject importance into training overrides (realistic path:
        # trainer_config carries importance block in unified manifests)
        if not manifest.stages.training.overrides:
            manifest.stages.training.overrides = {}
        manifest.stages.training.overrides["importance.enabled"] = True
        manifest.stages.training.overrides["importance.n_permutations"] = 500

        _, components = compute_fingerprint_explain(manifest, paths)

        _assert_no_field_in_canonical_input(
            components,
            field_name="importance",
            rationale="Manifest-level importance overrides must still be stripped.",
        )


def _assert_no_field_in_canonical_input(
    canonical_input: Any,
    *,
    field_name: str,
    rationale: str,
) -> None:
    """Post-audit round-2 architect-Q6: API-insulated helper for
    fingerprint-invariant tests.

    Walks ``canonical_input`` (a dict / list / scalar tree) and asserts
    that no key named ``field_name`` appears at any depth. Decoupled
    from the specific ``compute_fingerprint_explain`` return shape so
    it survives future refactors — Phase 9 SQLite migration may
    restructure the fingerprint-input canonical form without changing
    the INVARIANT (certain fields must NEVER enter the canonical input).

    Args:
        canonical_input: The canonical fingerprint input (typically a
            dict returned by compute_fingerprint_explain). May contain
            nested dicts + lists.
        field_name: The key that must NEVER appear at any depth.
        rationale: Why this invariant exists (included in assertion
            message for debugging).
    """
    def _walk(obj: Any, path: str = "root") -> List[str]:
        violations: List[str] = []
        if isinstance(obj, dict):
            if field_name in obj:
                violations.append(f"{path}.{field_name}")
            for k, v in obj.items():
                violations.extend(_walk(v, f"{path}.{k}"))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                violations.extend(_walk(v, f"{path}[{i}]"))
        return violations

    violations = _walk(canonical_input)
    assert not violations, (
        f"Fingerprint canonical input leaked '{field_name}' at: "
        f"{violations}. {rationale}"
    )


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
