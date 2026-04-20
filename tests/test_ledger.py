"""Tests for experiment ledger CRUD operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_ops.ledger.experiment_record import ExperimentRecord
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.provenance.lineage import Provenance, GitInfo


def _make_record(
    experiment_id: str = "test_exp_20260305T120000_abcd1234",
    name: str = "test_exp",
    **kwargs,
) -> ExperimentRecord:
    """Create a minimal ExperimentRecord for testing."""
    return ExperimentRecord(
        experiment_id=experiment_id,
        name=name,
        fingerprint=kwargs.get("fingerprint", "fp_" + experiment_id),
        status=kwargs.get("status", "completed"),
        tags=kwargs.get("tags", ["test"]),
        training_metrics=kwargs.get(
            "training_metrics",
            {"accuracy": 0.55, "macro_f1": 0.42},
        ),
        training_config=kwargs.get(
            "training_config",
            {"model": {"model_type": "tlob"}, "data": {"labeling_strategy": "tlob"}},
        ),
    )


class TestExperimentRecord:
    def test_roundtrip_json(self, tmp_path: Path):
        record = _make_record()
        path = tmp_path / "record.json"
        record.save(path)

        loaded = ExperimentRecord.load(path)
        assert loaded.experiment_id == record.experiment_id
        assert loaded.name == record.name
        assert loaded.fingerprint == record.fingerprint
        assert loaded.training_metrics == record.training_metrics

    def test_index_entry(self):
        record = _make_record()
        entry = record.index_entry()

        assert entry["experiment_id"] == record.experiment_id
        assert entry["name"] == "test_exp"
        assert entry["fingerprint"] == record.fingerprint
        assert entry["model_type"] == "tlob"
        assert entry["labeling_strategy"] == "tlob"
        assert "accuracy" in entry["training_metrics"]

    def test_provenance_roundtrip(self, tmp_path: Path):
        record = _make_record()
        record.provenance = Provenance(
            git=GitInfo(commit_hash="abc123", branch="main", dirty=False),
            contract_version="2.2",
        )
        path = tmp_path / "record.json"
        record.save(path)

        loaded = ExperimentRecord.load(path)
        assert loaded.provenance.git.commit_hash == "abc123"
        assert loaded.provenance.contract_version == "2.2"

    def test_created_at_auto_set(self):
        record = ExperimentRecord(experiment_id="test")
        assert record.created_at != ""


class TestExperimentLedger:
    def test_register_and_get(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        record = _make_record()

        exp_id = ledger.register(record)
        assert exp_id == record.experiment_id

        loaded = ledger.get(exp_id)
        assert loaded is not None
        assert loaded.name == "test_exp"

    def test_duplicate_register_raises(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        record = _make_record()

        ledger.register(record)
        with pytest.raises(ValueError, match="already registered"):
            ledger.register(record)

    def test_list_all(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")

        for i in range(3):
            record = _make_record(experiment_id=f"exp_{i}")
            ledger.register(record)

        entries = ledger.list_all()
        assert len(entries) == 3

    def test_count(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        assert ledger.count() == 0

        ledger.register(_make_record())
        assert ledger.count() == 1

    def test_filter_by_tags(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        ledger.register(_make_record(experiment_id="a", tags=["nvda", "tlob"]))
        ledger.register(_make_record(experiment_id="b", tags=["nvda", "hmhp"]))
        ledger.register(_make_record(experiment_id="c", tags=["aapl"]))

        results = ledger.filter(tags=["nvda"])
        assert len(results) == 2

        results = ledger.filter(tags=["nvda", "tlob"])
        assert len(results) == 1

    def test_filter_by_model_type(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        ledger.register(
            _make_record(
                experiment_id="a",
                training_config={"model": {"model_type": "tlob"}, "data": {}},
            )
        )
        ledger.register(
            _make_record(
                experiment_id="b",
                training_config={"model": {"model_type": "hmhp"}, "data": {}},
            )
        )

        results = ledger.filter(model_type="tlob")
        assert len(results) == 1
        assert results[0]["model_type"] == "tlob"

    def test_filter_by_min_f1(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        ledger.register(
            _make_record(
                experiment_id="low",
                training_metrics={"macro_f1": 0.30, "accuracy": 0.40},
            )
        )
        ledger.register(
            _make_record(
                experiment_id="high",
                training_metrics={"macro_f1": 0.50, "accuracy": 0.60},
            )
        )

        results = ledger.filter(min_f1=0.40)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "high"

    def test_find_by_fingerprint(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        record = _make_record(fingerprint="unique_fp_123")
        ledger.register(record)

        result = ledger.find_by_fingerprint("unique_fp_123")
        assert result is not None
        assert result["experiment_id"] == record.experiment_id

        result = ledger.find_by_fingerprint("nonexistent")
        assert result is None

    def test_update_notes(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        record = _make_record()
        ledger.register(record)

        assert ledger.update_notes(record.experiment_id, "good results!")
        loaded = ledger.get(record.experiment_id)
        assert loaded is not None
        assert loaded.notes == "good results!"

    def test_update_notes_not_found(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        assert not ledger.update_notes("nonexistent", "notes")

    def test_get_not_found(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        assert ledger.get("nonexistent") is None

    def test_summary(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        ledger.register(_make_record(experiment_id="a", status="completed"))
        ledger.register(_make_record(experiment_id="b", status="failed"))

        summary = ledger.summary()
        assert summary["total_experiments"] == 2
        assert summary["by_status"]["completed"] == 1
        assert summary["by_status"]["failed"] == 1

    def test_persistence_across_instances(self, tmp_path: Path):
        ledger_dir = tmp_path / "ledger"

        ledger1 = ExperimentLedger(ledger_dir)
        ledger1.register(_make_record(experiment_id="persistent"))

        ledger2 = ExperimentLedger(ledger_dir)
        assert ledger2.count() == 1
        assert ledger2.get("persistent") is not None

    def test_empty_id_raises(self, tmp_path: Path):
        ledger = ExperimentLedger(tmp_path / "ledger")
        record = _make_record(experiment_id="")
        with pytest.raises(ValueError, match="experiment_id"):
            ledger.register(record)


class TestRecordType:
    """Phase 1.3: RecordType enum + record_type / sub_records / parent_experiment_id fields."""

    def test_default_record_type_is_training(self):
        r = _make_record()
        assert r.record_type == "training"
        assert r.sub_records == []
        assert r.parent_experiment_id == ""

    def test_record_type_enum_values(self):
        from hft_ops.ledger.experiment_record import RecordType
        assert RecordType.TRAINING.value == "training"
        assert RecordType.ANALYSIS.value == "analysis"
        assert RecordType.CALIBRATION.value == "calibration"
        assert RecordType.BACKTEST.value == "backtest"
        assert RecordType.EVALUATION.value == "evaluation"
        assert RecordType.SWEEP_AGGREGATE.value == "sweep_aggregate"

    def test_analysis_record_roundtrip(self, tmp_path: Path):
        from hft_ops.ledger.experiment_record import RecordType
        r = _make_record(experiment_id="e7_regime")
        r.record_type = RecordType.ANALYSIS.value
        r.training_metrics = {"diagnostic": "regime_filter_no_value"}
        path = tmp_path / "e7.json"
        r.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.record_type == "analysis"

    def test_calibration_with_parent(self, tmp_path: Path):
        from hft_ops.ledger.experiment_record import RecordType
        r = _make_record(experiment_id="e6_calibrated")
        r.record_type = RecordType.CALIBRATION.value
        r.parent_experiment_id = "e5_60s_huber_cvml"
        path = tmp_path / "e6.json"
        r.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.record_type == "calibration"
        assert loaded.parent_experiment_id == "e5_60s_huber_cvml"

    def test_sweep_aggregate_with_sub_records(self, tmp_path: Path):
        from hft_ops.ledger.experiment_record import RecordType
        r = _make_record(experiment_id="e4_baselines_aggregate")
        r.record_type = RecordType.SWEEP_AGGREGATE.value
        r.sub_records = [
            {"name": "ridge", "training_metrics": {"r_squared": 0.32}},
            {"name": "gradboost", "training_metrics": {"r_squared": 0.40}},
        ]
        path = tmp_path / "e4.json"
        r.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.record_type == "sweep_aggregate"
        assert len(loaded.sub_records) == 2
        assert loaded.sub_records[1]["name"] == "gradboost"

    def test_index_entry_includes_record_type_and_retroactive(self, tmp_path: Path):
        r = _make_record(experiment_id="indexed")
        r.record_type = "evaluation"
        r.parent_experiment_id = "parent_exp"
        r.provenance.retroactive = True
        entry = r.index_entry()
        assert entry["record_type"] == "evaluation"
        assert entry["parent_experiment_id"] == "parent_exp"
        assert entry["retroactive"] is True

    def test_old_record_loads_with_default_record_type(self, tmp_path: Path):
        """Backward compat: old JSON without record_type loads as 'training'."""
        import json
        # Manually construct old-format JSON (no record_type field)
        old_data = {
            "experiment_id": "legacy",
            "name": "legacy_exp",
            "fingerprint": "fp_legacy",
            "provenance": {"git": {}, "config_hashes": {}, "contract_version": "2.2"},
            "tags": [],
            "training_metrics": {"accuracy": 0.6},
            "training_config": {},
            "status": "completed",
            "created_at": "2026-03-01T00:00:00+00:00",
        }
        path = tmp_path / "legacy.json"
        with open(path, "w") as f:
            json.dump(old_data, f)
        loaded = ExperimentRecord.load(path)
        assert loaded.record_type == "training"  # default
        assert loaded.sub_records == []
        assert loaded.parent_experiment_id == ""


class TestLedgerIndexSchemaEnvelope:
    """Phase 8B: ``index.json`` envelope format + auto-invalidation.

    The envelope wraps entries in ``{"schema": {"version": ...}, "entries": [...]}``.
    When on-disk MAJOR.MINOR version differs from code-side
    ``hft_contracts.INDEX_SCHEMA_VERSION``, ``_load_index`` auto-rebuilds
    (loudly logged) — closing the silent-omission class that occurred when
    ``ExperimentRecord.index_entry()`` whitelist extensions were invisible
    to old records until manual ``hft-ops ledger rebuild-index``.
    """

    def test_save_writes_envelope_format(self, tmp_path: Path) -> None:
        """`_save_index` produces envelope-shaped JSON with `schema` + `entries`."""
        import json

        from hft_contracts import INDEX_SCHEMA_VERSION

        ledger = ExperimentLedger(tmp_path)
        ledger.register(_make_record("t_aaaa_20260101T000000_11111111"))

        with open(tmp_path / "index.json", "r") as f:
            on_disk = json.load(f)

        assert isinstance(on_disk, dict), (
            f"envelope format should be dict with 'schema' + 'entries'; got {type(on_disk).__name__}"
        )
        assert "schema" in on_disk, "envelope must carry 'schema' key"
        assert "entries" in on_disk, "envelope must carry 'entries' key"
        assert on_disk["schema"]["version"] == INDEX_SCHEMA_VERSION
        assert "written_at" in on_disk["schema"]
        assert "last_rebuild_source" in on_disk["schema"]
        assert isinstance(on_disk["entries"], list)
        assert len(on_disk["entries"]) == 1

    def test_load_envelope_with_matching_version_no_rebuild(
        self, tmp_path: Path, caplog
    ) -> None:
        """Matching MAJOR.MINOR version → fast path, no rebuild, no WARN."""
        import logging

        # First ledger instance: establishes envelope with current version.
        ledger1 = ExperimentLedger(tmp_path)
        ledger1.register(_make_record("t_bbbb_20260101T000000_22222222"))

        # Second instance: should load directly from envelope without rebuild.
        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger2 = ExperimentLedger(tmp_path)

        assert ledger2.count() == 1
        # No WARN messages — matching version is the fast path.
        warn_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert warn_messages == [], (
            f"matching version should produce zero WARN messages; got {warn_messages}"
        )

    def test_load_legacy_bare_list_auto_migrates_to_envelope(
        self, tmp_path: Path, caplog
    ) -> None:
        """Pre-Phase-8B bare-list index.json auto-migrates on first load."""
        import json
        import logging

        # Simulate pre-Phase-8B state: bare-list index.json.
        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_cccc_20260101T000000_33333333")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        legacy_index = [record.index_entry()]
        with open(tmp_path / "index.json", "w") as f:
            json.dump(legacy_index, f)

        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger = ExperimentLedger(tmp_path)

        assert ledger.count() == 1

        # After load, the file should be in envelope format.
        with open(tmp_path / "index.json", "r") as f:
            on_disk = json.load(f)
        assert isinstance(on_disk, dict) and "entries" in on_disk, (
            "legacy bare-list must auto-migrate to envelope format on load"
        )
        assert on_disk["schema"]["last_rebuild_source"] == "auto_legacy_bare_list"

        # WARN must be emitted — migration is a silent-omission-class guard event.
        warn_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert any("auto_legacy_bare_list" in m for m in warn_messages), (
            f"legacy migration must WARN; got {warn_messages}"
        )

    def test_load_malformed_json_triggers_rebuild(
        self, tmp_path: Path, caplog
    ) -> None:
        """Truncated/malformed JSON at index.json → auto-rebuild from records/."""
        import logging

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_dddd_20260101T000000_44444444")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        # Simulate power-loss / mid-write truncation.
        with open(tmp_path / "index.json", "w") as f:
            f.write('{"schema": {"version": "1.0.0"')  # Truncated mid-dict

        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger = ExperimentLedger(tmp_path)

        # Rebuild should recover the on-disk record.
        assert ledger.count() == 1
        # WARN must surface the rebuild reason.
        warn_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert any("auto_malformed" in m for m in warn_messages), (
            f"malformed JSON must WARN; got {warn_messages}"
        )

    def test_load_version_mismatch_triggers_rebuild(
        self, tmp_path: Path, caplog
    ) -> None:
        """On-disk envelope with different MAJOR.MINOR → auto-rebuild + WARN.

        This is THE test for the silent-omission class that Phase 8B exists to
        eliminate. A developer extends ``index_entry()`` whitelist + bumps
        MINOR; on next load the old envelope is detected as stale and the
        index is regenerated.
        """
        import json
        import logging

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_eeee_20260101T000000_55555555")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        # Simulate an older code version having written the index.
        stale_envelope = {
            "schema": {
                "version": "0.9.0",  # Older MAJOR.MINOR
                "written_at": "2026-01-01T00:00:00+00:00",
                "last_rebuild_source": "manual",
            },
            "entries": [],  # Old projection may have had fewer keys
        }
        with open(tmp_path / "index.json", "w") as f:
            json.dump(stale_envelope, f)

        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger = ExperimentLedger(tmp_path)

        assert ledger.count() == 1, (
            "version-mismatch rebuild must re-project from records/*.json"
        )
        warn_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert any("version_mismatch" in m and "0.9.0" in m for m in warn_messages), (
            f"version mismatch must WARN with on-disk version; got {warn_messages}"
        )

        # After rebuild, on-disk envelope should carry the current version +
        # the rebuild-source diagnostic.
        with open(tmp_path / "index.json", "r") as f:
            on_disk = json.load(f)
        assert on_disk["schema"]["last_rebuild_source"].startswith(
            "auto_version_mismatch"
        )


class TestIndexSchemaNeedsRebuildHelper:
    """Phase 8B: direct tests on the `_index_schema_needs_rebuild` helper.

    Exercises the MAJOR.MINOR comparison logic independently of ledger
    construction, ensuring PATCH-only diffs do NOT trigger rebuild and
    unparseable versions DO (fail-safe).
    """

    def test_same_version_no_rebuild(self) -> None:
        from hft_contracts import INDEX_SCHEMA_VERSION
        from hft_ops.ledger.ledger import _index_schema_needs_rebuild

        assert not _index_schema_needs_rebuild(INDEX_SCHEMA_VERSION)

    def test_patch_diff_no_rebuild(self) -> None:
        """PATCH-only diff is reserved for docstring changes; no rebuild.

        Phase 8C-α Stage C.2 (2026-04-20) bumped code-side to "1.3.0", so
        this test uses "1.3.99" (same MAJOR.MINOR, differing PATCH) to
        preserve the PATCH-invariance contract.
        """
        from hft_ops.ledger.ledger import _index_schema_needs_rebuild

        assert not _index_schema_needs_rebuild("1.3.99")

    def test_minor_diff_triggers_rebuild(self) -> None:
        """MINOR diff is the default whitelist-extension trigger.

        Phase 8A.0 (2026-04-20): code-side is now "1.1.0", so the legacy
        envelope at "1.0.0" MUST trigger rebuild (this is the actual
        flow executed on Phase 8A.0's first post-deploy load).
        """
        from hft_ops.ledger.ledger import _index_schema_needs_rebuild

        assert _index_schema_needs_rebuild("1.0.0")

    def test_major_diff_triggers_rebuild(self) -> None:
        from hft_ops.ledger.ledger import _index_schema_needs_rebuild

        assert _index_schema_needs_rebuild("2.0.0")

    def test_load_index_robust_against_non_dict_schema_field(
        self, tmp_path: Path
    ) -> None:
        """Post-audit agent-A HIGH-1: if on-disk index.json has a
        ``schema`` field that is NOT a dict (e.g., user hand-edit to
        ``{"schema": "1.0.0", "entries": []}``), the previous
        ``data.get("schema", {}).get("version")`` would raise
        AttributeError — uncaught, crashing __init__ with a raw
        traceback, bypassing both auto-rebuild and strict-mode.
        Fix: defensive ``isinstance(schema_field, dict)`` check.
        """
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        records_dir = ledger_dir / "records"
        records_dir.mkdir()
        # Write malformed schema-as-string envelope
        idx_path = ledger_dir / "index.json"
        idx_path.write_text('{"schema": "1.0.0", "entries": []}')

        # Must NOT raise AttributeError; must gracefully route to
        # rebuild path (or strict-mode StaleLedgerIndexError).
        import json as _json
        from hft_ops.ledger.ledger import ExperimentLedger
        ledger = ExperimentLedger(ledger_dir)  # non-strict → rebuild
        # After rebuild the envelope now has a proper dict schema
        data = _json.loads((ledger_dir / "index.json").read_text())
        assert isinstance(data.get("schema"), dict), (
            "Rebuild must produce a proper dict schema envelope"
        )

    def test_unparseable_version_triggers_rebuild(self) -> None:
        """Fail-safe: unrecognised / missing version forces rebuild."""
        from hft_ops.ledger.ledger import _index_schema_needs_rebuild

        assert _index_schema_needs_rebuild("")
        assert _index_schema_needs_rebuild("not-a-version")
        assert _index_schema_needs_rebuild("1.0")  # Missing PATCH
        assert _index_schema_needs_rebuild("1.0.0-pre")  # Pre-release not supported


class TestPersistPostStageArtifacts:
    """Phase 8C-α Stage C.3: post-stage artifact routing invariants."""

    def _make_artifact_dict(self, feature_name: str = "depth_norm_ofi") -> Dict:
        """Minimal valid FeatureImportanceArtifact dict for fixture use."""
        return {
            "schema_version": "1",
            "method": "permutation",
            "baseline_metric": "val_ic",
            "baseline_value": 0.245,
            "block_size_days": 1,
            "n_permutations": 500,
            "n_seeds": 5,
            "seed": 42,
            "eval_split": "test",
            "features": [
                {
                    "feature_name": feature_name,
                    "feature_index": 85,
                    "importance_mean": 0.023,
                    "importance_std": 0.004,
                    "ci_lower_95": 0.015,
                    "ci_upper_95": 0.031,
                    "n_permutations": 500,
                    "n_seeds_aggregated": 5,
                    "stability": 0.85,
                }
            ],
            "feature_set_ref": {"name": "test_v1", "content_hash": "a" * 64},
            "experiment_id": "exp_test",
            "fingerprint": "b" * 64,
            "model_type": "tlob",
            "timestamp_utc": "2026-04-20T12:00:00+00:00",
            "method_caveats": [],
        }

    def _make_record(self) -> ExperimentRecord:
        """Minimal ExperimentRecord fixture."""
        return ExperimentRecord(
            experiment_id="test_exp",
            name="test",
            fingerprint="f" * 64,
            contract_version="2.2",
            status="completed",
            created_at="2026-04-20T00:00:00+00:00",
            provenance=Provenance(
                git=GitInfo(commit_hash="x", branch="main", dirty=False),
                contract_version="2.2",
            ),
        )

    def test_routes_feature_importance_artifact_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """Happy path: trainer writes feature_importance_v1.json →
        ledger.persist_post_stage_artifacts() → file copied to
        content-addressed storage → record.artifacts[] populated.
        """
        import json as _json
        # Create pipeline-root-like layout
        pipeline_root = tmp_path / "pipeline"
        ledger_dir = pipeline_root / "hft-ops" / "ledger"
        ledger_dir.mkdir(parents=True)
        output_dir = pipeline_root / "outputs" / "test_exp"
        output_dir.mkdir(parents=True)

        # Write a trainer-like artifact
        artifact_path = output_dir / "feature_importance_v1.json"
        artifact_data = self._make_artifact_dict()
        artifact_path.write_text(_json.dumps(artifact_data, sort_keys=True))

        # Route
        ledger = ExperimentLedger(ledger_dir)
        record = self._make_record()
        count = ledger.persist_post_stage_artifacts(
            "training", output_dir, record,
        )

        assert count == 1
        assert len(record.artifacts) == 1

        entry = record.artifacts[0]
        assert entry["kind"] == "feature_importance"
        assert entry["method"] == "permutation"
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0

        # Target file must exist in ledger storage
        target_path_rel = entry["path"]
        target_path = pipeline_root / target_path_rel
        assert target_path.exists(), (
            f"Artifact must be copied to ledger storage at "
            f"{target_path} (relative: {target_path_rel})"
        )
        assert target_path.read_bytes() == artifact_path.read_bytes(), (
            "Routed file content must match source byte-for-byte"
        )

    def test_content_address_is_sha256_stable(self, tmp_path: Path) -> None:
        """Same content → same SHA-256 → same target path. Invariant for
        the content-addressing contract.
        """
        import json as _json
        pipeline_root = tmp_path / "pipeline"
        ledger_dir = pipeline_root / "hft-ops" / "ledger"
        ledger_dir.mkdir(parents=True)
        output_dir = pipeline_root / "outputs" / "e1"
        output_dir.mkdir(parents=True)
        artifact_data = self._make_artifact_dict()
        (output_dir / "feature_importance_v1.json").write_text(
            _json.dumps(artifact_data, sort_keys=True)
        )

        ledger = ExperimentLedger(ledger_dir)
        rec1 = self._make_record()
        ledger.persist_post_stage_artifacts("training", output_dir, rec1)
        sha1 = rec1.artifacts[0]["sha256"]

        # Re-route the SAME content (e.g., second run with identical output)
        rec2 = self._make_record()
        ledger.persist_post_stage_artifacts("training", output_dir, rec2)
        sha2 = rec2.artifacts[0]["sha256"]

        assert sha1 == sha2, "Same file content MUST produce same SHA-256"
        assert rec1.artifacts[0]["path"] == rec2.artifacts[0]["path"], (
            "Same SHA → same target path (content-addressed storage)"
        )

    def test_idempotent_re_routing_does_not_duplicate(
        self, tmp_path: Path
    ) -> None:
        """Re-routing an already-stored artifact is a no-op for the
        filesystem (target already exists, copy skipped) but still
        appends metadata to record.artifacts[] (different record may
        need the reference).
        """
        import json as _json
        pipeline_root = tmp_path / "pipeline"
        ledger_dir = pipeline_root / "hft-ops" / "ledger"
        ledger_dir.mkdir(parents=True)
        output_dir = pipeline_root / "outputs" / "e1"
        output_dir.mkdir(parents=True)
        (output_dir / "feature_importance_v1.json").write_text(
            _json.dumps(self._make_artifact_dict(), sort_keys=True)
        )

        ledger = ExperimentLedger(ledger_dir)
        rec1 = self._make_record()
        ledger.persist_post_stage_artifacts("training", output_dir, rec1)
        target_after_first = pipeline_root / rec1.artifacts[0]["path"]
        first_mtime = target_after_first.stat().st_mtime

        # Re-route — target file should NOT be touched (copy skipped)
        rec2 = self._make_record()
        ledger.persist_post_stage_artifacts("training", output_dir, rec2)
        target_after_second = pipeline_root / rec2.artifacts[0]["path"]
        second_mtime = target_after_second.stat().st_mtime

        assert target_after_first == target_after_second
        assert first_mtime == second_mtime, (
            "Idempotent re-routing must NOT re-copy (preserves target mtime)"
        )

    def test_missing_output_dir_returns_zero(self, tmp_path: Path) -> None:
        """Graceful no-op when stage was SKIPPED or FAILED and output_dir
        doesn't exist. Does NOT raise.
        """
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        ledger = ExperimentLedger(ledger_dir)
        rec = self._make_record()

        count = ledger.persist_post_stage_artifacts("training", None, rec)
        assert count == 0
        assert rec.artifacts == []

        count2 = ledger.persist_post_stage_artifacts(
            "training", tmp_path / "nonexistent", rec,
        )
        assert count2 == 0
        assert rec.artifacts == []

    def test_malformed_artifact_skipped_with_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """hft-rules §8 boundary validation: malformed artifacts are
        logged + skipped, never partial-routed. Legal file present but
        content is not a valid FeatureImportanceArtifact → skipped.
        """
        import logging as _logging
        pipeline_root = tmp_path / "pipeline"
        ledger_dir = pipeline_root / "hft-ops" / "ledger"
        ledger_dir.mkdir(parents=True)
        output_dir = pipeline_root / "outputs" / "e_bad"
        output_dir.mkdir(parents=True)

        # Write a file with the expected filename but missing required fields
        (output_dir / "feature_importance_v1.json").write_text(
            '{"schema_version": "1", "method": "permutation"}'
            # missing required fields: baseline_metric, features, etc.
        )

        ledger = ExperimentLedger(ledger_dir)
        rec = self._make_record()
        with caplog.at_level(_logging.WARNING):
            count = ledger.persist_post_stage_artifacts(
                "training", output_dir, rec,
            )

        assert count == 0, "Malformed artifact must NOT be routed"
        assert rec.artifacts == [], "record.artifacts must remain empty"
        # The validator must have produced a warning
        assert any(
            "validation failed" in r.message.lower()
            or "malformed" in r.message.lower()
            for r in caplog.records
        ), (
            f"Expected WARN about validation failure. Got: "
            f"{[r.message for r in caplog.records]}"
        )


class TestAllIndexConsumersGoThroughLoadIndex:
    """Phase 8B Step B.3 regression: lock the invariant that every ledger
    consumer method reads from ``self._index`` (populated once by
    ``_load_index``) rather than by direct ``json.load`` on the on-disk
    ``index.json`` file. Agent 1's adversarial validation flagged that
    ``dedup.py::check_duplicate`` was the only direct-read site (already
    fixed in Step B.2a); this regression guard ensures no future
    refactor re-introduces a direct-read path that bypasses the
    ``_load_index`` envelope + auto-rebuild logic.

    Method: construct a ledger with one record, confirm ``index.json``
    exists, then RENAME the file so any attempt to re-read it raises
    ``FileNotFoundError``. Call every consumer — if the invariant holds
    they all work from the in-memory ``self._index`` without re-touching
    the renamed file. If a future commit re-introduces a direct read,
    the rename causes an observable failure.
    """

    def test_consumers_work_after_index_file_renamed(
        self, tmp_path: Path
    ) -> None:
        import os

        ledger = ExperimentLedger(tmp_path)
        record = _make_record("t_audit_20260420T000000_auditaud")
        ledger.register(record)

        index_path = tmp_path / "index.json"
        renamed_path = tmp_path / "_index_renamed.json"
        assert index_path.exists(), "register() should have produced index.json"
        os.rename(index_path, renamed_path)
        assert not index_path.exists(), "rename must remove the original path"

        # All 6 consumer methods must work from self._index (in-memory),
        # not from a re-read of the (now-absent) index.json.
        assert len(ledger.list_all()) == 1, "list_all must use self._index"
        assert ledger.list_ids() == [record.experiment_id], (
            "list_ids must use self._index"
        )
        assert ledger.count() == 1, "count must use self._index"
        assert len(ledger.filter()) == 1, "filter must use self._index"
        assert ledger.find_by_fingerprint("nonexistent") is None, (
            "find_by_fingerprint must use self._index (returning None is OK — "
            "the point is that it doesn't crash on missing file)"
        )
        found_entry = ledger.find_by_fingerprint(record.fingerprint)
        assert found_entry is not None, (
            "find_by_fingerprint must locate the real fingerprint via self._index"
        )
        summary = ledger.summary()
        assert summary["total_experiments"] == 1, (
            "summary must use self._index"
        )


class TestStrictIndexMode:
    """Phase 8B Step B.2b: ``strict_index=True`` elevates schema mismatch
    from "auto-rebuild + WARN" to "fail-fast with StaleLedgerIndexError".

    Intended for CI: a developer who extends ``index_entry()`` whitelist
    without bumping ``INDEX_SCHEMA_VERSION``, or who forgets to commit a
    refreshed ``index.json``, sees a hard CI failure rather than silent
    auto-rebuild that masks the drift.

    Three trigger paths all must raise under strict mode:
    1. Version mismatch (older MAJOR.MINOR on disk vs code).
    2. Legacy bare-list format.
    3. Malformed JSON (truncated / non-list-non-dict root).

    A fourth test verifies CI=true auto-detection (resolves the prior
    open question about whether env detection should be the default).
    """

    def test_strict_mode_raises_on_version_mismatch(
        self, tmp_path: Path
    ) -> None:
        import json

        from hft_ops.ledger.ledger import StaleLedgerIndexError

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_strict_v_20260420T000000_strictv00")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        stale_envelope = {
            "schema": {
                "version": "0.9.0",  # Older MAJOR.MINOR than current 1.0.0.
                "written_at": "2026-01-01T00:00:00+00:00",
                "last_rebuild_source": "manual",
            },
            "entries": [],
        }
        with open(tmp_path / "index.json", "w") as f:
            json.dump(stale_envelope, f)

        with pytest.raises(StaleLedgerIndexError) as exc_info:
            ExperimentLedger(tmp_path, strict_index=True)

        msg = str(exc_info.value)
        assert "stale" in msg.lower(), (
            f"error message must include 'stale'; got: {msg}"
        )
        assert "0.9.0" in msg, (
            f"error message must include the on-disk version; got: {msg}"
        )
        assert "rebuild-index" in msg, (
            f"error message must suggest the rebuild-index recovery path; got: {msg}"
        )

    def test_strict_mode_raises_on_legacy_bare_list(
        self, tmp_path: Path
    ) -> None:
        import json

        from hft_ops.ledger.ledger import StaleLedgerIndexError

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy_format"}], f)

        with pytest.raises(StaleLedgerIndexError, match=r"legacy"):
            ExperimentLedger(tmp_path, strict_index=True)

    def test_strict_mode_raises_on_malformed_json(self, tmp_path: Path) -> None:
        from hft_ops.ledger.ledger import StaleLedgerIndexError

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "index.json", "w") as f:
            f.write('{"schema": {"version":')  # Truncated mid-dict.

        with pytest.raises(StaleLedgerIndexError, match=r"malformed"):
            ExperimentLedger(tmp_path, strict_index=True)

    def test_strict_mode_does_NOT_raise_on_fresh_ledger(
        self, tmp_path: Path
    ) -> None:
        """First-time init on an empty ledger dir is NOT a strict-fail
        condition — a fresh ledger is the expected starting state, not
        drift. Strict only fires when an existing index.json signals
        staleness (version mismatch, legacy format, corruption).
        """
        # Fresh tmp_path — no records/ or index.json exists yet.
        ledger = ExperimentLedger(tmp_path, strict_index=True)
        assert ledger.count() == 0

    def test_strict_mode_does_NOT_raise_on_matching_version(
        self, tmp_path: Path
    ) -> None:
        """Current INDEX_SCHEMA_VERSION envelope should load without
        complaint under strict mode (the fast path).
        """
        ledger1 = ExperimentLedger(tmp_path, strict_index=False)
        ledger1.register(_make_record("t_strict_ok_20260420T000000_strictok0"))

        # Second construction in strict mode — should load cleanly.
        ledger2 = ExperimentLedger(tmp_path, strict_index=True)
        assert ledger2.count() == 1

    def test_ci_env_var_enables_strict_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """CI=true env var auto-enables strict mode when
        ``strict_index`` kwarg is not explicitly passed (resolving the
        prior open question about whether env detection should be the
        default). Mirrors GitHub Actions / GitLab CI / CircleCI /
        Buildkite runner conventions.
        """
        import json

        from hft_ops.ledger.ledger import StaleLedgerIndexError

        # Set CI=true in environment.
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("HFT_OPS_STRICT_INDEX", raising=False)

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        # Write a stale envelope to trigger the check.
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy"}], f)

        # Note: no explicit strict_index arg; env var should elevate.
        with pytest.raises(StaleLedgerIndexError):
            ExperimentLedger(tmp_path)

    def test_hft_ops_strict_index_env_var_enables_strict_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """HFT_OPS_STRICT_INDEX=1 env var (explicit override) also
        enables strict mode when no kwarg is passed.
        """
        import json

        from hft_ops.ledger.ledger import StaleLedgerIndexError

        monkeypatch.setenv("HFT_OPS_STRICT_INDEX", "1")
        monkeypatch.delenv("CI", raising=False)

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy"}], f)

        with pytest.raises(StaleLedgerIndexError):
            ExperimentLedger(tmp_path)

    # Phase 8B MUST-FIX (Agent 2 P1, 2026-04-20): symmetry + negative-case
    # coverage for env-var truthy parsing.

    def test_hft_ops_strict_index_accepts_true_alias(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """HFT_OPS_STRICT_INDEX=true (lower/upper case), "yes", "on"
        also enable strict mode — symmetric with CI env var parsing.
        """
        import json

        from hft_ops.ledger.ledger import StaleLedgerIndexError

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy"}], f)

        for truthy in ("true", "TRUE", "True", "yes", "YES", "on", "1"):
            monkeypatch.setenv("HFT_OPS_STRICT_INDEX", truthy)
            monkeypatch.delenv("CI", raising=False)
            with pytest.raises(StaleLedgerIndexError):
                ExperimentLedger(tmp_path)

    def test_strict_mode_NOT_triggered_by_falsy_env_values(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Negative case: CI=false, CI=0, HFT_OPS_STRICT_INDEX=false,
        HFT_OPS_STRICT_INDEX=0, and empty-string values all leave strict
        mode DISABLED. Prevents CI gate bypass if user sets a falsy
        literal, expecting it to mean "no strict mode".
        """
        import json

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        # Legacy-format index on disk — would trigger strict mode if enabled.
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy"}], f)

        falsy_values = [("CI", "false"), ("CI", "0"), ("CI", "FALSE"),
                        ("CI", "no"), ("CI", "off"), ("CI", ""),
                        ("HFT_OPS_STRICT_INDEX", "false"),
                        ("HFT_OPS_STRICT_INDEX", "0"),
                        ("HFT_OPS_STRICT_INDEX", "")]
        for var_name, var_value in falsy_values:
            monkeypatch.setenv(var_name, var_value)
            monkeypatch.delenv("CI" if var_name != "CI" else "HFT_OPS_STRICT_INDEX",
                               raising=False)
            # Should NOT raise — auto-rebuild proceeds since strict is off.
            ledger = ExperimentLedger(tmp_path)
            assert ledger is not None, (
                f"{var_name}={var_value!r} must NOT enable strict mode"
            )


class TestDedupCheckDuplicateEnvelopeCompat:
    """Phase 8B MUST-FIX (Agent 1 BUG-1, 2026-04-20):
    ``dedup.py::check_duplicate`` previously did a direct ``json.load`` on
    ``index.json`` bypassing ``ExperimentLedger._load_index`` — breaking
    the strict-mode contract and allowing silent duplicate registration
    when on-disk envelope was stale. Fixed by routing the dedup check
    through ``ExperimentLedger``. These tests lock the post-fix behavior.
    """

    def test_check_duplicate_reads_envelope_format(self, tmp_path: Path) -> None:
        """After registering a record (envelope on disk), `check_duplicate`
        finds the fingerprint via the envelope's `entries` list — matching
        what `ExperimentLedger.find_by_fingerprint` would return.
        """
        from hft_ops.ledger.dedup import check_duplicate

        ledger = ExperimentLedger(tmp_path)
        record = _make_record("t_dedup_env_20260420T000000_dedupenv1")
        ledger.register(record)

        # On-disk is envelope-format post-register.
        found = check_duplicate(record.fingerprint, tmp_path)
        assert found is not None, (
            "check_duplicate must find registered fingerprint in envelope"
        )
        assert found["experiment_id"] == record.experiment_id

    def test_check_duplicate_reads_legacy_bare_list_via_autoload(
        self, tmp_path: Path
    ) -> None:
        """Legacy bare-list index.json on disk: `check_duplicate` goes
        through `ExperimentLedger._load_index` which auto-migrates to
        envelope on construction. Dedup then succeeds against the
        re-projected entries.
        """
        import json

        from hft_ops.ledger.dedup import check_duplicate

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_dedup_legacy_20260420T000000_dedupleg")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        # Write legacy bare-list with the record already present.
        legacy = [record.index_entry()]
        with open(tmp_path / "index.json", "w") as f:
            json.dump(legacy, f)

        # check_duplicate should find the record (auto-migration re-projects).
        found = check_duplicate(record.fingerprint, tmp_path)
        assert found is not None, (
            "check_duplicate must find fingerprint via auto-migrated envelope"
        )

    def test_check_duplicate_propagates_stale_error_under_strict_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Under `CI=true` (strict), `check_duplicate` on a stale envelope
        must raise `StaleLedgerIndexError` — NOT silently return None. This
        was the critical BUG-1 fix: prior to Phase 8B MUST-FIX, `check_duplicate`
        was the most-called silent-failure vector for stale-index drift.
        """
        import json

        from hft_ops.ledger.dedup import check_duplicate
        from hft_ops.ledger.ledger import StaleLedgerIndexError

        monkeypatch.setenv("CI", "true")

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        with open(tmp_path / "index.json", "w") as f:
            json.dump([{"experiment_id": "legacy"}], f)  # Legacy bare-list.

        with pytest.raises(StaleLedgerIndexError):
            check_duplicate("any_fingerprint", tmp_path)

    def test_check_duplicate_returns_none_when_ledger_dir_missing(
        self, tmp_path: Path
    ) -> None:
        """Preserve permissive behavior: missing ledger dir → return None
        (caller proceeds without dedup). This path pre-dates Phase 8B but
        must continue to work through the refactor.
        """
        from hft_ops.ledger.dedup import check_duplicate

        # Never-created index.json → early-return.
        assert check_duplicate("fp", tmp_path) is None


class TestAutoMigrationRoundtrip:
    """Phase 8B MUST-ADD (Agent 2 P1): explicit roundtrip after rebuild.
    After auto-migration from legacy bare-list, a second
    `ExperimentLedger(dir)` MUST hit the fast path with zero WARN log —
    proves the auto-migration output is re-loadable at fast-path speed.
    """

    def test_second_load_hits_fast_path_after_legacy_migration(
        self, tmp_path: Path, caplog
    ) -> None:
        import json
        import logging

        (tmp_path / "records").mkdir(parents=True, exist_ok=True)
        record = _make_record("t_migrate_20260420T000000_migrate00")
        record.save(tmp_path / "records" / f"{record.experiment_id}.json")

        # Pre-Phase-8B legacy bare-list.
        with open(tmp_path / "index.json", "w") as f:
            json.dump([record.index_entry()], f)

        # First load: migrates — expect WARN.
        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger1 = ExperimentLedger(tmp_path)
        assert ledger1.count() == 1
        migrate_warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("auto_legacy_bare_list" in m for m in migrate_warns), (
            f"first load must WARN on migration; got {migrate_warns}"
        )

        # Second load: envelope is now current, fast path — expect ZERO warn.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hft_ops.ledger.ledger"):
            ledger2 = ExperimentLedger(tmp_path)
        assert ledger2.count() == 1
        post_migration_warns = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert post_migration_warns == [], (
            f"second load after migration must hit fast path (no WARN); "
            f"got {post_migration_warns}"
        )
