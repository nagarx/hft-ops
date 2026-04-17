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
