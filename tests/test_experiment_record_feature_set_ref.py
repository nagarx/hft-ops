"""Phase 4 Batch 4c.4: ExperimentRecord.feature_set_ref round-trip + index_entry.

Locks:
1. `feature_set_ref: {name, content_hash}` round-trips through to_dict/from_dict.
2. `index_entry()` surfaces the field (flattened to dict, empty when None).
3. Legacy records (no feature_set_ref field in JSON) load with the field = None.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hft_ops.ledger.experiment_record import ExperimentRecord


class TestFeatureSetRefRoundTrip:
    def test_to_dict_and_from_dict_preserve_ref(self):
        ref = {"name": "momentum_v1", "content_hash": "a" * 64}
        record = ExperimentRecord(
            name="test",
            fingerprint="f" * 64,
            contract_version="2.2",
            feature_set_ref=ref,
        )
        d = record.to_dict()
        assert d["feature_set_ref"] == ref

        record2 = ExperimentRecord.from_dict(d)
        assert record2.feature_set_ref == ref

    def test_none_ref_roundtrip(self):
        record = ExperimentRecord(name="test", feature_set_ref=None)
        d = record.to_dict()
        assert d.get("feature_set_ref") is None

        record2 = ExperimentRecord.from_dict(d)
        assert record2.feature_set_ref is None


class TestIndexEntrySurface:
    def test_index_entry_contains_feature_set_ref(self):
        ref = {"name": "x_v1", "content_hash": "b" * 64}
        record = ExperimentRecord(name="test", feature_set_ref=ref)
        entry = record.index_entry()
        assert entry["feature_set_ref"] == ref, (
            "index_entry must surface feature_set_ref for "
            "`hft-ops ledger list --feature-set <name>` filtering."
        )

    def test_index_entry_empty_dict_when_none(self):
        record = ExperimentRecord(name="test", feature_set_ref=None)
        entry = record.index_entry()
        assert entry["feature_set_ref"] == {}, (
            "Plan convention: empty dict (not None) when feature_set_ref is unset."
        )


class TestLegacyRecordCompat:
    def test_legacy_json_without_field_loads_with_none(self):
        """A ledger record.json written BEFORE Batch 4c.4 has no feature_set_ref
        key. The post-4c.4 loader must default it to None."""
        legacy_data = {
            "experiment_id": "legacy_exp_001",
            "name": "legacy",
            "manifest_path": "/tmp/legacy.yaml",
            "fingerprint": "e" * 64,
            "provenance": {},
            "contract_version": "2.2",
            "extraction_config": {},
            "training_config": {},
            "backtest_params": {},
            "training_metrics": {},
            "backtest_metrics": {},
            "dataset_health": {},
            "tags": [],
            "hypothesis": "",
            "description": "",
            "notes": "",
            "created_at": "2026-04-15T00:00:00+00:00",
            "duration_seconds": 0.0,
            "status": "completed",
            "stages_completed": [],
        }
        record = ExperimentRecord.from_dict(legacy_data)
        assert record.feature_set_ref is None

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        ref = {"name": "roundtrip_v1", "content_hash": "c" * 64}
        record = ExperimentRecord(
            name="roundtrip",
            fingerprint="g" * 64,
            contract_version="2.2",
            feature_set_ref=ref,
        )
        path = tmp_path / "rec.json"
        record.save(path)
        loaded = ExperimentRecord.load(path)
        assert loaded.feature_set_ref == ref
