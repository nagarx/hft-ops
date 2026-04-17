"""Phase 4 Batch 4c.4: `_harvest_feature_set_ref` unit tests.

Locks the signal_metadata.json harvester behavior in isolation:
1. Harvest from flat layout (`<output_dir>/signal_metadata.json`)
2. Harvest from nested layout (`<output_dir>/<split>/signal_metadata.json`)
3. Missing file → None (best-effort, no crash)
4. Malformed JSON → None
5. Missing feature_set_ref key → None
"""

from __future__ import annotations

import json
from pathlib import Path

from hft_ops.stages.signal_export import _harvest_feature_set_ref


class TestHarvestFlatLayout:
    def test_flat_layout_with_ref(self, tmp_path: Path):
        meta = {
            "signal_type": "regression",
            "feature_set_ref": {"name": "x_v1", "content_hash": "a" * 64},
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        result = _harvest_feature_set_ref(tmp_path)
        assert result == {"name": "x_v1", "content_hash": "a" * 64}


class TestHarvestNestedLayout:
    def test_nested_layout_with_ref(self, tmp_path: Path):
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        meta = {
            "signal_type": "classification",
            "feature_set_ref": {"name": "nested_v1", "content_hash": "b" * 64},
        }
        (split_dir / "signal_metadata.json").write_text(json.dumps(meta))
        result = _harvest_feature_set_ref(tmp_path)
        assert result == {"name": "nested_v1", "content_hash": "b" * 64}


class TestBestEffortNone:
    def test_missing_dir_returns_none(self, tmp_path: Path):
        assert _harvest_feature_set_ref(tmp_path / "does_not_exist") is None

    def test_none_arg_returns_none(self):
        assert _harvest_feature_set_ref(None) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        # dir exists but no signal_metadata.json
        assert _harvest_feature_set_ref(tmp_path) is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        (tmp_path / "signal_metadata.json").write_text("{not valid json")
        assert _harvest_feature_set_ref(tmp_path) is None

    def test_missing_ref_key_returns_none(self, tmp_path: Path):
        meta = {"signal_type": "regression"}  # no feature_set_ref
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_feature_set_ref(tmp_path) is None

    def test_invalid_ref_shape_returns_none(self, tmp_path: Path):
        # feature_set_ref present but missing content_hash
        meta = {"feature_set_ref": {"name": "x"}}
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_feature_set_ref(tmp_path) is None

        # feature_set_ref not a dict
        meta2 = {"feature_set_ref": "not_a_dict"}
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta2))
        assert _harvest_feature_set_ref(tmp_path) is None
