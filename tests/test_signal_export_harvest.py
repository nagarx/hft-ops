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

from hft_ops.stages.signal_export import (
    _harvest_compatibility_fingerprint,
    _harvest_feature_set_ref,
)


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


# =============================================================================
# Phase V.A.4 (2026-04-21): _harvest_compatibility_fingerprint unit tests.
# Mirror the _harvest_feature_set_ref test matrix — same best-effort semantics,
# same dual-layout support (flat + nested), same CONTENT_HASH_RE validation
# gate.
# =============================================================================


class TestHarvestCompatFpFlatLayout:
    def test_top_level_field(self, tmp_path: Path):
        """Primary harvest path: top-level `compatibility_fingerprint` field."""
        hex_fp = "a" * 64
        meta = {
            "signal_type": "regression",
            "compatibility_fingerprint": hex_fp,
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_compatibility_fingerprint(tmp_path) == hex_fp

    def test_nested_under_compatibility_block(self, tmp_path: Path):
        """Forward-compat path: `compatibility.fingerprint` nested location.

        Exporters may emit fingerprint as a field of the compatibility
        block itself (same 64-hex value, different JSON location). Harvester
        accepts both — caller doesn't care which exporter shape was used.
        """
        hex_fp = "b" * 64
        meta = {
            "signal_type": "regression",
            "compatibility": {
                "contract_version": "2.2",
                "fingerprint": hex_fp,
                # (other fields omitted for brevity)
            },
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_compatibility_fingerprint(tmp_path) == hex_fp

    def test_top_level_takes_precedence_over_nested(self, tmp_path: Path):
        """If both locations present, top-level wins (explicit > derived)."""
        top_fp = "c" * 64
        nested_fp = "d" * 64
        meta = {
            "compatibility_fingerprint": top_fp,
            "compatibility": {"fingerprint": nested_fp},
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_compatibility_fingerprint(tmp_path) == top_fp


class TestHarvestCompatFpNestedLayout:
    def test_nested_layout_split_subdir(self, tmp_path: Path):
        split_dir = tmp_path / "test"
        split_dir.mkdir()
        hex_fp = "e" * 64
        meta = {
            "signal_type": "classification",
            "compatibility_fingerprint": hex_fp,
        }
        (split_dir / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_compatibility_fingerprint(tmp_path) == hex_fp


class TestHarvestCompatFpBestEffortNone:
    def test_missing_dir_returns_none(self, tmp_path: Path):
        assert _harvest_compatibility_fingerprint(tmp_path / "does_not_exist") is None

    def test_none_arg_returns_none(self):
        assert _harvest_compatibility_fingerprint(None) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        # dir exists but no signal_metadata.json
        assert _harvest_compatibility_fingerprint(tmp_path) is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        (tmp_path / "signal_metadata.json").write_text("{not valid json")
        assert _harvest_compatibility_fingerprint(tmp_path) is None

    def test_missing_fingerprint_returns_none(self, tmp_path: Path):
        """Legacy Phase-II-unaware manifest (no compatibility block)."""
        meta = {"signal_type": "regression"}
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_compatibility_fingerprint(tmp_path) is None

    def test_malformed_fingerprint_returns_none(self, tmp_path: Path):
        """Non-64-hex value → silently drop (CONTENT_HASH_RE gate)."""
        for bad in [
            "not-hex",                 # invalid chars
            "a" * 63,                  # too short
            "a" * 65,                  # too long
            "A" * 64,                  # uppercase (regex is lowercase-only)
            "gg" + "a" * 62,           # leading non-hex chars
            "",                        # empty string
        ]:
            meta = {"compatibility_fingerprint": bad}
            (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
            assert _harvest_compatibility_fingerprint(tmp_path) is None, (
                f"Malformed fingerprint {bad!r} should harvest as None; "
                f"got non-None"
            )

    def test_non_string_fingerprint_returns_none(self, tmp_path: Path):
        """Type-check gate: int / list / dict values are NOT valid fingerprints."""
        for bad in [42, ["a" * 64], {"hex": "a" * 64}, None]:
            meta = {"compatibility_fingerprint": bad}
            (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
            assert _harvest_compatibility_fingerprint(tmp_path) is None
