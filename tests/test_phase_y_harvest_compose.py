"""Phase Y deployment regression tests (2026-05-05).

Tests for the model_config_hash harvester + experiment_provenance_hash
composition + ledger filter wiring. Mirrors the V.A.4 compatibility_fingerprint
test surface (test_signal_export_harvest.py / test_cli_compatibility_fp_validator.py).

Coverage:
1. _harvest_model_config_hash happy paths (top-level + dual-layout)
2. _harvest_model_config_hash best-effort None (missing dir/file/JSON, absent field, malformed value)
3. _harvest_model_config_hash WARN observability (malformed uppercase, non-string, post-cutoff)
4. ExperimentLedger.filter(experiment_provenance_hash=...) exact-match
5. compute_experiment_provenance_hash end-to-end composition with all 4 fields populated
6. compute_experiment_provenance_hash returns None when any field missing
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from hft_ops.stages.signal_export import (
    _harvest_model_config_hash,
    MODEL_CONFIG_HASH_REQUIRED_AFTER_ISO,
)


_VALID_HASH = "a" * 64  # canonical 64-lowercase-hex
_VALID_HASH_2 = "b" * 64


# ============================================================================
# 1. Happy path harvest
# ============================================================================


class TestHarvestModelCfgHashFlatLayout:
    """signal_metadata.json at <output_dir>/signal_metadata.json — flat layout."""

    def _write_meta(self, output_dir: Path, **fields):
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",  # post-cutoff
            **fields,
        }
        (output_dir / "signal_metadata.json").write_text(
            json.dumps(meta, sort_keys=True)
        )

    def test_top_level_field(self, tmp_path: Path):
        self._write_meta(tmp_path, model_config_hash=_VALID_HASH)
        assert _harvest_model_config_hash(tmp_path) == _VALID_HASH

    def test_returns_lowercase_hex(self, tmp_path: Path):
        self._write_meta(tmp_path, model_config_hash=_VALID_HASH_2)
        result = _harvest_model_config_hash(tmp_path)
        assert result == _VALID_HASH_2
        assert result is not None and result.islower()


class TestHarvestModelCfgHashNestedLayout:
    """signal_metadata.json at <output_dir>/<split>/signal_metadata.json."""

    def test_split_subdir_layout(self, tmp_path: Path):
        # Mirror SignalExporter writing to <signals>/<split>/ pattern
        sub = tmp_path / "test"
        sub.mkdir()
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",
            "model_config_hash": _VALID_HASH,
        }
        (sub / "signal_metadata.json").write_text(json.dumps(meta, sort_keys=True))
        assert _harvest_model_config_hash(tmp_path) == _VALID_HASH


# ============================================================================
# 2. Best-effort None
# ============================================================================


class TestHarvestModelCfgHashBestEffortNone:
    def test_none_arg_returns_none(self):
        assert _harvest_model_config_hash(None) is None

    def test_missing_dir_returns_none(self, tmp_path: Path):
        assert _harvest_model_config_hash(tmp_path / "nonexistent") is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert _harvest_model_config_hash(tmp_path) is None

    def test_malformed_json_returns_none(self, tmp_path: Path):
        (tmp_path / "signal_metadata.json").write_text("not valid json {{{")
        assert _harvest_model_config_hash(tmp_path) is None

    def test_missing_field_returns_none(self, tmp_path: Path):
        # Pre-cutoff manifest with no model_config_hash → silent None
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-04-01T00:00:00Z",  # pre-cutoff
            "compatibility_fingerprint": _VALID_HASH,
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        assert _harvest_model_config_hash(tmp_path) is None

    def test_malformed_uppercase_returns_none(self, tmp_path: Path):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",
            "model_config_hash": _VALID_HASH.upper(),  # uppercase = malformed
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress the WARN; we test it separately
            assert _harvest_model_config_hash(tmp_path) is None

    def test_truncated_hex_returns_none(self, tmp_path: Path):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",
            "model_config_hash": "a" * 32,  # 32 chars, not 64
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _harvest_model_config_hash(tmp_path) is None

    def test_non_string_returns_none(self, tmp_path: Path):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",
            "model_config_hash": 42,  # int, not string
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _harvest_model_config_hash(tmp_path) is None


# ============================================================================
# 3. WARN observability (mirrors V.1.5 SDR-2 pattern)
# ============================================================================


class TestHarvestModelCfgHashWarningEmission:
    """Malformed values emit RuntimeWarning per hft-rules §8."""

    def _write(self, tmp_path: Path, value):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-05-06T00:00:00Z",
            "model_config_hash": value,
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))

    def test_uppercase_emits_warning(self, tmp_path: Path):
        self._write(tmp_path, _VALID_HASH.upper())
        with pytest.warns(RuntimeWarning, match="rejected malformed value"):
            assert _harvest_model_config_hash(tmp_path) is None

    def test_non_string_emits_warning(self, tmp_path: Path):
        self._write(tmp_path, 12345)
        with pytest.warns(RuntimeWarning, match="rejected malformed value"):
            assert _harvest_model_config_hash(tmp_path) is None

    def test_valid_emits_no_warning(self, tmp_path: Path):
        self._write(tmp_path, _VALID_HASH)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = _harvest_model_config_hash(tmp_path)
        assert result == _VALID_HASH
        runtime_warns = [w for w in captured if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) == 0


class TestHarvestModelCfgHashCutoffPolicy:
    """Phase Y cutoff (2026-05-05) — post-cutoff manifests missing the
    field emit a logger.warning (drift signal); pre-cutoff are silent."""

    def test_cutoff_constant_value(self):
        assert MODEL_CONFIG_HASH_REQUIRED_AFTER_ISO == "2026-05-05"

    def test_pre_cutoff_no_warn(self, tmp_path: Path, caplog):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-04-01T00:00:00Z",  # pre-cutoff
            # no model_config_hash
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level("WARNING"):
            result = _harvest_model_config_hash(tmp_path)
        assert result is None
        # No post-cutoff WARN should fire
        post_cutoff_warns = [
            r for r in caplog.records
            if "post-Phase-Y" in r.getMessage()
        ]
        assert len(post_cutoff_warns) == 0

    def test_post_cutoff_emits_warn(self, tmp_path: Path, caplog):
        meta = {
            "schema_version": "3.0",
            "exported_at": "2026-06-01T00:00:00Z",  # post-cutoff
            # no model_config_hash
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level("WARNING"):
            result = _harvest_model_config_hash(tmp_path)
        assert result is None
        post_cutoff_warns = [
            r for r in caplog.records
            if "post-Phase-Y" in r.getMessage()
        ]
        assert len(post_cutoff_warns) == 1


# ============================================================================
# 4. Composition end-to-end
# ============================================================================


class TestComputeExperimentProvenanceHash:
    """Phase Y composition wiring — verify all-4-present produces valid 64-hex,
    any-1-missing returns None gracefully."""

    def _build_record(self, **field_overrides):
        """Build a minimal ExperimentRecord with all 4 source fields populated."""
        from hft_contracts.experiment_record import ExperimentRecord
        from hft_contracts.provenance import Provenance, GitInfo

        # Default: all 4 source fields populated
        data_dir_hash = field_overrides.pop("data_dir_hash", "d" * 64)
        feature_set_content_hash = field_overrides.pop("feature_set_content_hash", "e" * 64)
        compatibility_fingerprint = field_overrides.pop("compatibility_fingerprint", "f" * 64)
        model_config_hash = field_overrides.pop("model_config_hash", "1" * 64)

        # Allow tests to clear specific fields by passing None
        provenance = Provenance(
            git=GitInfo(commit_hash="abc", branch="main", dirty=False),
            config_hashes={},
            data_dir_hash=data_dir_hash if data_dir_hash else "",
            contract_version="3.0",
            timestamp_utc="2026-05-06T00:00:00Z",
        )

        feature_set_ref = (
            {"name": "test_set", "content_hash": feature_set_content_hash}
            if feature_set_content_hash
            else None
        )

        training_config = {}
        if model_config_hash:
            training_config["model_config_hash"] = model_config_hash

        return ExperimentRecord(
            experiment_id="test_exp",
            name="test",
            manifest_path="/tmp/manifest.yaml",
            fingerprint="test_fp",
            feature_set_ref=feature_set_ref,
            compatibility_fingerprint=compatibility_fingerprint,
            provenance=provenance,
            contract_version="3.0",
            training_config=training_config,
            created_at="2026-05-06T00:00:00Z",
        )

    def test_all_4_present_returns_64_hex(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        import re
        record = self._build_record()
        result = compute_experiment_provenance_hash(record)
        assert result is not None
        assert re.match(r"^[0-9a-f]{64}$", result), (
            f"Expected 64-lowercase-hex SHA-256, got {result!r}"
        )

    def test_data_dir_hash_missing_returns_none(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record = self._build_record(data_dir_hash=None)
        assert compute_experiment_provenance_hash(record) is None

    def test_feature_set_ref_missing_returns_none(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record = self._build_record(feature_set_content_hash=None)
        assert compute_experiment_provenance_hash(record) is None

    def test_compat_fp_missing_returns_none(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record = self._build_record(compatibility_fingerprint=None)
        assert compute_experiment_provenance_hash(record) is None

    def test_model_config_hash_missing_returns_none(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record = self._build_record(model_config_hash=None)
        assert compute_experiment_provenance_hash(record) is None

    def test_same_inputs_same_output_deterministic(self):
        """Determinism invariant — composition is pure SHA-256 over canonical
        JSON; same inputs MUST produce same output across calls."""
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record_a = self._build_record()
        record_b = self._build_record()
        assert compute_experiment_provenance_hash(record_a) == compute_experiment_provenance_hash(record_b)

    def test_different_model_config_hash_different_output(self):
        """Phase Y composability invariant — different model_config_hash
        MUST produce different experiment_provenance_hash."""
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record_a = self._build_record(model_config_hash="a" * 64)
        record_b = self._build_record(model_config_hash="b" * 64)
        h_a = compute_experiment_provenance_hash(record_a)
        h_b = compute_experiment_provenance_hash(record_b)
        assert h_a is not None and h_b is not None
        assert h_a != h_b, (
            "Different model_config_hash MUST yield different "
            "experiment_provenance_hash for cross-experiment composability."
        )

    def test_different_compat_fp_different_output(self):
        from hft_contracts.experiment_record import compute_experiment_provenance_hash
        record_a = self._build_record(compatibility_fingerprint="a" * 64)
        record_b = self._build_record(compatibility_fingerprint="b" * 64)
        h_a = compute_experiment_provenance_hash(record_a)
        h_b = compute_experiment_provenance_hash(record_b)
        assert h_a != h_b


# ============================================================================
# 5. Ledger filter
# ============================================================================


class TestExperimentLedgerProvenanceHashFilter:
    """ExperimentLedger.filter(experiment_provenance_hash=...) exact-match."""

    def test_filter_signature_accepts_kwarg(self):
        """Phase Y deployment: experiment_provenance_hash kwarg added to
        ExperimentLedger.filter alongside compatibility_fingerprint. Locks
        the public API surface so removing it is a breaking change."""
        from hft_ops.ledger.ledger import ExperimentLedger
        import inspect
        sig = inspect.signature(ExperimentLedger.filter)
        assert "experiment_provenance_hash" in sig.parameters, (
            "ExperimentLedger.filter MUST accept experiment_provenance_hash kwarg "
            "for Phase Y deployment ledger queries."
        )
        # Must be keyword-only with default None (matches V.A.4 pattern)
        param = sig.parameters["experiment_provenance_hash"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_filter_exact_match(self, tmp_path: Path):
        from hft_ops.ledger.ledger import ExperimentLedger
        # Synthesize a ledger with 2 entries differing only in
        # experiment_provenance_hash. Index entry projection is the
        # filter target (per Phase V.A.4 pattern at ledger.py:743).
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        ledger = ExperimentLedger(ledger_dir)

        # Build minimal index entries directly (bypass full record cycle
        # to keep this test focused on the filter logic).
        ledger._index = [
            {
                "experiment_id": "exp_a",
                "experiment_provenance_hash": "a" * 64,
                "status": "completed",
            },
            {
                "experiment_id": "exp_b",
                "experiment_provenance_hash": "b" * 64,
                "status": "completed",
            },
        ]

        # Filter to exp_a
        results = ledger.filter(experiment_provenance_hash="a" * 64)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_a"

        # Filter to exp_b
        results = ledger.filter(experiment_provenance_hash="b" * 64)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_b"

        # No filter — both pass through
        results = ledger.filter()
        assert len(results) == 2

    def test_filter_empty_string_matches_records_without_hash(self, tmp_path: Path):
        """Phase V.A.4 convention: explicit empty-string filter matches
        records WITHOUT a populated hash (graceful-degradation projection)."""
        from hft_ops.ledger.ledger import ExperimentLedger
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        ledger = ExperimentLedger(ledger_dir)
        ledger._index = [
            {"experiment_id": "exp_legacy", "experiment_provenance_hash": ""},
            {"experiment_id": "exp_new", "experiment_provenance_hash": "a" * 64},
        ]
        results = ledger.filter(experiment_provenance_hash="")
        assert len(results) == 1
        assert results[0]["experiment_id"] == "exp_legacy"
