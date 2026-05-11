"""Tests for ``_harvest_trust_columns`` helper + ``_HarvestedTrustColumns`` dataclass.

Cluster Z Closure C (2026-05-11): consolidates the 4-site Phase Y trust-column
harvester at ``hft_ops.cli`` (cli.py:557-609 pre-Closure-C) into one
``_harvest_trust_columns(captured_metrics) -> _HarvestedTrustColumns`` helper.

Closes #PY-155 — sister of #PY-115 silent-None policy violation at the
Phase Y composer producer→consumer boundary.

These tests lock the helper's three-state semantics for each of the 4
trust-column fields:

- **absent**: field stays ``None`` silently (valid: stage skipped/disabled).
- **present + valid**: field populated.
- **present + invalid format**: field stays ``None``, error appended to
  ``.harvest_errors``. Record still persists (observation-tier failure).
"""

from __future__ import annotations

from hft_ops.cli import _HarvestedTrustColumns, _harvest_trust_columns


# Canonical valid 64-hex SHA-256 fingerprints used across multiple tests.
_VALID_FP_A = "a" * 64
_VALID_FP_B = "b" * 64
_VALID_FP_C = "c" * 64


class TestEmptyAndAbsent:
    """All-absent harvest yields all-None fields + no errors."""

    def test_empty_captured_metrics(self):
        out = _harvest_trust_columns({})
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint is None
        assert out.model_config_hash is None
        assert out.signal_export_output_dir is None
        assert out.harvest_errors == []

    def test_all_keys_explicitly_none(self):
        """Explicit None == key absent (matches signal_metadata legacy)."""
        out = _harvest_trust_columns(
            {
                "feature_set_ref": None,
                "compatibility_fingerprint": None,
                "model_config_hash": None,
                "signal_export_output_dir": None,
            }
        )
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint is None
        assert out.model_config_hash is None
        assert out.signal_export_output_dir is None
        assert out.harvest_errors == []

    def test_unrelated_keys_ignored(self):
        """Helper consumes ONLY the 4 declared keys; ignores extras."""
        out = _harvest_trust_columns(
            {
                "experiment_id": "foo",
                "unrelated_metric": 0.42,
                "extras": [1, 2, 3],
            }
        )
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint is None
        assert out.harvest_errors == []


class TestValidHarvest:
    """All-valid harvest populates all 4 fields with zero errors."""

    def test_full_valid_harvest(self):
        out = _harvest_trust_columns(
            {
                "feature_set_ref": {
                    "name": "nvda_short_term_98_src98_v1",
                    "content_hash": _VALID_FP_A,
                },
                "compatibility_fingerprint": _VALID_FP_B,
                "model_config_hash": _VALID_FP_C,
                "signal_export_output_dir": "/data/exports/run_x/signals",
            }
        )
        assert out.feature_set_ref == {
            "name": "nvda_short_term_98_src98_v1",
            "content_hash": _VALID_FP_A,
        }
        assert out.compatibility_fingerprint == _VALID_FP_B
        assert out.model_config_hash == _VALID_FP_C
        assert out.signal_export_output_dir == "/data/exports/run_x/signals"
        assert out.harvest_errors == []

    def test_partial_valid_subset(self):
        """Only some fields present — those populated, rest None."""
        out = _harvest_trust_columns(
            {
                "compatibility_fingerprint": _VALID_FP_B,
                "signal_export_output_dir": "/tmp/signals",
            }
        )
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint == _VALID_FP_B
        assert out.model_config_hash is None
        assert out.signal_export_output_dir == "/tmp/signals"
        assert out.harvest_errors == []


class TestInvalidFormatHarvest:
    """Present-but-invalid input keeps field None + appends error."""

    def test_feature_set_ref_not_a_dict(self):
        out = _harvest_trust_columns({"feature_set_ref": "not_a_dict"})
        assert out.feature_set_ref is None
        assert len(out.harvest_errors) == 1
        assert "not a dict" in out.harvest_errors[0]
        assert "str" in out.harvest_errors[0]

    def test_feature_set_ref_non_string_name(self):
        out = _harvest_trust_columns(
            {"feature_set_ref": {"name": 123, "content_hash": _VALID_FP_A}}
        )
        assert out.feature_set_ref is None
        assert len(out.harvest_errors) == 1
        assert "non-string name" in out.harvest_errors[0]

    def test_feature_set_ref_non_string_content_hash(self):
        out = _harvest_trust_columns(
            {"feature_set_ref": {"name": "foo", "content_hash": 12345}}
        )
        assert out.feature_set_ref is None
        assert len(out.harvest_errors) == 1
        assert "content_hash" in out.harvest_errors[0]

    def test_feature_set_ref_missing_keys(self):
        """Dict missing name OR content_hash → invalid."""
        out = _harvest_trust_columns({"feature_set_ref": {"name": "foo"}})
        assert out.feature_set_ref is None
        assert len(out.harvest_errors) == 1
        assert "content_hash" in out.harvest_errors[0]

    def test_compatibility_fingerprint_wrong_format(self):
        out = _harvest_trust_columns(
            {"compatibility_fingerprint": "TOO_SHORT"}
        )
        assert out.compatibility_fingerprint is None
        assert len(out.harvest_errors) == 1
        assert "compatibility_fingerprint" in out.harvest_errors[0]
        assert "64-hex SHA-256" in out.harvest_errors[0]

    def test_compatibility_fingerprint_uppercase_rejected(self):
        """CONTENT_HASH_RE is case-sensitive lowercase-only per signal_manifest."""
        out = _harvest_trust_columns(
            {"compatibility_fingerprint": ("A" * 64)}
        )
        assert out.compatibility_fingerprint is None
        assert len(out.harvest_errors) == 1

    def test_compatibility_fingerprint_non_string(self):
        out = _harvest_trust_columns(
            {"compatibility_fingerprint": 12345}
        )
        assert out.compatibility_fingerprint is None
        assert "compatibility_fingerprint" in out.harvest_errors[0]
        assert "int" in out.harvest_errors[0]

    def test_model_config_hash_wrong_format(self):
        out = _harvest_trust_columns({"model_config_hash": "deadbeef"})
        assert out.model_config_hash is None
        assert "model_config_hash" in out.harvest_errors[0]
        assert "64-hex SHA-256" in out.harvest_errors[0]

    def test_signal_export_output_dir_empty(self):
        out = _harvest_trust_columns({"signal_export_output_dir": ""})
        assert out.signal_export_output_dir is None
        assert "signal_export_output_dir invalid" in out.harvest_errors[0]

    def test_signal_export_output_dir_non_string(self):
        """Path object NOT acceptable — caller MUST stringify upstream."""
        out = _harvest_trust_columns(
            {"signal_export_output_dir": ["not", "a", "str"]}
        )
        assert out.signal_export_output_dir is None
        assert "list" in out.harvest_errors[0]


class TestMultipleErrors:
    """Multiple invalid fields → all errors accumulate; valid fields populate."""

    def test_all_4_invalid(self):
        out = _harvest_trust_columns(
            {
                "feature_set_ref": "not_a_dict",
                "compatibility_fingerprint": "TOO_SHORT",
                "model_config_hash": 12345,
                "signal_export_output_dir": "",
            }
        )
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint is None
        assert out.model_config_hash is None
        assert out.signal_export_output_dir is None
        assert len(out.harvest_errors) == 4

    def test_mixed_valid_and_invalid(self):
        """One valid + three invalid → 3 errors, 1 populated field."""
        out = _harvest_trust_columns(
            {
                "feature_set_ref": {
                    "name": "x",
                    "content_hash": _VALID_FP_A,
                },
                "compatibility_fingerprint": "TOO_SHORT",
                "model_config_hash": 999,
                "signal_export_output_dir": None,
            }
        )
        assert out.feature_set_ref == {
            "name": "x",
            "content_hash": _VALID_FP_A,
        }
        assert out.compatibility_fingerprint is None
        assert out.model_config_hash is None
        assert out.signal_export_output_dir is None
        assert len(out.harvest_errors) == 2  # absent None doesn't add error


class TestNoRaiseInvariant:
    """Helper must NEVER raise — observation tier degrades gracefully."""

    def test_pathological_input_does_not_raise(self):
        """Even bizarre input shapes return a result with errors collected."""
        out = _harvest_trust_columns(
            {
                "feature_set_ref": {"name": object(), "content_hash": []},
                "compatibility_fingerprint": object(),
                "model_config_hash": [1, 2, 3],
                "signal_export_output_dir": {"nested": "dict"},
            }
        )
        # All fields None due to format violations.
        assert out.feature_set_ref is None
        assert out.compatibility_fingerprint is None
        assert out.model_config_hash is None
        assert out.signal_export_output_dir is None
        # All 4 errors accumulated.
        assert len(out.harvest_errors) == 4


class TestDataclassContract:
    """``_HarvestedTrustColumns`` schema lock — Phase Y composer reads these."""

    def test_default_construction(self):
        """Default-constructed dataclass has all-None + empty errors."""
        t = _HarvestedTrustColumns()
        assert t.feature_set_ref is None
        assert t.compatibility_fingerprint is None
        assert t.model_config_hash is None
        assert t.signal_export_output_dir is None
        assert t.harvest_errors == []

    def test_field_set_works(self):
        """Caller can populate fields directly (used in _record_experiment)."""
        t = _HarvestedTrustColumns()
        t.feature_set_ref = {"name": "x", "content_hash": _VALID_FP_A}
        t.compatibility_fingerprint = _VALID_FP_B
        assert t.feature_set_ref["name"] == "x"
        assert t.compatibility_fingerprint == _VALID_FP_B

    def test_harvest_errors_is_independent_list(self):
        """Each instance has its own errors list (no shared mutable default)."""
        a = _HarvestedTrustColumns()
        b = _HarvestedTrustColumns()
        a.harvest_errors.append("a-error")
        assert b.harvest_errors == []
