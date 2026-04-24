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


class TestHarvestCompatFpWarningEmission:
    """V.1.5 follow-up (2026-04-23) — SDR-2 observability lock.

    hft-rules §8 says "Never silently drop, clamp, or 'fix' data without
    recording diagnostics." Harvester must emit a `RuntimeWarning` when a
    fingerprint FIELD IS PRESENT but malformed — distinguishing producer
    drift (bad) from legacy-record (expected).

    Absent field → silent (no warning): legitimate pre-V.A.4 manifest, not
    a drift signal.

    Malformed field → warning: the trainer's exporter wrote something that
    doesn't pass validation. Surfaces the producer/consumer contract skew
    without failing the entire signal_export stage.
    """

    def test_absent_field_emits_no_warning(self, tmp_path: Path):
        """Legacy manifest without the field → no WARN (expected path)."""
        import warnings
        (tmp_path / "signal_metadata.json").write_text(
            json.dumps({"signal_type": "regression"})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _harvest_compatibility_fingerprint(tmp_path) is None
        # Filter to ONLY our module's warnings (ignore deprecation noise
        # from re-export shims)
        ours = [
            w for w in caught
            if "harvest_compatibility_fingerprint" in str(w.message)
        ]
        assert ours == [], (
            f"Absent field should not warn (legacy manifests are expected); "
            f"got {[str(w.message) for w in ours]}"
        )

    def test_malformed_uppercase_emits_warning(self, tmp_path: Path):
        """Field present but uppercase hex → RuntimeWarning citing the value."""
        import warnings
        (tmp_path / "signal_metadata.json").write_text(
            json.dumps({"compatibility_fingerprint": "A" * 64})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        ours = [
            w for w in caught
            if "harvest_compatibility_fingerprint" in str(w.message)
        ]
        assert len(ours) == 1, (
            f"Expected exactly 1 warning for malformed value; "
            f"got {len(ours)}: {[str(w.message) for w in ours]}"
        )
        msg = str(ours[0].message)
        assert "malformed" in msg.lower() or "expected 64" in msg.lower()
        assert "AAAA" in msg  # value surfaced in the warning
        assert ours[0].category is RuntimeWarning

    def test_malformed_non_string_emits_warning(self, tmp_path: Path):
        """Non-string (int, list, dict) value → RuntimeWarning."""
        import warnings
        (tmp_path / "signal_metadata.json").write_text(
            json.dumps({"compatibility_fingerprint": 42})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        ours = [
            w for w in caught
            if "harvest_compatibility_fingerprint" in str(w.message)
        ]
        assert len(ours) == 1

    def test_truncated_hex_emits_warning(self, tmp_path: Path):
        """63-char hex → RuntimeWarning (too-short is a drift signal)."""
        import warnings
        (tmp_path / "signal_metadata.json").write_text(
            json.dumps({"compatibility_fingerprint": "a" * 63})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        ours = [
            w for w in caught
            if "harvest_compatibility_fingerprint" in str(w.message)
        ]
        assert len(ours) == 1

    def test_valid_fingerprint_emits_no_warning(self, tmp_path: Path):
        """Happy path — 64-lowercase-hex → no warning, fingerprint returned."""
        import warnings
        valid = "abc123" + "d" * 58  # 64 chars, lowercase hex
        (tmp_path / "signal_metadata.json").write_text(
            json.dumps({"compatibility_fingerprint": valid})
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result == valid
        ours = [
            w for w in caught
            if "harvest_compatibility_fingerprint" in str(w.message)
        ]
        assert ours == []


class TestHarvestCompatFpPhaseACutoff:
    """Phase A (2026-04-23) follow-up observability — post-Phase-A manifests
    without a fingerprint trigger a WARN log (operator-facing signal of
    producer-path regressions). Pre-cutoff manifests remain silent (legacy
    behavior preserved).

    Complements the V.1.5 malformed-field WARN path (
    :class:`TestHarvestCompatFpWarningEmission`): that class locks "field
    present but wrong", this class locks "field absent post-cutoff".

    See ``FINGERPRINT_REQUIRED_AFTER_ISO`` in signal_export.py for the
    cutoff constant definition.
    """

    def test_pre_cutoff_manifest_without_fingerprint_emits_no_warn(
        self, tmp_path: Path, caplog,
    ):
        """Manifest with ``exported_at`` pre-cutoff OR missing → silent (legacy).

        Before the Phase A cutoff date, absent fingerprint is EXPECTED
        (Phase II shipped 2026-04-20, Phase A fix shipped 2026-04-23). We
        MUST NOT WARN on legacy data — that would spam logs on every
        historical record.
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "2026-03-19T12:00:00+00:00",  # pre-cutoff
            # No ``compatibility_fingerprint`` key.
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        # No WARN about post-Phase-A — legacy manifest is silent.
        phase_a_warnings = [
            rec for rec in caplog.records
            if "post-Phase-A" in rec.getMessage()
        ]
        assert phase_a_warnings == [], (
            "Pre-cutoff manifest should NOT emit a post-Phase-A WARN. "
            "Absent fingerprint on legacy records is expected — any log spam "
            "here would fire on every historical ledger record."
        )

    def test_post_cutoff_manifest_without_fingerprint_emits_warn(
        self, tmp_path: Path, caplog,
    ):
        """Post-cutoff manifest missing fingerprint → WARN log with diagnostic.

        This is the operator-facing signal that a trainer venv or producer
        path has regressed — the trainer shipped a post-Phase-A signal
        directory but the fingerprint was dropped. Harvester still returns
        None gracefully; the WARN surfaces the regression without blocking
        ledger ingestion.
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "2026-05-01T12:00:00+00:00",  # post-cutoff
            # No ``compatibility_fingerprint`` key.
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        # WARN should cite both the manifest path AND the exported_at stamp
        # so operators can trace back to the producer invocation.
        phase_a_warnings = [
            rec for rec in caplog.records
            if "post-Phase-A manifest" in rec.getMessage()
        ]
        assert len(phase_a_warnings) == 1, (
            f"Expected exactly 1 post-Phase-A WARN for missing fingerprint "
            f"on post-cutoff manifest; got {len(phase_a_warnings)}"
        )
        msg = phase_a_warnings[0].getMessage()
        assert "2026-05-01" in msg, "WARN should cite the exported_at stamp"
        assert "signal_metadata.json" in msg, "WARN should cite the manifest path"


# =============================================================================
# Phase A.5.2 (2026-04-24): timezone-aware cutoff comparison tests.
#
# Locks the BUG FIX for the pre-A.5.2 silent-wrong-result lexicographic
# ``exported_at >= cutoff`` pattern. See commit A.5.2 (hft-ops) +
# hft_contracts.timestamp_utils module docstring for the full bug-class
# analysis.
#
# These tests collectively prove:
#   1. The CORE BUG is fixed (non-UTC offset crossing cutoff midnight now
#      correctly identifies as post-cutoff and triggers WARN). Pre-A.5.2
#      would have silently dropped this.
#   2. Naive timestamps (no offset) are interpreted as UTC per hft-rules §3
#      canonical convention; inclusive `>=` semantics at exact boundary.
#   3. Malformed ISO-8601 input does NOT crash the harvester; it emits a
#      diagnostic WARN (hft-rules §8) + treats as pre-cutoff conservatively.
#   4. The Z-suffix format works end-to-end (complements
#      ``test_post_cutoff_manifest_without_fingerprint_emits_warn`` above
#      which exercises ``+00:00`` notation).
# =============================================================================


class TestHarvestCompatFpTimezoneAware:
    """A.5.2 UTC-aware cutoff comparison — locks the pre-A.5.2 bug-class fix.

    Pre-A.5.2: ``exported_at >= FINGERPRINT_REQUIRED_AFTER_ISO`` was a
    lexicographic string compare, silently returning pre-cutoff for any
    non-UTC offset that crosses midnight (e.g. ``"2026-04-22T23:59:00-05:00"``
    is strictly post-cutoff in UTC — ``2026-04-23T04:59:00+00:00`` — but
    lex-compares as pre-cutoff). This silently suppressed the operator-
    facing producer-regression WARN for every non-UTC-offset manifest.

    Post-A.5.2: harvester routes through
    ``hft_contracts.timestamp_utils.is_after_cutoff`` which normalizes both
    sides to timezone-aware UTC datetimes before comparison.
    """

    def test_non_utc_offset_post_cutoff_triggers_warn(
        self, tmp_path: Path, caplog,
    ):
        """THE A.5.2 BUG FIX. Pre-A.5.2 would NOT have fired the WARN.

        ``2026-04-22T23:59:00-05:00`` is ``2026-04-23T04:59:00+00:00`` in UTC
        — strictly after the ``2026-04-23`` cutoff. The pre-A.5.2 lex compare
        returned False (the raw string `"2026-04-22T..."` lex-compares as
        pre-cutoff vs `"2026-04-23"`). Under A.5.2 is_after_cutoff this
        correctly returns True → post-Phase-A WARN fires.
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "2026-04-22T23:59:00-05:00",  # UTC: 2026-04-23T04:59Z
            # No ``compatibility_fingerprint`` key.
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        phase_a_warnings = [
            rec for rec in caplog.records
            if "post-Phase-A manifest" in rec.getMessage()
        ]
        assert len(phase_a_warnings) == 1, (
            f"A.5.2 BUG FIX: non-UTC offset timestamp that crosses the "
            f"cutoff midnight MUST trigger the post-Phase-A WARN "
            f"(exported_at={meta['exported_at']!r} is UTC=2026-04-23T04:59Z, "
            f"strictly post-cutoff). Pre-A.5.2 lex comparison missed this. "
            f"Got {len(phase_a_warnings)} WARNs — if 0, the harvester has "
            f"regressed to the pre-A.5.2 behavior."
        )
        # Explicitly cite the UTC conversion in the assertion so a future
        # regression trace can reconstruct the bug class.
        msg = phase_a_warnings[0].getMessage()
        assert "-05:00" in msg, "WARN should cite the original exported_at stamp"

    def test_naive_exact_cutoff_triggers_warn(
        self, tmp_path: Path, caplog,
    ):
        """Naive timestamp at exact cutoff → WARN (inclusive `>=` semantics).

        hft-rules §3 canonical convention: naive timestamps are UTC. A naive
        ``"2026-04-23T00:00:00"`` equals the cutoff exactly in UTC. The
        ``is_after_cutoff`` helper uses `>=`, so the boundary is inclusive
        → post-Phase-A WARN fires.
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "2026-04-23T00:00:00",  # Naive, exact cutoff
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        phase_a_warnings = [
            rec for rec in caplog.records
            if "post-Phase-A manifest" in rec.getMessage()
        ]
        assert len(phase_a_warnings) == 1, (
            f"Naive exact-cutoff timestamp must trigger WARN via inclusive "
            f"`>=` semantics. Got {len(phase_a_warnings)}."
        )

    def test_malformed_exported_at_does_not_crash(
        self, tmp_path: Path, caplog,
    ):
        """Malformed ISO-8601 → diagnostic WARN + treat as pre-cutoff.

        Harvester MUST NOT crash the signal-export stage on malformed
        manifest data. The `is_after_cutoff` helper raises ValueError on
        malformed input; the harvester catches it, emits a DIAGNOSTIC WARN
        per hft-rules §8 ("never silently drop / clamp / 'fix' data without
        recording diagnostics"), and treats as pre-cutoff (no post-Phase-A
        WARN, since we cannot determine which side of the cutoff).
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "not-a-real-timestamp",  # Malformed ISO-8601
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        # Primary invariant: no crash + None return.
        assert result is None
        # hft-rules §8: malformed data IS diagnostic-worthy → WARN.
        malformed_warns = [
            rec for rec in caplog.records
            if "malformed exported_at" in rec.getMessage()
        ]
        assert len(malformed_warns) == 1, (
            f"Malformed exported_at should emit a diagnostic WARN per "
            f"hft-rules §8; got {len(malformed_warns)} WARNs."
        )
        # NO post-Phase-A WARN (we cannot determine cutoff side safely).
        phase_a_warns = [
            rec for rec in caplog.records
            if "post-Phase-A manifest" in rec.getMessage()
        ]
        assert phase_a_warns == [], (
            f"Malformed timestamp MUST NOT trigger the post-Phase-A WARN "
            f"(conservative: we can't say which side of cutoff). "
            f"Got: {[w.getMessage() for w in phase_a_warns]}"
        )

    def test_z_suffix_post_cutoff_triggers_warn(
        self, tmp_path: Path, caplog,
    ):
        """Z-suffix post-cutoff → WARN.

        Complements ``test_post_cutoff_manifest_without_fingerprint_emits_warn``
        (which uses `+00:00` notation). A.5.2's ``parse_iso8601_utc``
        normalizes `Z` → `+00:00` defensively before ``datetime.fromisoformat``
        (Python < 3.11 compat). If the normalization is ever broken, this
        test fires.
        """
        import logging as _logging
        meta = {
            "signal_type": "regression",
            "exported_at": "2026-05-01T12:00:00Z",  # Z suffix, post-cutoff
        }
        (tmp_path / "signal_metadata.json").write_text(json.dumps(meta))
        with caplog.at_level(_logging.WARNING, logger="hft_ops.stages.signal_export"):
            result = _harvest_compatibility_fingerprint(tmp_path)
        assert result is None
        phase_a_warnings = [
            rec for rec in caplog.records
            if "post-Phase-A manifest" in rec.getMessage()
        ]
        assert len(phase_a_warnings) == 1, (
            f"Z-suffix post-cutoff timestamp must trigger WARN (via "
            f"is_after_cutoff Z-normalization path). Got "
            f"{len(phase_a_warnings)}."
        )
