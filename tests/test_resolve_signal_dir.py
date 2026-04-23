"""Phase V.1.5 follow-up (2026-04-23): real-fs `_resolve_signal_dir` tests.

Closes the Agent 1 3rd-round audit finding that `_resolve_signal_dir` has
ZERO real-filesystem tests — all 8 existing call sites in
`test_statistical_compare.py` mock it via `unittest.mock.patch(...)`. A
regression in the preferred-path resolution (V.1 L1.2) or the
`.is_dir()` guard (V.1.5 A1) would silently slip past unit coverage.

Also closes SDR-6 from the Agent 4 3rd-round data-flow audit: when
`stored_sig_dir` is present but not a valid directory (file, symlink
race, post-run deletion), the resolver silently falls through to manifest
re-parse. V.1.5 adds a RuntimeWarning on this path per hft-rules §8
"Never silently drop ... without recording diagnostics."

Test layers covered here:

* COMPONENT: the resolver function with a stub ledger/record/paths —
  exercises resolver branches in isolation.
* INTEGRATION: real Path-based filesystem with real stored paths — covers
  the `candidate.is_dir()` call against real OS filesystem behavior
  (symlinks, regular files, missing paths).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from hft_ops.ledger.statistical_compare import _resolve_signal_dir


# =============================================================================
# Stubs (minimal shapes for the resolver function signature)
# =============================================================================


@dataclass
class _StubRecord:
    """Minimal ExperimentRecord-shaped object for resolver testing."""
    signal_export_output_dir: Optional[str] = None
    manifest_path: Optional[str] = None


class _StubLedger:
    """Minimal ExperimentLedger stub supporting `.get(exp_id)`."""

    def __init__(self, records: dict):
        self._records = records

    def get(self, exp_id: str):
        return self._records.get(exp_id)


@dataclass
class _StubPaths:
    """Minimal PipelinePaths-shaped object for resolver testing. Only
    `.resolve()` is used via manifest-fallback path; tests that exercise
    the preferred-path branch never call it."""

    def resolve(self, p: str) -> Path:
        return Path(p)


# =============================================================================
# Preferred-path branch (V.1 L1.2) — stored signal_export_output_dir
# =============================================================================


class TestPreferredPath:
    """V.1 L1.2 introduced `signal_export_output_dir` for manifest-move
    resilience. When the stored path IS a directory, use it directly — no
    manifest re-parse, no variable substitution."""

    def test_returns_stored_path_when_is_dir(self, tmp_path: Path):
        """Stored path exists AND is_dir() → resolver returns it unchanged."""
        sig_dir = tmp_path / "signals"
        sig_dir.mkdir()
        record = _StubRecord(signal_export_output_dir=str(sig_dir))
        ledger = _StubLedger({"exp_1": record})
        result = _resolve_signal_dir(
            {"experiment_id": "exp_1"}, ledger, _StubPaths(),
        )
        assert result == sig_dir

    def test_stored_path_absolute_is_preserved(self, tmp_path: Path):
        """Absolute path stored → absolute path returned (no truncation)."""
        sig_dir = tmp_path / "signals_abs"
        sig_dir.mkdir()
        assert sig_dir.is_absolute()
        record = _StubRecord(signal_export_output_dir=str(sig_dir))
        ledger = _StubLedger({"exp_abs": record})
        result = _resolve_signal_dir(
            {"experiment_id": "exp_abs"}, ledger, _StubPaths(),
        )
        assert result.is_absolute()
        assert result == sig_dir

    def test_symlink_to_dir_resolves_via_is_dir(self, tmp_path: Path):
        """Symlink that resolves to a directory → treated as dir (is_dir()
        follows symlinks by default in pathlib)."""
        real_dir = tmp_path / "real_signals"
        real_dir.mkdir()
        link = tmp_path / "signals_link"
        link.symlink_to(real_dir, target_is_directory=True)
        record = _StubRecord(signal_export_output_dir=str(link))
        ledger = _StubLedger({"exp_link": record})
        result = _resolve_signal_dir(
            {"experiment_id": "exp_link"}, ledger, _StubPaths(),
        )
        # Resolver returns the link path unchanged — is_dir() passed.
        assert result == link


# =============================================================================
# SDR-6 fallback-with-WARN — V.1.5 A1 + V.1.5 SDR-2 observability
# =============================================================================


class TestFallbackWarn:
    """SDR-6: when stored_sig_dir is set but NOT a valid directory, the
    resolver falls through to manifest re-parse AND emits a WARN —
    observability per hft-rules §8."""

    def test_stored_path_is_file_not_dir_warns_and_falls_through(self, tmp_path: Path):
        """V.1.5 A1 `.is_dir()` guard catches file-not-dir case; V.1.5 SDR-6
        adds WARN. Without a manifest fallback, this raises, but the WARN
        MUST appear before the raise."""
        # Create a FILE (not a dir) at the stored path
        fake_file = tmp_path / "oops_this_is_a_file.txt"
        fake_file.write_text("not a signal dir")
        record = _StubRecord(
            signal_export_output_dir=str(fake_file),
            manifest_path=None,  # no fallback → will raise after warn
        )
        ledger = _StubLedger({"exp_file": record})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="no resolvable signal directory"):
                _resolve_signal_dir(
                    {"experiment_id": "exp_file"}, ledger, _StubPaths(),
                )
        # Exactly one resolver WARN observed; mentions "file, not a directory"
        ours = [w for w in caught if "resolve_signal_dir" in str(w.message)]
        assert len(ours) == 1, (
            f"Expected 1 resolver WARN on file-not-dir fall-through; "
            f"got {len(ours)}: {[str(w.message) for w in ours]}"
        )
        msg = str(ours[0].message)
        assert "file, not a directory" in msg
        assert str(fake_file) in msg
        assert ours[0].category is RuntimeWarning

    def test_stored_path_missing_warns_and_falls_through(self, tmp_path: Path):
        """Stored path points to non-existent location (dir was deleted /
        moved / not yet created) — WARN with 'no longer exists'."""
        missing = tmp_path / "moved_to_cold_storage"
        # Intentionally NOT created
        record = _StubRecord(
            signal_export_output_dir=str(missing),
            manifest_path=None,
        )
        ledger = _StubLedger({"exp_missing": record})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="no resolvable signal directory"):
                _resolve_signal_dir(
                    {"experiment_id": "exp_missing"}, ledger, _StubPaths(),
                )
        ours = [w for w in caught if "resolve_signal_dir" in str(w.message)]
        assert len(ours) == 1
        assert "no longer exists" in str(ours[0].message)


# =============================================================================
# Legacy records (pre-V.1.L1.2) — no signal_export_output_dir → silent fallback
# =============================================================================


class TestLegacyRecordNoStoredPath:
    """Pre-V.1.L1.2 records have `signal_export_output_dir=None`. Resolver
    must fall through to manifest re-parse WITHOUT a WARN (expected path
    for legacy data, not a drift signal)."""

    def test_no_stored_path_no_warning(self, tmp_path: Path):
        """signal_export_output_dir=None (legacy) → silent fallback,
        no WARN. Manifest also missing → raises, but that's a separate
        check; the test here is 'no WARN in the preferred-path branch'."""
        record = _StubRecord(
            signal_export_output_dir=None,   # legacy
            manifest_path=None,              # also absent
        )
        ledger = _StubLedger({"exp_legacy": record})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="no resolvable signal directory"):
                _resolve_signal_dir(
                    {"experiment_id": "exp_legacy"}, ledger, _StubPaths(),
                )
        ours = [w for w in caught if "resolve_signal_dir" in str(w.message)]
        assert ours == [], (
            f"Legacy records with signal_export_output_dir=None should NOT "
            f"emit a WARN in the preferred-path branch (it's the expected path); "
            f"got {[str(w.message) for w in ours]}"
        )

    def test_empty_string_stored_path_no_warning(self, tmp_path: Path):
        """Falsy string (e.g., "" or None) → treated as absent, no WARN."""
        record = _StubRecord(
            signal_export_output_dir="",
            manifest_path=None,
        )
        ledger = _StubLedger({"exp_empty": record})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError):
                _resolve_signal_dir(
                    {"experiment_id": "exp_empty"}, ledger, _StubPaths(),
                )
        ours = [w for w in caught if "resolve_signal_dir" in str(w.message)]
        assert ours == []


# =============================================================================
# Missing-record guard
# =============================================================================


class TestMissingRecord:
    """Index entry points to a record that doesn't exist on disk —
    resolver raises a clear error pointing to `ledger rebuild-index`."""

    def test_record_not_found_raises(self, tmp_path: Path):
        ledger = _StubLedger({})  # no records
        with pytest.raises(ValueError, match="stale index"):
            _resolve_signal_dir(
                {"experiment_id": "ghost_exp"}, ledger, _StubPaths(),
            )
