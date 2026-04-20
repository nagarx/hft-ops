"""
Experiment ledger: append-only JSON-backed storage.

The ledger stores experiment records in a flat directory structure:
    ledger/
        index.json          -- lightweight index for fast queries
        records/
            {experiment_id}.json  -- full record per experiment

The index is rebuilt from records on load, so it is always consistent.
Records are immutable after creation (append-only design).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hft_contracts import INDEX_SCHEMA_VERSION
from hft_contracts.atomic_io import atomic_write_json
from hft_ops.ledger.experiment_record import ExperimentRecord

_LOGGER = logging.getLogger(__name__)


class LedgerError(ValueError):
    """Root of the hft-ops ledger exception hierarchy (Phase 8B).

    ``ValueError`` base is canonical for data-validation conditions (on-disk
    state does not match code-side contract). Extension sub-phases add:
    Phase 8A.1 will introduce GPU-acquire / scheduler errors; Phase 8C will
    introduce feedback-artifact errors. All should inherit from this root so
    consumers can dispatch on ``except LedgerError:`` without string-matching
    specific subclasses.
    """


class StaleLedgerIndexError(LedgerError):
    """Raised by ``ExperimentLedger._load_index`` (and ``check_duplicate`` —
    Phase 8B BUG-1 fix) under strict-index mode when the on-disk
    ``index.json`` schema version differs from the code-side
    ``hft_contracts.INDEX_SCHEMA_VERSION`` (MAJOR.MINOR comparison) or the
    file is malformed / in legacy bare-list format.

    The non-strict mode logs a WARNING and auto-rebuilds (the default —
    matches developer-inner-loop ergonomics). Strict mode is intended for CI
    environments: the rebuild is a cue that a developer forgot to commit
    the refreshed ``index.json`` or to bump ``INDEX_SCHEMA_VERSION`` when
    extending the projection, and CI should FAIL rather than silently
    re-generate. ``--strict-index`` on the CLI or ``CI=true``/``CI=1`` (or
    ``HFT_OPS_STRICT_INDEX`` truthy) in the environment enables strict mode.
    """


def _parse_truthy_env(value: str) -> bool:
    """Parse an env-var string as a boolean truthy value, case-insensitive.

    Accepts: ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive).
    Rejects everything else (including empty string, ``"0"``, ``"false"``,
    ``"no"``, ``"off"``, arbitrary text).

    Phase 8B MUST-FIX (Agent 2 P1): `HFT_OPS_STRICT_INDEX` and `CI` now use
    the same truthy-parsing rule so user expectations are symmetric —
    setting either to ``"true"`` enables strict mode, setting either to
    ``"false"`` / ``"0"`` / unset does not.
    """
    return value.strip().lower() in ("1", "true", "yes", "on")


def _detect_strict_index_from_env() -> bool:
    """Auto-detect whether strict-index mode should be enabled based on
    environment variables. Used when ``ExperimentLedger.__init__`` is
    called without an explicit ``strict_index`` argument.

    Enabled when either:
      - ``HFT_OPS_STRICT_INDEX`` is truthy (``"1"``/``"true"``/``"yes"``/
        ``"on"``, case-insensitive) — explicit developer opt-in, OR
      - ``CI`` is truthy — standard CI-runner convention (GitHub Actions,
        GitLab CI, CircleCI, Buildkite all export ``CI=true``).
    """
    if _parse_truthy_env(os.environ.get("HFT_OPS_STRICT_INDEX", "")):
        return True
    return _parse_truthy_env(os.environ.get("CI", ""))

# Regex-based MAJOR.MINOR.PATCH parse. Matches the format invariant tested in
# ``hft-contracts/tests/test_experiment_record.py::TestPackageSurface``. Using
# stdlib ``re`` rather than ``packaging.version.Version`` keeps this module
# dep-light; the ``packaging`` library is a valid future upgrade if richer
# SemVer semantics (pre-release, local-version) are ever needed.
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.\d+$")


class ExperimentLedger:
    """Append-only experiment record storage.

    Args:
        ledger_dir: Root directory for ledger storage. Created if missing.
    """

    def __init__(
        self, ledger_dir: Path, strict_index: Optional[bool] = None
    ) -> None:
        """
        Args:
            ledger_dir: Root directory for ledger storage. Created if missing.
            strict_index: Phase 8B — when True, any detected schema mismatch
                (version drift, legacy bare-list, malformed JSON) raises
                :class:`StaleLedgerIndexError` instead of auto-rebuilding.
                When None (default), auto-detects from env vars
                ``HFT_OPS_STRICT_INDEX=1`` or ``CI=true`` / ``CI=1``. CLI
                callers pass ``strict_index=True`` explicitly via the
                ``--strict-index`` top-level flag.
        """
        self._ledger_dir = Path(ledger_dir)
        self._records_dir = self._ledger_dir / "records"
        self._index_path = self._ledger_dir / "index.json"
        self._index: List[Dict[str, Any]] = []
        # Phase 8B: forensic field surfaced in index.json envelope header.
        # Set by ``_rebuild_index`` callers so operators can tell WHY the
        # index was last regenerated (initial empty dir, legacy migration,
        # version mismatch, malformed JSON, or a manual ``rebuild-index``).
        self._last_rebuild_source: str = "initial"
        # Phase 8B: strict mode elevates auto-rebuild to hard error.
        # Intended for CI — a failed run forces the developer to commit a
        # refreshed ``index.json`` or bump ``INDEX_SCHEMA_VERSION`` rather
        # than relying on silent auto-rebuild at load time.
        self._strict_index: bool = (
            strict_index if strict_index is not None else _detect_strict_index_from_env()
        )

        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        self._records_dir.mkdir(parents=True, exist_ok=True)

        self._load_index()

    def _load_index(self) -> None:
        """Load or rebuild the index from disk.

        Phase 8B (2026-04-20): reads the envelope format
        ``{"schema": {"version": ..., ...}, "entries": [...]}`` introduced
        by this phase. Handles four on-disk shapes gracefully:

        1. **Envelope, matching MAJOR.MINOR** → load ``entries`` directly;
           O(1) fast path.
        2. **Envelope, MAJOR.MINOR mismatch** → log WARNING + auto-rebuild
           from ``records/*.json``. Handles whitelist drift (the exact
           silent-omission class this phase eliminates).
        3. **Legacy bare-list** (pre-Phase-8B ``index.json``) → rebuild
           once (converts file to envelope on next ``_save_index``).
           Back-compat for the existing 34+ records at HEAD.
        4. **Malformed JSON** (truncated/partial write / NaN root / non-
           list-non-dict root) → rebuild. Preserves prior fallback
           behavior while making the rebuild reason diagnostic.
        """
        if not self._index_path.exists():
            # First-time init on fresh ledger dir — not a "rebuild"; just
            # populate an empty index and persist the envelope. This is
            # NEVER a strict-mode failure (fresh ledger is the expected
            # starting state, not a drift condition).
            self._last_rebuild_source = "initial"
            self._rebuild_index(source="initial", warn=False)
            return

        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Malformed or unreadable — rebuild from records.
            self._handle_stale_index_or_rebuild(
                source="auto_malformed",
                detail="index.json is malformed or unreadable",
            )
            return

        # Envelope format (post-Phase-8B): dict with "entries" key.
        if isinstance(data, dict) and "entries" in data and isinstance(
            data["entries"], list
        ):
            on_disk_version = data.get("schema", {}).get("version")
            if isinstance(on_disk_version, str) and not _index_schema_needs_rebuild(
                on_disk_version
            ):
                # Fast path — matching MAJOR.MINOR, no rebuild needed.
                self._index = data["entries"]
                # Preserve the prior rebuild source for forensics (don't
                # overwrite with "initial" just because load succeeded).
                self._last_rebuild_source = data.get("schema", {}).get(
                    "last_rebuild_source", self._last_rebuild_source
                )
                return
            # Version mismatch (or missing version field) — rebuild or
            # fail-fast under strict mode.
            self._handle_stale_index_or_rebuild(
                source=f"auto_version_mismatch_{on_disk_version}",
                detail=(
                    f"on-disk schema version {on_disk_version!r} differs "
                    f"from code INDEX_SCHEMA_VERSION={INDEX_SCHEMA_VERSION!r}"
                ),
            )
            return

        # Legacy bare-list format (pre-Phase-8B) — migrate by rebuild.
        if isinstance(data, list):
            self._handle_stale_index_or_rebuild(
                source="auto_legacy_bare_list",
                detail="on-disk index.json is in pre-Phase-8B bare-list format",
            )
            return

        # Unknown root shape (null, string, number, dict without "entries",
        # dict with non-list "entries", etc.) — rebuild.
        self._handle_stale_index_or_rebuild(
            source="auto_unknown_root",
            detail=f"on-disk index.json root has unexpected shape: {type(data).__name__}",
        )

    def _handle_stale_index_or_rebuild(self, source: str, detail: str) -> None:
        """Phase 8B: dispatch between auto-rebuild and strict-mode raise.

        Called from the four non-fast-path branches of ``_load_index``.
        Under ``strict_index=True`` raises :class:`StaleLedgerIndexError`
        with a clear actionable message; otherwise falls through to
        ``_rebuild_index`` with WARN.
        """
        if self._strict_index:
            raise StaleLedgerIndexError(
                f"ledger index is stale (reason: {source}). {detail}. "
                f"Under strict mode (--strict-index or CI=true), this does NOT "
                f"auto-rebuild — you must explicitly run `hft-ops ledger "
                f"rebuild-index` locally and commit the refreshed index.json "
                f"(and, if index_entry() was extended, bump INDEX_SCHEMA_VERSION "
                f"in hft-contracts)."
            )
        self._rebuild_index(source=source, warn=True)

    def _rebuild_index(
        self, source: str = "manual", warn: bool = False
    ) -> None:
        """Rebuild index from individual record files.

        Args:
            source: Diagnostic string recorded in the envelope's
                ``last_rebuild_source`` field for forensic auditing.
                Conventional values: ``"initial"``, ``"manual"``,
                ``"auto_malformed"``, ``"auto_legacy_bare_list"``,
                ``"auto_version_mismatch_<version>"``, ``"auto_unknown_root"``.
            warn: If True, emit a WARNING log. Reserved for auto-rebuild
                paths (the silent-omission-class fixes) — explicit
                ``rebuild-index`` CLI invocations pass ``warn=False`` since
                the operator intentionally triggered it.
        """
        if warn:
            _LOGGER.warning(
                "stale ledger index detected: %s (code INDEX_SCHEMA_VERSION=%s); "
                "auto-rebuilding from records/*.json",
                source,
                INDEX_SCHEMA_VERSION,
            )
        self._last_rebuild_source = source
        self._index = []
        for record_file in sorted(self._records_dir.glob("*.json")):
            try:
                record = ExperimentRecord.load(record_file)
                self._index.append(record.index_entry())
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        self._save_index()

    def _save_index(self) -> None:
        """Persist the index to disk atomically, in envelope format.

        Phase 8B (2026-04-20): writes the envelope
        ``{"schema": {"version": INDEX_SCHEMA_VERSION, "written_at": ...,
        "last_rebuild_source": ...}, "entries": [...]}``. The envelope
        carries a schema version that ``_load_index`` compares against the
        code-side ``hft_contracts.INDEX_SCHEMA_VERSION`` — MAJOR.MINOR
        mismatch triggers an automatic rebuild that eliminates the silent-
        omission class when ``ExperimentRecord.index_entry()`` whitelist
        is extended.

        Phase 7 Stage 7.4 Round 5 (2026-04-20): delegates to the canonical
        ``hft_contracts.atomic_io.atomic_write_json`` — unified with
        ``ExperimentRecord.save`` and ``hft_ops.feature_sets.writer``.
        Canonical convention (``sort_keys=True`` + trailing newline)
        ensures index.json is byte-stable across runs for diff tooling.

        Atomicity closes the transient-bad-state window on SIGKILL /
        ENOSPC / power failure: prior non-atomic ``open(w) + json.dump``
        could leave index.json truncated, forcing a full
        ``_rebuild_index`` on the next load — still handled gracefully via
        the ``auto_malformed`` path in ``_load_index``, but O(N) cost on
        every startup until a clean write lands.
        """
        envelope = {
            "schema": {
                "version": INDEX_SCHEMA_VERSION,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "last_rebuild_source": self._last_rebuild_source,
            },
            "entries": self._index,
        }
        atomic_write_json(self._index_path, envelope)

    def register(self, record: ExperimentRecord) -> str:
        """Register a new experiment record.

        Args:
            record: The experiment record to store.

        Returns:
            The experiment_id of the stored record.

        Raises:
            ValueError: If a record with the same experiment_id already exists.
        """
        if not record.experiment_id:
            raise ValueError("ExperimentRecord must have an experiment_id")

        existing_ids = {entry["experiment_id"] for entry in self._index}
        if record.experiment_id in existing_ids:
            raise ValueError(
                f"Experiment already registered: {record.experiment_id}"
            )

        record_path = self._records_dir / f"{record.experiment_id}.json"
        record.save(record_path)

        self._index.append(record.index_entry())
        self._save_index()

        return record.experiment_id

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Load a full experiment record by ID.

        Returns None if not found.
        """
        record_path = self._records_dir / f"{experiment_id}.json"
        if not record_path.exists():
            return None
        return ExperimentRecord.load(record_path)

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all index entries (lightweight metadata)."""
        return list(self._index)

    def list_ids(self) -> List[str]:
        """Return all experiment IDs."""
        return [entry["experiment_id"] for entry in self._index]

    def count(self) -> int:
        """Return the number of registered experiments."""
        return len(self._index)

    def filter(
        self,
        *,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        labeling_strategy: Optional[str] = None,
        status: Optional[str] = None,
        min_f1: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        sweep_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter index entries by criteria.

        All criteria are AND-combined: an entry must match ALL specified filters.
        """
        results: List[Dict[str, Any]] = []

        for entry in self._index:
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not all(t in entry_tags for t in tags):
                    continue

            if model_type and entry.get("model_type") != model_type:
                continue

            if labeling_strategy and entry.get("labeling_strategy") != labeling_strategy:
                continue

            if status and entry.get("status") != status:
                continue

            train_metrics = entry.get("training_metrics", {})
            if min_f1 is not None:
                f1 = train_metrics.get("macro_f1", 0.0)
                if f1 < min_f1:
                    continue

            if min_accuracy is not None:
                acc = train_metrics.get("accuracy", 0.0)
                if acc < min_accuracy:
                    continue

            if created_after and entry.get("created_at", "") < created_after:
                continue

            if created_before and entry.get("created_at", "") > created_before:
                continue

            if sweep_id and entry.get("sweep_id", "") != sweep_id:
                continue

            results.append(entry)

        return results

    def find_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Find an index entry by fingerprint (for dedup)."""
        for entry in self._index:
            if entry.get("fingerprint") == fingerprint:
                return entry
        return None

    def update_notes(self, experiment_id: str, notes: str) -> bool:
        """Update the notes field of an existing record.

        This is the only mutable operation on records, used for
        post-experiment observations.

        Returns True if updated, False if not found.
        """
        record = self.get(experiment_id)
        if record is None:
            return False

        record.notes = notes
        record_path = self._records_dir / f"{experiment_id}.json"
        record.save(record_path)

        return True

    def summary(self) -> Dict[str, Any]:
        """Return aggregate statistics about the ledger."""
        total = len(self._index)
        statuses = {}
        model_types = {}
        tag_counts: Dict[str, int] = {}

        for entry in self._index:
            s = entry.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1

            mt = entry.get("model_type", "unknown")
            model_types[mt] = model_types.get(mt, 0) + 1

            for tag in entry.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_experiments": total,
            "by_status": statuses,
            "by_model_type": model_types,
            "top_tags": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }


def _index_schema_needs_rebuild(on_disk_version: str) -> bool:
    """Phase 8B: compare an on-disk index envelope version against the
    code-side ``hft_contracts.INDEX_SCHEMA_VERSION``.

    Returns ``True`` when a rebuild is required:

    - Either version is unparseable (missing, wrong shape, None) — fail
      safe by rebuilding.
    - MAJOR differs — breaking schema change, must rebuild.
    - MINOR differs — additive whitelist change, rebuild needed to pick
      up newly-projected keys (or to drop a removed key on downgrade).
    - MAJOR.MINOR match + PATCH differs — NO rebuild. PATCH is reserved
      for docstring / non-functional touches that don't affect projection.

    Exposed at module level (not as a method) so tests can exercise the
    comparison logic directly without constructing a full ``ExperimentLedger``.
    """
    disk_match = _SEMVER_RE.match(on_disk_version) if on_disk_version else None
    code_match = _SEMVER_RE.match(INDEX_SCHEMA_VERSION)
    if disk_match is None or code_match is None:
        return True  # Unparseable → force rebuild (safe default).
    disk_major_minor = (int(disk_match.group(1)), int(disk_match.group(2)))
    code_major_minor = (int(code_match.group(1)), int(code_match.group(2)))
    return disk_major_minor != code_major_minor
