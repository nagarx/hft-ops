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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hft_contracts._atomic_io import atomic_write_json
from hft_ops.ledger.experiment_record import ExperimentRecord


class ExperimentLedger:
    """Append-only experiment record storage.

    Args:
        ledger_dir: Root directory for ledger storage. Created if missing.
    """

    def __init__(self, ledger_dir: Path) -> None:
        self._ledger_dir = Path(ledger_dir)
        self._records_dir = self._ledger_dir / "records"
        self._index_path = self._ledger_dir / "index.json"
        self._index: List[Dict[str, Any]] = []

        self._ledger_dir.mkdir(parents=True, exist_ok=True)
        self._records_dir.mkdir(parents=True, exist_ok=True)

        self._load_index()

    def _load_index(self) -> None:
        """Load or rebuild the index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
                return
            except (json.JSONDecodeError, OSError):
                pass

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from individual record files."""
        self._index = []
        for record_file in sorted(self._records_dir.glob("*.json")):
            try:
                record = ExperimentRecord.load(record_file)
                self._index.append(record.index_entry())
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        self._save_index()

    def _save_index(self) -> None:
        """Persist the index to disk atomically.

        Phase 7 Stage 7.4 Round 5 (2026-04-20): delegates to the
        canonical ``hft_contracts._atomic_io.atomic_write_json`` —
        unified with ``ExperimentRecord.save`` and
        ``hft_ops.feature_sets.writer.atomic_write_json``. Canonical
        convention (``sort_keys=True`` + trailing newline) ensures
        index.json is byte-stable across runs for diff tooling.

        Atomicity closes the transient-bad-state window on
        SIGKILL/ENOSPC/power failure: prior non-atomic
        ``open(w) + json.dump`` could leave index.json truncated,
        forcing a full ``_rebuild_index`` on the next load (correct
        behavior — no data loss thanks to the JSONDecodeError
        fallback at ``_load_index:48``, but O(N) cost on every
        startup until a clean write lands).
        """
        atomic_write_json(self._index_path, self._index)

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
