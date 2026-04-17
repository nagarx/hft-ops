"""
Sweep-aggregate record writer (Phase 5 FULL-A Block 3, 2026-04-17).

Writes one ``ExperimentRecord`` with ``record_type="sweep_aggregate"`` per
``hft-ops sweep run`` invocation, capturing per-grid-point summaries in
``sub_records`` so ``hft-ops sweep results`` / future dashboard tools can
consume a single roll-up record rather than re-scanning per-point records.

**Fingerprint invariant** (SHOULD-ADOPT / CRITICAL-FIX 8 synthesis): aggregate
fingerprint is content-addressed on ``(sorted(child_fingerprints),
sweep_name)`` — two independent invocations of the SAME sweep that dedupe to
identical child fingerprints also dedupe to the SAME aggregate fingerprint.
The aggregate's ``experiment_id`` is ``{sweep_id}_aggregate`` (deterministic)
so re-running a sweep overwrites the aggregate record file. Per-invocation
identity is carried by ``sweep_id``; ``aggregate_fingerprint`` is a
content hash useful for detecting equivalent-content sweeps across runs.

**Record path** (CRITICAL-FIX 2): written to ``paths.ledger_dir / "records" /
{sweep_id}_aggregate.json`` so ``ExperimentLedger._rebuild_index`` picks it up
via its ``records/*.json`` glob. Writing to the ledger-root sibling of
``records/`` would silently skip indexing.

**Index visibility** (CRITICAL-FIX 3 pair): aggregate records DO show up in
``ledger.filter(sweep_id=...)``. ``sweep_results`` CLI filters them out
explicitly (``record_type != "sweep_aggregate"``) so the results table
doesn't render the aggregate as a phantom row; future dashboards that want
aggregate visibility can opt IN by querying ``record_type="sweep_aggregate"``.

Extracted to its own module (per SHOULD-ADOPT F3) to keep ``cli.py`` focused
on command dispatch; the writer class is independently testable and future
strategies (streaming aggregate, hierarchical roll-ups) become new writer
classes that don't require touching ``cli.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_ops.ledger.experiment_record import ExperimentRecord
from hft_ops.manifest.schema import ExperimentManifest


class SweepAggregateWriter:
    """Writes one parent ``sweep_aggregate`` record per ``hft-ops sweep run``.

    Pure data-side class. No I/O or CLI concerns. Tests target this class
    directly; ``cli.py::sweep_run`` is a single call site.
    """

    def write(
        self,
        *,
        ledger_dir: Path,
        sweep_id: str,
        sweep_name: str,
        manifest: ExperimentManifest,
        child_summaries: List[Dict[str, Any]],
        completed: int,
        failed: int,
    ) -> ExperimentRecord:
        """Create and persist the aggregate record.

        Args:
            ledger_dir: Ledger root (``paths.ledger_dir``).
            sweep_id: Per-invocation identity (e.g. ``"e5_phase2_20260417T102345"``).
            sweep_name: Sweep manifest's ``sweep.name`` field.
            manifest: The (original, sweep-enabled) manifest, for tag /
                hypothesis extraction.
            child_summaries: Per-grid-point summary dicts. Required keys:
                ``experiment_id``, ``fingerprint``, ``axis_values``, ``status``.
                Optional: ``name``, ``training_metrics``, ``backtest_metrics``,
                ``duration_seconds``.
            completed: Number of children that completed.
            failed: Number of children that failed.

        Returns:
            The ``ExperimentRecord`` written to
            ``ledger_dir/records/{sweep_id}_aggregate.json``.

        Side effects:
            Writes one JSON file. Caller is responsible for calling
            ``ledger._rebuild_index()`` afterwards (aggregate saves don't
            auto-register).
        """
        # Import here to break the hft_contracts → hft_ops → hft_contracts loop
        # if future additions to this module need hft_contracts imports.
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        # CRITICAL-FIX 8: content-addressed aggregate fingerprint.
        # Two invocations of the same sweep (different sweep_id, same axis
        # definitions and trainer configs) produce equal child_fps → equal
        # aggregate_fp. This is a feature, not a bug: cross-invocation equality
        # indicates "equivalent experiments ran". Record-level identity is
        # preserved via the deterministic experiment_id ({sweep_id}_aggregate).
        child_fps_sorted = sorted(
            s.get("fingerprint", "") for s in child_summaries if s.get("fingerprint")
        )
        aggregate_fp = sha256_hex(canonical_json_blob({
            "children": child_fps_sorted,
            "sweep_name": sweep_name,
        }))

        # Status synthesis: completed/partial/failed based on child statuses.
        if failed == 0 and completed > 0:
            status = "completed"
        elif completed > 0 and failed > 0:
            status = "partial"
        elif failed > 0:
            status = "failed"
        else:
            status = "unknown"  # 0 children — should not happen in practice

        # Duration: sum across grid points.
        # NB: sequential grid execution (current sweep_run behavior); if
        # parallel execution lands in a future phase, switch to
        # max(durations) with a comment.
        total_duration = sum(
            float(s.get("duration_seconds", 0) or 0) for s in child_summaries
        )

        record = ExperimentRecord(
            experiment_id=f"{sweep_id}_aggregate",
            record_type="sweep_aggregate",
            name=sweep_name,
            fingerprint=aggregate_fp,
            sub_records=list(child_summaries),
            status=status,
            duration_seconds=total_duration,
            hypothesis=manifest.experiment.hypothesis or "",
            tags=list(manifest.experiment.tags or []),
            created_at=datetime.now(timezone.utc).isoformat(),
            sweep_id=sweep_id,
            axis_values={},  # aggregate has no single axis value; children carry theirs
            feature_set_ref=None,  # aggregate doesn't use a single FeatureSet
        )

        # CRITICAL-FIX 2: write into records/ subdir (NOT ledger_dir root)
        # so ExperimentLedger._rebuild_index picks it up via its records/*.json glob.
        records_dir = ledger_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        record.save(records_dir / f"{sweep_id}_aggregate.json")

        return record
