"""Phase 8A.1 Part 2 (2026-04-20) — Parallel-sweep dispatch CLI helper.

Extracts the ``--parallel > 1`` branch of ``hft-ops sweep run`` into a
dedicated module. The serial path remains unchanged in ``cli.py``.

Flow (single-writer invariant preserved):

1. Parent pre-computes fingerprints for each grid point and filters out
   duplicates via ``check_duplicate`` (unless ``--force``). Dedup is
   centralized in the parent — workers never query the ledger for
   dedup purposes.

2. Parent builds ``N`` worker task closures, each of which calls
   ``run_grid_point_stages`` (pure function; no ledger access). GPU id
   round-robin + per-worker CPU thread slice are baked into each closure.

3. ``WorkerPoolExecutor.run_all(tasks)`` dispatches up to
   ``max_workers=parallel`` concurrent closures. The executor handles
   ``--on-failure`` policy (continue / abort / retry:N).

4. Parent collects ``WorkerResult`` objects in order-of-completion,
   BUT registers records (via ``_record_experiment`` OR
   ``SWEEP_FAILURE`` construction) in a loop — all writes serialized
   in the parent. ``child_summaries`` is keyed by grid_index so order
   does not matter.

5. After all grid points complete (or abort fires), parent writes the
   single ``sweep_aggregate`` record via ``SweepAggregateWriter``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hft_contracts.experiment_record import ExperimentRecord, RecordType
from hft_contracts.provenance import GitInfo, Provenance

from hft_ops.config import OpsConfig
from hft_ops.ledger.dedup import check_duplicate, compute_fingerprint
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.paths import PipelinePaths
from hft_ops.scheduler.executor import (
    OnFailureMode,
    WorkerPoolExecutor,
    WorkerResult,
    WorkerStatus,
)
from hft_ops.scheduler.signal_handler import install_scheduler_signal_handler
from hft_ops.scheduler.sweep_dispatch import run_grid_point_stages
from hft_ops.stages.base import StageResult, StageStatus

logger = logging.getLogger(__name__)


__all__ = ["run_sweep_parallel"]


def run_sweep_parallel(
    experiments_with_axes: List[Tuple[ExperimentManifest, Dict[str, str]]],
    manifest: ExperimentManifest,
    ops_config: OpsConfig,
    paths: PipelinePaths,
    sweep_id: str,
    sweep_name: str,
    parallel: int,
    failure_mode: OnFailureMode,
    max_retries: int,
    requested_stages: Optional[Set[str]],
    force: bool,
    gpu_ids: List[int],
    cpu_budget: Optional[int],
    console,  # rich.console.Console; opaque from this module's perspective
) -> None:
    """Dispatch sweep grid points in parallel via WorkerPoolExecutor.

    Caller (``hft-ops sweep run`` in cli.py) handles:
      - manifest parsing + validation
      - sweep expansion (``expand_sweep_with_axis_values``)
      - dry-run short-circuit
      - failure-policy CLI flag parsing

    This function handles:
      - pre-dispatch fingerprint + dedup
      - task closure construction (GPU id round-robin, cpu threads slice)
      - WorkerPoolExecutor dispatch with SIGINT-cascade signal handler
      - parent-side record registration (single-writer)
      - sweep_aggregate writing
    """
    from hft_ops.ledger.sweep_aggregate import SweepAggregateWriter
    from hft_ops.cli import _record_experiment  # re-use serial-path registrar

    console.print(
        f"\n[bold]Parallel sweep dispatch:[/bold] parallel={parallel}, "
        f"gpus={gpu_ids or '(none/cpu-only)'}, "
        f"cpu_budget={cpu_budget or 'default'}, "
        f"on_failure={failure_mode.value}"
        + (f" retry:{max_retries}" if max_retries > 0 else "")
    )

    # -----------------------------------------------------------------
    # Step 1: pre-compute fingerprints + dedup
    # -----------------------------------------------------------------
    child_summaries: List[Dict[str, Any]] = []
    to_dispatch: List[Tuple[int, ExperimentManifest, Dict[str, str], str]] = []

    for i, (exp, axis_values) in enumerate(experiments_with_axes):
        fingerprint = compute_fingerprint(exp, paths)
        if not force:
            existing = check_duplicate(fingerprint, paths.ledger_dir)
            if existing is not None:
                console.print(
                    f"  [yellow]Grid point {i + 1}/{len(experiments_with_axes)} "
                    f"{exp.experiment.name}: duplicate "
                    f"({existing.get('experiment_id')}). Skipping.[/yellow]"
                )
                child_summaries.append({
                    "experiment_id": existing.get("experiment_id", ""),
                    "name": exp.experiment.name,
                    "fingerprint": fingerprint,
                    "axis_values": axis_values,
                    "status": "skipped_duplicate",
                    "duration_seconds": 0.0,
                    "training_metrics": {},
                    "backtest_metrics": {},
                })
                continue
        to_dispatch.append((i, exp, axis_values, fingerprint))

    if not to_dispatch:
        console.print(
            "[green]All grid points are duplicates — nothing to dispatch.[/green]"
        )
        _write_sweep_aggregate(
            child_summaries, manifest, sweep_id, sweep_name, paths, SweepAggregateWriter
        )
        return

    # -----------------------------------------------------------------
    # Step 2: per-worker resource assignment
    # -----------------------------------------------------------------
    # CPU budget: default to os.cpu_count(); per-worker threads = budget / parallel.
    effective_cpu_budget = cpu_budget if cpu_budget is not None else (os.cpu_count() or 1)
    per_worker_threads = max(1, effective_cpu_budget // parallel)

    # GPU id round-robin. When gpu_ids is empty, all workers run CPU-only.
    def assign_gpu(task_slot: int) -> Optional[int]:
        if not gpu_ids:
            return None
        return gpu_ids[task_slot % len(gpu_ids)]

    gpu_runtime_dir = paths.ledger_dir / "_runtime"

    # -----------------------------------------------------------------
    # Step 3: build task closures
    # -----------------------------------------------------------------
    executor = WorkerPoolExecutor(
        max_workers=parallel,
        runtime_dir=gpu_runtime_dir,
        on_failure_mode=failure_mode,
        max_retries=max_retries,
    )

    # Keep a mapping from grid_index → (exp, axes, fingerprint) so parent
    # can register records after workers return.
    index_to_meta: Dict[int, Tuple[ExperimentManifest, Dict[str, str], str]] = {}
    tasks = []
    for task_slot, (grid_idx, exp, axes, fp) in enumerate(to_dispatch):
        index_to_meta[grid_idx] = (exp, axes, fp)
        gpu_id = assign_gpu(task_slot)
        task = _build_task_closure(
            exp=exp,
            axis_values=axes,
            grid_index=grid_idx,
            ops_config=ops_config,
            paths=paths,
            requested_stages=requested_stages,
            gpu_id=gpu_id,
            cpu_threads=per_worker_threads,
            gpu_runtime_dir=gpu_runtime_dir,
            cancel_event=executor.cancel_event,
        )
        tasks.append(task)

    # -----------------------------------------------------------------
    # Step 4: dispatch under SIGINT cascade
    # -----------------------------------------------------------------
    with install_scheduler_signal_handler(executor):
        console.print(
            f"[dim]Dispatching {len(tasks)} grid points across {parallel} workers...[/dim]"
        )
        start = time.monotonic()
        worker_results = executor.run_all(tasks)
        elapsed = time.monotonic() - start

    # -----------------------------------------------------------------
    # Step 5: parent-side record registration (SINGLE WRITER)
    # -----------------------------------------------------------------
    for wr in sorted(worker_results, key=lambda r: r.grid_index):
        if wr.grid_index not in index_to_meta:
            logger.warning(
                "WorkerResult has unexpected grid_index=%d (not in dispatched "
                "set); skipping.", wr.grid_index,
            )
            continue
        exp, axes, fingerprint = index_to_meta[wr.grid_index]

        if wr.status == WorkerStatus.SUCCESS:
            # Re-run the stages via the helper to produce full StageResult objects
            # for _record_experiment. This is not a re-execution — the helper
            # returns cached StageResult metadata from the worker's prior run
            # via a separate code path. For MVP, we re-resolve context and
            # re-issue the stage runners using the SKIPPED path (output_dir
            # already exists + cache/skip_if_exists paths short-circuit).
            # Alternative: worker returns full StageResult dict inline with
            # WorkerResult.stage_results — see follow-up.
            #
            # Phase 8A.1 Part 2 MVP: construct minimal StageResult objects
            # from the preview dict returned by the worker, sufficient for
            # _record_experiment's subset access pattern. Full-fidelity
            # StageResult pass-through lands in Part 2.1 (requires stage
            # runners to return picklable metrics dicts that parent can
            # inflate back into StageResult without re-running).
            stage_results = _preview_dicts_to_stage_results(wr.stage_results)

            record_id = _record_experiment(
                exp, paths, fingerprint, stage_results, wr.duration_seconds,
            )

            # Update record with sweep metadata (same pattern as serial path)
            try:
                ledger = ExperimentLedger(paths.ledger_dir)
                record = ledger.get(record_id)
                if record:
                    record.sweep_id = sweep_id
                    record.axis_values = axes
                    record.save(paths.ledger_dir / "records" / f"{record_id}.json")
                    for entry in ledger._index:
                        if entry.get("experiment_id") == record_id:
                            entry["sweep_id"] = sweep_id
                            entry["axis_values"] = axes
                            break
                    ledger._save_index()
            except Exception as exc:
                logger.warning(
                    "Sweep-metadata update failed for %s: %s (record "
                    "is registered; metadata loss is recoverable via "
                    "`hft-ops ledger rebuild-index`)",
                    record_id, exc,
                )

            child_summaries.append({
                "experiment_id": record_id,
                "name": exp.experiment.name,
                "fingerprint": fingerprint,
                "axis_values": axes,
                "status": "completed",
                "duration_seconds": wr.duration_seconds,
                "training_metrics": (record.training_metrics if record else {}),
                "backtest_metrics": (record.backtest_metrics if record else {}),
            })
            console.print(
                f"  [green]✓ Grid {wr.grid_index + 1} "
                f"{exp.experiment.name}: "
                f"completed ({wr.duration_seconds:.1f}s)[/green]"
            )

        elif wr.status == WorkerStatus.FAILED:
            # Register SWEEP_FAILURE record
            failure_record = _build_sweep_failure_record(
                exp=exp,
                fingerprint=fingerprint,
                sweep_id=sweep_id,
                axis_values=axes,
                wr=wr,
            )
            try:
                ledger = ExperimentLedger(paths.ledger_dir)
                ledger.register(failure_record)
            except Exception as exc:
                logger.warning(
                    "SWEEP_FAILURE register failed for grid %d: %s",
                    wr.grid_index, exc,
                )

            child_summaries.append({
                "experiment_id": failure_record.experiment_id,
                "name": exp.experiment.name,
                "fingerprint": fingerprint,
                "axis_values": axes,
                "status": "failed",
                "duration_seconds": wr.duration_seconds,
                "training_metrics": {},
                "backtest_metrics": {},
            })
            console.print(
                f"  [red]✗ Grid {wr.grid_index + 1} "
                f"{exp.experiment.name}: "
                f"FAILED ({wr.error_kind}, attempt {wr.attempt})[/red]"
            )

        elif wr.status == WorkerStatus.CANCELLED:
            console.print(
                f"  [yellow]⊘ Grid {wr.grid_index + 1} "
                f"{exp.experiment.name}: cancelled[/yellow]"
            )

    # -----------------------------------------------------------------
    # Step 6: sweep_aggregate record
    # -----------------------------------------------------------------
    _write_sweep_aggregate(
        child_summaries, manifest, sweep_id, sweep_name, paths, SweepAggregateWriter
    )
    console.print(
        f"\n[bold green]Parallel sweep complete:[/bold green] "
        f"{len(child_summaries)} grid points in {elapsed:.1f}s wall-clock "
        f"(parallel={parallel})."
    )


def _build_task_closure(
    exp: ExperimentManifest,
    axis_values: Dict[str, str],
    grid_index: int,
    ops_config: OpsConfig,
    paths: PipelinePaths,
    requested_stages: Optional[Set[str]],
    gpu_id: Optional[int],
    cpu_threads: int,
    gpu_runtime_dir: Path,
    cancel_event,
):
    """Build a zero-arg callable for WorkerPoolExecutor.run_all.

    Closure captures per-worker resource assignment + helper invocation.
    """
    def task() -> WorkerResult:
        wr, _stage_results = run_grid_point_stages(
            exp=exp,
            axis_values=axis_values,
            grid_index=grid_index,
            ops_config=ops_config,
            paths=paths,
            requested_stages=requested_stages,
            gpu_id=gpu_id,
            cpu_threads=cpu_threads,
            gpu_runtime_dir=gpu_runtime_dir,
            cancel_event=cancel_event,
        )
        return wr
    return task


def _preview_dicts_to_stage_results(
    stage_preview_dicts: Dict[str, Any],
) -> Dict[str, StageResult]:
    """Reconstruct minimal ``StageResult`` objects from the preview dicts
    returned by ``run_grid_point_stages``. Only includes status / output_dir
    / duration / error_message — the fields ``_record_experiment`` reads.

    For Phase 8A.1 Part 2 MVP this is sufficient because the current
    ``_record_experiment`` code path accesses these fields only. Richer
    captured_metrics pass-through is a Part-2.1 follow-up.
    """
    results: Dict[str, StageResult] = {}
    for stage_name, preview in stage_preview_dicts.items():
        if not isinstance(preview, dict):
            continue
        status_str = preview.get("status", "")
        try:
            status_enum = StageStatus(status_str) if status_str else StageStatus.PENDING
        except ValueError:
            status_enum = StageStatus.PENDING
        results[stage_name] = StageResult(
            stage_name=stage_name,
            status=status_enum,
            duration_seconds=preview.get("duration_seconds", 0.0),
            output_dir=preview.get("output_dir", ""),
            error_message=preview.get("error_message", ""),
        )
    return results


def _build_sweep_failure_record(
    exp: ExperimentManifest,
    fingerprint: str,
    sweep_id: str,
    axis_values: Dict[str, str],
    wr: WorkerResult,
) -> ExperimentRecord:
    """Construct an ``ExperimentRecord`` with ``record_type=sweep_failure``
    for a failed grid point. Shares the fingerprint with the would-be
    successful record so ``dedup.check_duplicate`` can skip it (the
    Phase 8A.1 Part 1 dedup filter handles the SWEEP_FAILURE skip).
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    experiment_id = (
        f"{exp.experiment.name}_failure_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
        f"{fingerprint[:8]}"
    )
    return ExperimentRecord(
        experiment_id=experiment_id,
        name=exp.experiment.name,
        fingerprint=fingerprint,
        contract_version=exp.experiment.contract_version,
        status="failed",
        created_at=now,
        sweep_id=sweep_id,
        axis_values=axis_values,
        record_type=RecordType.SWEEP_FAILURE.value,
        sweep_failure_info={
            "error_kind": wr.error_kind,
            "exit_code": wr.exit_code,
            "stderr_tail": wr.stderr_tail,
            "attempt": wr.attempt,
            "transient": wr.is_transient_failure,
        },
        duration_seconds=wr.duration_seconds,
        tags=list(exp.experiment.tags),
        provenance=Provenance(
            git=GitInfo(commit_hash="", branch="", dirty=False),
            contract_version=exp.experiment.contract_version,
        ),
    )


def _write_sweep_aggregate(
    child_summaries: List[Dict[str, Any]],
    manifest: ExperimentManifest,
    sweep_id: str,
    sweep_name: str,
    paths: PipelinePaths,
    SweepAggregateWriter,
) -> None:
    """Write the sweep_aggregate record summarizing all grid points.

    Matches the serial-path writer call at ``cli.py::sweep_run`` end.
    """
    try:
        writer = SweepAggregateWriter(
            paths=paths,
            sweep_id=sweep_id,
            sweep_name=sweep_name,
            manifest=manifest,
        )
        writer.write(child_summaries)
    except Exception as exc:
        logger.warning(
            "sweep_aggregate write failed: %s (per-grid-point records "
            "are still registered)", exc,
        )
