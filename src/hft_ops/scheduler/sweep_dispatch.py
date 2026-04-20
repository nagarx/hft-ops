"""Phase 8A.1 Part 2 — Grid-point stage-runner + parallel dispatch helpers.

This module separates STAGE EXECUTION (pure, no ledger writes) from RECORD
REGISTRATION (parent-only, single-writer). Both the sequential and the
parallel ``hft-ops sweep run`` code paths call the same helper
``run_grid_point_stages`` — eliminating duplication and guaranteeing
identical semantics.

Single-writer invariant (Phase 8A.1 architectural-invariant #1): workers
NEVER touch the ledger. They return ``WorkerResult`` dicts; the parent
process serializes ``_record_experiment`` calls after collecting all
worker results.

GPU semaphore integration: ``run_grid_point_stages`` acquires the
``GPUSemaphore`` for the duration of the GPU-using stages
(``training`` + ``signal_export``) only. Non-GPU stages (extraction,
analysis, backtest) run without holding the semaphore, maximizing
parallelism across workers that share a GPU.

Thread-count env injection: parent computes ``per_worker_threads =
cpu_budget / n_parallel`` and passes via ``cpu_threads`` kwarg. The
helper builds a per-worker ``OpsConfig`` with ``env_overrides`` set to
``{RAYON_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS}`` so the
Rust extractor (rayon) + Python trainer (torch) do not oversubscribe
physical cores.

Cancellation: parent signal handler sets ``cancel_event`` on Ctrl-C.
The helper checks ``cancel_event.is_set()`` between stages, aborting
cleanly (returning WorkerResult with status=CANCELLED) so subsequent
grid points are not dispatched.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hft_ops.config import OpsConfig
from hft_ops.ledger.dedup import compute_fingerprint
from hft_ops.manifest.validator import (
    apply_resolved_context,
    resolve_manifest_context,
)
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.scheduler.executor import (
    FATAL_ERROR_KINDS,
    TRANSIENT_ERROR_KINDS,
    WorkerResult,
    WorkerStatus,
)
from hft_ops.scheduler.resources import GPUSemaphore
from hft_ops.stages.base import StageResult, StageStatus
from hft_ops.stages.extraction import ExtractionRunner
from hft_ops.stages.raw_analysis import RawAnalysisRunner
from hft_ops.stages.dataset_analysis import DatasetAnalysisRunner
from hft_ops.stages.validation import ValidationRunner
from hft_ops.stages.training import TrainingRunner
from hft_ops.stages.post_training_gate import PostTrainingGateRunner
from hft_ops.stages.signal_export import SignalExportRunner
from hft_ops.stages.backtesting import BacktestRunner
from hft_ops.paths import PipelinePaths

logger = logging.getLogger(__name__)


__all__ = [
    "run_grid_point_stages",
    "build_stage_runners_list",
    "classify_error_kind",
    "GPU_STAGES",
]


# Stages that require exclusive GPU access. These hold the GPU semaphore
# during their run; all other stages run without the lock so non-GPU
# work (extraction, analysis, backtest) can overlap across workers that
# share a single GPU.
GPU_STAGES: frozenset[str] = frozenset({"training", "signal_export"})


def build_stage_runners_list(exp: ExperimentManifest) -> List[Tuple[str, bool, Any]]:
    """Construct the canonical ordered list of stage runners.

    Order is load-bearing: extraction → raw_analysis → dataset_analysis →
    validation → training → post_training_gate → signal_export → backtesting.
    Must match the sequence used in both serial and parallel paths (and
    historically, in ``hft-ops run`` single-experiment flow).

    Returns list of (stage_name, enabled, runner_instance) tuples.
    """
    return [
        ("extraction", exp.stages.extraction.enabled, ExtractionRunner()),
        ("raw_analysis", exp.stages.raw_analysis.enabled, RawAnalysisRunner()),
        ("dataset_analysis", exp.stages.dataset_analysis.enabled, DatasetAnalysisRunner()),
        ("validation", exp.stages.validation.enabled, ValidationRunner()),
        ("training", exp.stages.training.enabled, TrainingRunner()),
        (
            "post_training_gate",
            exp.stages.post_training_gate.enabled,
            PostTrainingGateRunner(),
        ),
        ("signal_export", exp.stages.signal_export.enabled, SignalExportRunner()),
        ("backtesting", exp.stages.backtesting.enabled, BacktestRunner()),
    ]


def classify_error_kind(error_message: str, stderr: str) -> str:
    """Heuristically classify a failed stage's error into the taxonomy
    used by ``ExperimentRecord.sweep_failure_info.error_kind``.

    Keys match ``TRANSIENT_ERROR_KINDS`` / ``FATAL_ERROR_KINDS`` in
    ``hft_ops.scheduler.executor``. A future enhancement would have
    stage runners explicitly set ``StageResult.error_kind``; for MVP we
    pattern-match on the error text.
    """
    text = f"{error_message}\n{stderr}".lower()
    if "cuda out of memory" in text or "oom" in text or "killed" in text:
        return "oom"
    if "timeout" in text and "gpu" in text:
        return "gpu_acquire_timeout"
    if "assertion" in text or "assert" in text:
        return "assertion"
    if "contract" in text and ("version" in text or "schema" in text):
        return "contract_error"
    if "validation" in text or "invalid" in text:
        return "validation_error"
    if error_message and "subprocess" in error_message.lower():
        return "subprocess_nonzero"
    if "io" in text or "file not found" in text:
        return "io_error"
    return "unknown"


def run_grid_point_stages(
    exp: ExperimentManifest,
    axis_values: Dict[str, str],
    grid_index: int,
    ops_config: OpsConfig,
    paths: PipelinePaths,
    requested_stages: Optional[Set[str]] = None,
    *,
    gpu_id: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    gpu_runtime_dir: Optional[Path] = None,
    gpu_timeout_seconds: float = 1800.0,
    cancel_event: Optional[threading.Event] = None,
) -> WorkerResult:
    """Run all enabled stages for one grid point. Pure function —
    does NOT touch the ledger. Safe to call from a worker thread.

    Returns a ``WorkerResult`` with ``stage_results`` populated as
    ``Dict[str, StageResult]`` (full objects, not preview dicts).

    Post-audit fix (2026-04-20): originally returned a tuple of
    (WorkerResult-with-preview-dicts, full-StageResult-dict). Caller
    discarded the second element, forcing parent to reconstruct
    StageResult from preview — which DROPPED ``captured_metrics``.
    That silently broke cache_info / gate_reports / feature_set_ref /
    training-metrics harvest under ``--parallel > 1``. Since we run
    on ThreadPoolExecutor (shared memory), the full StageResult
    objects pass through the WorkerResult without serialization.
    A future ProcessPoolExecutor migration would require explicit
    pickle-safe projection — but that's out-of-scope for Part 2 MVP.

    Parent is responsible for:
      - Pre-dispatch dedup (``check_duplicate`` + skip)
      - Console progress output
      - Record registration (``_record_experiment`` OR
        ``SWEEP_FAILURE`` record construction)
      - Sweep-metadata update (sweep_id, axis_values)
      - child_summaries aggregation

    Per-worker resource coordination:
      - ``gpu_id`` (if not None): worker acquires ``GPUSemaphore(gpu_id)``
        around GPU-using stages (training, signal_export). Non-GPU
        stages run without the lock.
      - ``cpu_threads`` (if not None): injected into subprocess env as
        RAYON_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS to prevent
        N workers × all_cores oversubscription.

    Args:
      exp: Resolved per-grid-point ExperimentManifest (already expanded
        from sweep; separate instance from sibling workers).
      axis_values: {axis_name → selected_label} for this grid point.
      grid_index: 0-based index in the expanded sweep (for logging + result).
      ops_config: Base OpsConfig. A per-worker copy with env_overrides
        is constructed internally; the caller's ops_config is not mutated.
      paths: PipelinePaths (read-only; all workers share).
      requested_stages: Optional whitelist of stage names to run. None =
        all enabled stages. Matches ``--stages`` CLI flag.
      gpu_id: If assigned, GPU index for GPU-using stages. None = CPU-only.
      cpu_threads: If set, per-worker thread budget injected into
        subprocess env.
      gpu_runtime_dir: Directory for GPUSemaphore lock files. Required
        if gpu_id is not None.
      gpu_timeout_seconds: Seconds to wait for GPU acquire. Default 30
        min — long enough to queue behind one in-progress training run.
      cancel_event: Optional threading.Event set by parent signal
        handler on Ctrl-C; worker checks between stages.

    Returns:
      (WorkerResult, Dict[stage_name, StageResult])

    Raises:
      Never — all exceptions are caught and surfaced as FAILED
      WorkerResults with classified error_kind.
    """
    # Compute fingerprint. Parent has already dedup'd based on the same
    # fingerprint; computing here again is a couple-of-ms redundancy that
    # keeps the helper self-contained and avoids a mandatory parameter.
    fingerprint = compute_fingerprint(exp, paths)

    # Build per-worker ops_config with env_overrides. frozen dataclass
    # + replace() gives us an immutable copy that's safe to pass across
    # worker boundaries.
    worker_env: Dict[str, str] = dict(ops_config.env_overrides)
    if cpu_threads is not None and cpu_threads > 0:
        worker_env.setdefault("RAYON_NUM_THREADS", str(cpu_threads))
        worker_env.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        worker_env.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    base_ops_config = dataclasses.replace(ops_config, env_overrides=worker_env)

    # Resolve per-grid-point runtime values. No shared-state mutation —
    # exp is a separate instance per grid point (from expand_sweep).
    resolved_ctx = resolve_manifest_context(exp, paths)
    apply_resolved_context(exp, resolved_ctx)

    stage_runners = build_stage_runners_list(exp)

    results: Dict[str, StageResult] = {}
    point_start = time.monotonic()
    point_failed = False
    error_kind = ""
    stderr_tail = ""
    exit_code = 0

    for stage_name, enabled, runner in stage_runners:
        # Cancellation check — parent SIGINT handler sets this between stages.
        if cancel_event is not None and cancel_event.is_set():
            logger.info(
                "Grid point %d cancelled before stage '%s' (parent SIGINT)",
                grid_index, stage_name,
            )
            duration = time.monotonic() - point_start
            return WorkerResult(
                grid_index=grid_index,
                status=WorkerStatus.CANCELLED,
                fingerprint=fingerprint,
                axis_values=axis_values,
                # Full StageResult pass-through (threads share memory;
                # no pickle concern). Post-audit fix for captured_metrics
                # data-loss under --parallel > 1.
                stage_results=dict(results),
                error_kind="cancelled",
                stderr_tail="cancelled by parent signal handler",
                duration_seconds=duration,
            )

        if not enabled:
            continue
        if requested_stages is not None and stage_name not in requested_stages:
            continue

        # Build stage-specific ops_config: GPU env + training/signal_export
        # gets CUDA_VISIBLE_DEVICES; other stages get the base env_overrides.
        stage_uses_gpu = stage_name in GPU_STAGES and gpu_id is not None
        if stage_uses_gpu:
            stage_env = dict(base_ops_config.env_overrides)
            stage_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            stage_ops_config = dataclasses.replace(base_ops_config, env_overrides=stage_env)
        else:
            stage_ops_config = base_ops_config

        # Validate inputs
        input_errors = runner.validate_inputs(exp, stage_ops_config)
        if input_errors:
            err_text = "; ".join(input_errors)
            results[stage_name] = StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                error_message=err_text,
            )
            point_failed = True
            error_kind = "validation_error"
            stderr_tail = err_text[-4096:]
            break

        # Run the stage, acquiring GPU semaphore if this is a GPU stage.
        try:
            if stage_uses_gpu and gpu_runtime_dir is not None:
                sem = GPUSemaphore(
                    gpu_id=gpu_id,
                    runtime_dir=gpu_runtime_dir,
                    timeout_seconds=gpu_timeout_seconds,
                )
                try:
                    with sem:
                        result = runner.run(exp, stage_ops_config)
                except Exception as exc:
                    # filelock.Timeout or OSError from sem.acquire
                    exc_text = str(exc)
                    result = StageResult(
                        stage_name=stage_name,
                        status=StageStatus.FAILED,
                        error_message=(
                            f"GPU {gpu_id} acquire failed after "
                            f"{gpu_timeout_seconds}s: {exc_text}"
                        ),
                    )
                    error_kind = "gpu_acquire_timeout"
            else:
                result = runner.run(exp, stage_ops_config)
        except Exception as exc:  # pragma: no cover — defensive
            # Stage runner raised uncaught exception. Classify + continue.
            logger.exception("Stage '%s' raised uncaught exception", stage_name)
            result = StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                stderr=str(exc),
            )

        results[stage_name] = result

        if result.status == StageStatus.FAILED:
            point_failed = True
            if not error_kind:
                error_kind = classify_error_kind(
                    result.error_message or "",
                    result.stderr or "",
                )
            stderr_tail = (result.stderr or result.error_message or "")[-4096:]
            break

    duration = time.monotonic() - point_start

    worker_result = WorkerResult(
        grid_index=grid_index,
        status=WorkerStatus.FAILED if point_failed else WorkerStatus.SUCCESS,
        fingerprint=fingerprint,
        axis_values=axis_values,
        # Post-audit fix (2026-04-20): full StageResult pass-through.
        # ThreadPoolExecutor shares memory → no serialization needed.
        # Parent's _record_experiment reads captured_metrics from multiple
        # stages (extraction.cache_info, */.gate_report, signal_export.
        # feature_set_ref, training.training_metrics). Preview-dicts would
        # silently drop these fields → ledger data-loss under --parallel>1.
        stage_results=dict(results),
        error_kind=error_kind,
        stderr_tail=stderr_tail,
        exit_code=exit_code,
        attempt=1,  # Parent overrides on retry
        duration_seconds=duration,
    )
    return worker_result
