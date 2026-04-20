"""Phase 8A.1 — WorkerPoolExecutor: thread-backed parallel grid-point dispatch.

Executes a list of independent grid-point "tasks" concurrently, collects
results, applies the operator's --on-failure policy, and recovers from
broken-pool conditions.

-------------------------------------------------------------------------
Threads vs Processes — explicit trade-off
-------------------------------------------------------------------------

Phase 8A.1 MVP uses ``ThreadPoolExecutor``. Rationale:

  (+) Workers are subprocess-bound (``subprocess.run`` on cargo +
      lobtrainer + lobbacktest). The GIL is released during blocking
      subprocess I/O, so thread-level parallelism yields genuine
      concurrency for our workload — Python-level compute is < 1% of
      each grid point's wall-clock.

  (+) Picklability constraint absent — task callables + closures work.
      Significantly simplifies test infrastructure (tests use in-module
      factories, not module-level task functions).

  (+) Shared Python state — workers CAN read parent-constructed
      ``ExperimentLedger`` for prior-best queries / check_duplicate
      without cross-process serialization. Writes stay in parent per
      single-writer invariant.

  (−) Cannot use ``os.setsid()`` for per-worker process groups
      (threads share PID). Signal-cascade to subprocess trees therefore
      uses a manual PID tracker (``SubprocessPidTracker`` —
      ``hft_ops/scheduler/signal_handler.py``) that records subprocess
      PIDs at spawn and iterates them on SIGINT.

  (−) A Python-level bug in one worker (e.g., unhandled exception that
      bypasses our exception handler) could corrupt parent state.
      Mitigated by defensive wrapping in ``_run_task_safely``.

If a future concern forces process isolation (e.g., native-memory
leaks in Python deps, or a need to impose per-worker CPU affinity),
migrating to ProcessPoolExecutor requires:
  (a) Task functions must be module-level (not closures).
  (b) Task args must be picklable (our manifests are; good).
  (c) Workers gain ``os.setsid()`` in initializer.
The ``run_all`` public API stays unchanged.

-------------------------------------------------------------------------
Failure policy (--on-failure)
-------------------------------------------------------------------------

  - ``continue`` (DEFAULT): single failure → record ``SWEEP_FAILURE`` +
    proceed to remaining tasks. Transient kinds (oom / gpu_timeout)
    trigger retry up to ``max_retries``; fatal kinds (validation_error /
    assertion) skip retry.
  - ``abort``: single failure → cancel all not-yet-started tasks;
    in-flight tasks finish; return partial result set.
  - ``retry:N``: same as continue but with ``max_retries=N``. (Retry is
    orthogonal to continue/abort; retry is per-task, continue/abort is
    sweep-level.)

-------------------------------------------------------------------------
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


__all__ = [
    "OnFailureMode",
    "WorkerStatus",
    "WorkerResult",
    "WorkerPoolExecutor",
    "parse_on_failure",
    "TRANSIENT_ERROR_KINDS",
    "FATAL_ERROR_KINDS",
]


# ---------------------------------------------------------------------------
# Enums / error-kind taxonomy
# ---------------------------------------------------------------------------


class OnFailureMode(str, Enum):
    CONTINUE = "continue"
    ABORT = "abort"
    RETRY = "retry"


class WorkerStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # duplicate fingerprint, dedup'd out upfront


# Error-kind taxonomy for sweep_failure_info.error_kind.
# Transient kinds trigger retry-up-to-max_retries under --on-failure retry:N.
# Fatal kinds never retry (the retry would just fail the same way).
TRANSIENT_ERROR_KINDS = frozenset({
    "oom",                  # exit 137 / CUDA OOM / OS OOM-killer
    "gpu_acquire_timeout",  # filelock.Timeout after 30 min
    "broken_process_pool",  # worker SIGSEGV recovered via new pool
    "subprocess_nonzero",   # non-zero exit we can't otherwise classify
    "io_error",             # OSError during file I/O
})

FATAL_ERROR_KINDS = frozenset({
    "validation_error",  # manifest schema invalid, config not found
    "assertion",         # AssertionError from stage runner
    "contract_error",    # schema_version mismatch, etc.
    "config_invalid",
    "unknown",           # assume fatal to be safe
})


def parse_on_failure(s: str) -> Tuple[OnFailureMode, int]:
    """Parse ``--on-failure`` CLI flag.

    Grammar:
      ``continue``      → (CONTINUE, 0)
      ``abort``         → (ABORT, 0)
      ``retry:N``       → (RETRY, N) for positive integer N

    Raises ValueError on unrecognized input.
    """
    s = s.strip().lower()
    if s == "continue":
        return OnFailureMode.CONTINUE, 0
    if s == "abort":
        return OnFailureMode.ABORT, 0
    if s.startswith("retry:"):
        try:
            n = int(s[len("retry:"):])
        except ValueError:
            raise ValueError(
                f"--on-failure retry:N requires integer N, got {s!r}"
            )
        if n <= 0:
            raise ValueError(
                f"--on-failure retry:N requires N > 0, got {n}"
            )
        return OnFailureMode.RETRY, n
    raise ValueError(
        f"--on-failure must be one of: continue | abort | retry:N; got {s!r}"
    )


# ---------------------------------------------------------------------------
# WorkerResult — picklable output from each grid-point task
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    """Result of running one grid point.

    Status semantics:
      - ``SUCCESS``   — all stages completed; ``stage_results`` populated.
      - ``FAILED``    — one stage failed; ``error_kind`` / ``stderr_tail``
        populated; remaining stages not run.
      - ``CANCELLED`` — parent cancelled before task started (abort mode
        OR pool shutdown).
      - ``SKIPPED``   — parent pre-dedup'd this fingerprint; task never
        dispatched.
    """

    grid_index: int
    status: WorkerStatus
    fingerprint: str
    axis_values: Dict[str, str] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    error_kind: str = ""
    stderr_tail: str = ""
    exit_code: int = 0
    attempt: int = 1
    duration_seconds: float = 0.0

    @property
    def is_transient_failure(self) -> bool:
        """True iff status=FAILED and error_kind is retryable."""
        return (
            self.status == WorkerStatus.FAILED
            and self.error_kind in TRANSIENT_ERROR_KINDS
        )

    @property
    def is_fatal_failure(self) -> bool:
        return (
            self.status == WorkerStatus.FAILED
            and self.error_kind in FATAL_ERROR_KINDS
        )


# ---------------------------------------------------------------------------
# WorkerPoolExecutor — the thread-backed scheduler
# ---------------------------------------------------------------------------


class WorkerPoolExecutor:
    """Dispatches grid-point tasks across up to ``max_workers`` concurrent
    threads. Each task is a zero-arg callable that returns a
    ``WorkerResult``.

    Typical use:

        executor = WorkerPoolExecutor(
            max_workers=4,
            runtime_dir=paths.ledger_dir / "_runtime",
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=2,
        )
        results = executor.run_all(tasks)

    Attributes:
      max_workers: upper-bound on concurrent grid points.
      runtime_dir: where GPUSemaphore locks + signal_handler state live.
      on_failure_mode: CONTINUE | ABORT | RETRY policy.
      max_retries: when on_failure_mode == RETRY, max retries per
        transient-failing task. Zero means no retry.
      cancel_event: threading.Event set by signal handler on Ctrl-C;
        workers check between stages.
    """

    def __init__(
        self,
        max_workers: int,
        runtime_dir: Path,
        on_failure_mode: OnFailureMode = OnFailureMode.CONTINUE,
        max_retries: int = 0,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be ≥ 1, got {max_workers}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be ≥ 0, got {max_retries}")
        self.max_workers = max_workers
        self.runtime_dir = Path(runtime_dir)
        self.on_failure_mode = on_failure_mode
        self.max_retries = max_retries
        self.cancel_event = threading.Event()

    def run_all(
        self,
        tasks: List[Callable[[], WorkerResult]],
    ) -> List[WorkerResult]:
        """Execute all tasks concurrently; return collected results.

        Args:
          tasks: list of zero-arg callables that each return a
            WorkerResult. Typically produced by a higher-level factory
            that partially-applies (manifest, axis_values, ops_config)
            into the callable's closure.

        Behavior:
          - CONTINUE: all tasks run; failures collected into return list.
          - ABORT: first failure cancels not-yet-started tasks; in-flight
            tasks finish; partial list returned.
          - RETRY: transient failures retried up to ``max_retries``.
        """
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_event.clear()

        results: List[WorkerResult] = []
        task_queue = list(tasks)
        retry_counts: Dict[int, int] = {}  # task idx → retry attempts so far

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="hft-ops-worker",
        ) as pool:
            in_flight: Dict[concurrent.futures.Future, int] = {}
            next_task_idx = 0

            # Prime the pool
            while len(in_flight) < self.max_workers and next_task_idx < len(task_queue):
                task = task_queue[next_task_idx]
                future = pool.submit(_run_task_safely, task)
                in_flight[future] = next_task_idx
                next_task_idx += 1

            abort_requested = False

            while in_flight:
                done, _pending = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    task_idx = in_flight.pop(future)
                    try:
                        result = future.result()
                    except concurrent.futures.process.BrokenProcessPool:
                        # This branch is only hit if a user swaps in the
                        # process backend. Thread backend doesn't raise
                        # BrokenProcessPool. Captured defensively.
                        logger.error(
                            "BrokenProcessPool on task %d; marking failed",
                            task_idx,
                        )
                        result = WorkerResult(
                            grid_index=task_idx,
                            status=WorkerStatus.FAILED,
                            fingerprint="",
                            error_kind="broken_process_pool",
                        )
                    except Exception as exc:  # pragma: no cover — defensive
                        logger.exception(
                            "Unexpected exception from task %d", task_idx
                        )
                        result = WorkerResult(
                            grid_index=task_idx,
                            status=WorkerStatus.FAILED,
                            fingerprint="",
                            error_kind="unknown",
                            stderr_tail=str(exc)[:4096],
                        )

                    # Retry logic (transient failures only)
                    if (
                        result.status == WorkerStatus.FAILED
                        and self.on_failure_mode == OnFailureMode.RETRY
                        and result.is_transient_failure
                        and retry_counts.get(task_idx, 0) < self.max_retries
                    ):
                        retry_counts[task_idx] = retry_counts.get(task_idx, 0) + 1
                        attempt = retry_counts[task_idx] + 1
                        logger.info(
                            "Retrying task %d (attempt %d, prior error_kind=%s)",
                            task_idx, attempt, result.error_kind,
                        )
                        # Re-enqueue with same task (caller's closure)
                        new_future = pool.submit(_run_task_safely, task_queue[task_idx])
                        in_flight[new_future] = task_idx
                        continue  # don't collect the failed result; retry in flight

                    results.append(result)

                    # Abort behavior: on first failure, don't submit any more
                    if (
                        result.status == WorkerStatus.FAILED
                        and self.on_failure_mode == OnFailureMode.ABORT
                    ):
                        abort_requested = True
                        logger.warning(
                            "--on-failure abort: task %d failed (%s); "
                            "cancelling remaining %d tasks",
                            task_idx,
                            result.error_kind,
                            len(task_queue) - next_task_idx,
                        )
                        self.cancel_event.set()
                        # Cancel pending futures
                        for pending_future in list(in_flight.keys()):
                            pending_future.cancel()

                # Submit next task(s) to keep pool saturated (unless abort)
                if not abort_requested:
                    while (
                        len(in_flight) < self.max_workers
                        and next_task_idx < len(task_queue)
                    ):
                        task = task_queue[next_task_idx]
                        future = pool.submit(_run_task_safely, task)
                        in_flight[future] = next_task_idx
                        next_task_idx += 1

        return results


def _run_task_safely(task: Callable[[], WorkerResult]) -> WorkerResult:
    """Defensive wrapper that catches unhandled exceptions from task
    callables and returns them as FAILED WorkerResults. Prevents a
    buggy task from breaking the pool.
    """
    try:
        start = time.monotonic()
        result = task()
        if result.duration_seconds == 0.0:
            result.duration_seconds = time.monotonic() - start
        return result
    except AssertionError as exc:
        return WorkerResult(
            grid_index=-1,
            status=WorkerStatus.FAILED,
            fingerprint="",
            error_kind="assertion",
            stderr_tail=str(exc)[:4096],
        )
    except Exception as exc:  # pragma: no cover — defensive
        return WorkerResult(
            grid_index=-1,
            status=WorkerStatus.FAILED,
            fingerprint="",
            error_kind="unknown",
            stderr_tail=str(exc)[:4096],
        )
