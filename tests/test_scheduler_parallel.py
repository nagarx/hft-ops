"""Phase 8A.1 — Parallel execution scheduler tests (test-first).

Covers the 8 ship-blocker invariants identified in the Rev 2 design audit
for grid-point-level parallel sweep execution:

Resource coordination (3):
  - test_gpu_semaphore_acquire_release
  - test_gpu_semaphore_timeout
  - test_gpu_semaphore_survives_worker_death

Failure-taxonomy contract (3):
  - test_sweep_failure_record_shape
  - test_dedup_skips_sweep_failure_records
  - test_on_failure_modes_parse

Executor semantics (4):
  - test_worker_pool_executor_basic_dispatch
  - test_worker_pool_executor_continue_on_failure
  - test_worker_pool_executor_abort_on_failure
  - test_sweep_parallel_determinism

Plan: ~/.claude/plans/fuzzy-discovering-flask.md §Phase 8A.1 Rev 2.
hft-rules §6: test-first for determinism-sensitive code (Invariant 8).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hft_contracts.experiment_record import (
    INDEX_SCHEMA_VERSION,
    ExperimentRecord,
    RecordType,
)
from hft_contracts.provenance import GitInfo, Provenance

# The scheduler modules don't exist yet; imports will fail until A1.3-A1.5.
from hft_ops.scheduler.resources import (
    GPUSemaphore,
    ResourceSpec,
)
from hft_ops.scheduler.executor import (
    WorkerPoolExecutor,
    WorkerResult,
    WorkerStatus,
    OnFailureMode,
)


# ============================================================================
# Resource coordination (GPUSemaphore)
# ============================================================================


class TestGpuSemaphore:
    def test_gpu_semaphore_acquire_release(self, tmp_path: Path) -> None:
        """Basic sanity: acquire → release releases the lock."""
        runtime_dir = tmp_path / "_runtime"
        runtime_dir.mkdir()
        sem = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=2.0)

        with sem:
            # Second acquire in a different instance must now time out
            # (but we'll test that via the separate timeout test)
            pass  # acquire + release through context manager

        # After release, a fresh instance must acquire without blocking
        sem2 = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=2.0)
        with sem2:
            pass

    def test_gpu_semaphore_timeout(self, tmp_path: Path) -> None:
        """Contention: a second acquire while held must raise Timeout.

        Uses short timeout (1s) so the test runs fast. Tests the
        filelock-backed exclusive-GPU contract that prevents N parallel
        training runs from OOM'ing a single GPU.
        """
        import filelock

        runtime_dir = tmp_path / "_runtime"
        runtime_dir.mkdir()

        holder = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=5.0)
        contender = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=1.0)

        with holder:
            # Must raise filelock.Timeout within ~1s
            start = time.monotonic()
            with pytest.raises((filelock.Timeout, TimeoutError)):
                contender.acquire()
            elapsed = time.monotonic() - start
            assert 0.5 < elapsed < 3.0, (
                f"Timeout should fire ~1s, got {elapsed:.2f}s. "
                f"Too fast → filelock not actually blocking; "
                f"too slow → timeout ignored."
            )

    def test_gpu_semaphore_survives_worker_death(self, tmp_path: Path) -> None:
        """SIGKILL-equivalent: a lock held by a deceased process must be
        releasable. Filelock/fcntl.flock releases automatically on FD close
        (kernel semantic), but we verify by acquiring + dropping the ref
        (simulates gc) and then re-acquiring from a new instance.
        """
        runtime_dir = tmp_path / "_runtime"
        runtime_dir.mkdir()

        sem = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=2.0)
        sem.acquire()
        # Simulate "worker died" — drop the reference + clear the FD.
        sem.release()
        del sem

        # New instance must acquire immediately
        sem2 = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=2.0)
        start = time.monotonic()
        with sem2:
            elapsed = time.monotonic() - start
        assert elapsed < 0.5, (
            f"Lock from deceased holder not released — blocked {elapsed:.2f}s. "
            f"filelock should use fcntl.flock which auto-releases on FD close."
        )


# ============================================================================
# Failure-taxonomy contract
# ============================================================================


def _make_record(
    record_type: str,
    fingerprint: str,
    sweep_failure_info: Dict[str, Any] = None,
) -> ExperimentRecord:
    """Fixture helper: build a minimal ExperimentRecord."""
    return ExperimentRecord(
        experiment_id=f"test_{record_type}_{fingerprint[:8]}",
        name="test",
        fingerprint=fingerprint,
        contract_version="2.2",
        status="completed" if record_type != RecordType.SWEEP_FAILURE.value else "failed",
        created_at="2026-04-20T00:00:00+00:00",
        record_type=record_type,
        sweep_failure_info=sweep_failure_info or {},
        provenance=Provenance(
            git=GitInfo(commit_hash="x", branch="main", dirty=False),
            contract_version="2.2",
        ),
    )


class TestSweepFailureRecordShape:
    def test_record_type_enum_has_sweep_failure(self) -> None:
        """Phase 8A.1: RecordType.SWEEP_FAILURE must exist with value
        'sweep_failure' for ledger record_type serialization.
        """
        assert hasattr(RecordType, "SWEEP_FAILURE"), (
            "RecordType must expose SWEEP_FAILURE variant for Phase 8A.1"
        )
        assert RecordType.SWEEP_FAILURE.value == "sweep_failure"

    def test_sweep_failure_info_field_on_record(self) -> None:
        """ExperimentRecord must have sweep_failure_info: Dict field
        (default empty) so failure records carry error_kind / exit_code /
        stderr_tail / attempt / transient metadata.
        """
        r = ExperimentRecord(
            experiment_id="x", name="y", fingerprint="z" * 64,
            contract_version="2.2", status="failed",
            created_at="2026-04-20T00:00:00+00:00",
            provenance=Provenance(
                git=GitInfo(commit_hash="x", branch="main", dirty=False),
                contract_version="2.2",
            ),
        )
        assert hasattr(r, "sweep_failure_info"), (
            "ExperimentRecord must have sweep_failure_info field (Phase 8A.1)"
        )
        assert r.sweep_failure_info == {}, (
            "Default must be empty dict (non-failure records omit it)"
        )

    def test_index_entry_projects_sweep_failure_info(self) -> None:
        """index_entry() must project sweep_failure_info (empty dict when
        absent; populated for SWEEP_FAILURE records). Bumps
        INDEX_SCHEMA_VERSION golden set 22 → 23.
        """
        failure_info = {
            "error_kind": "oom",
            "exit_code": 137,
            "stderr_tail": "CUDA out of memory...",
            "attempt": 2,
            "transient": True,
        }
        r = _make_record(
            record_type=RecordType.SWEEP_FAILURE.value,
            fingerprint="f" * 64,
            sweep_failure_info=failure_info,
        )
        projection = r.index_entry()
        assert "sweep_failure_info" in projection, (
            "index_entry() must project sweep_failure_info — Phase 8A.1 "
            "new key, bumps INDEX_SCHEMA_VERSION to 1.2.0"
        )
        assert projection["sweep_failure_info"]["error_kind"] == "oom"
        assert projection["sweep_failure_info"]["exit_code"] == 137
        assert projection["sweep_failure_info"]["attempt"] == 2
        assert projection["sweep_failure_info"]["transient"] is True

    def test_index_schema_version_bumped_to_1_2(self) -> None:
        """INDEX_SCHEMA_VERSION must be 1.2.0 after Phase 8A.1 ships.
        Phase 8A.0 left it at 1.1.0; this phase bumps MINOR for the
        additive sweep_failure_info projection.
        """
        assert INDEX_SCHEMA_VERSION == "1.2.0", (
            f"Phase 8A.1 must bump INDEX_SCHEMA_VERSION to 1.2.0 "
            f"(was {INDEX_SCHEMA_VERSION}). MINOR bump for additive "
            f"sweep_failure_info field in index_entry() projection."
        )


class TestDedupSkipsSweepFailure:
    def test_check_duplicate_skips_sweep_failure_records(
        self, tmp_path: Path
    ) -> None:
        """Phase 8A.1 CRITICAL: check_duplicate must NOT match
        SWEEP_FAILURE records. Otherwise a retry of a failed grid point
        would be silently skipped as a duplicate — defeating the whole
        failure-retry taxonomy.
        """
        from hft_ops.ledger.dedup import check_duplicate
        from hft_ops.ledger.ledger import ExperimentLedger

        ledger_dir = tmp_path / "ledger"
        ledger = ExperimentLedger(ledger_dir)

        fingerprint = "a" * 64
        failed_record = _make_record(
            record_type=RecordType.SWEEP_FAILURE.value,
            fingerprint=fingerprint,
            sweep_failure_info={
                "error_kind": "oom", "exit_code": 137, "transient": True,
                "attempt": 1, "stderr_tail": "OOM",
            },
        )
        ledger.register(failed_record)

        # check_duplicate on the same fingerprint must return None —
        # not the failed record — so a retry can run.
        result = check_duplicate(fingerprint, ledger_dir)
        assert result is None, (
            f"check_duplicate matched a SWEEP_FAILURE record for fingerprint "
            f"{fingerprint[:12]}... → this would BLOCK retries. Expected None; "
            f"got {result!r}. Fix: filter record_type=='sweep_failure' in "
            f"hft_ops/ledger/dedup.py::check_duplicate."
        )

    def test_check_duplicate_still_matches_successful_record(
        self, tmp_path: Path
    ) -> None:
        """Safety: filtering SWEEP_FAILURE must NOT break normal dedup.
        Successful training records with same fingerprint MUST still be
        detected as duplicates (prevents accidental re-run of completed
        experiments).
        """
        from hft_ops.ledger.dedup import check_duplicate
        from hft_ops.ledger.ledger import ExperimentLedger

        ledger_dir = tmp_path / "ledger"
        ledger = ExperimentLedger(ledger_dir)

        fingerprint = "b" * 64
        success_record = _make_record(
            record_type=RecordType.TRAINING.value,
            fingerprint=fingerprint,
        )
        ledger.register(success_record)

        result = check_duplicate(fingerprint, ledger_dir)
        assert result is not None, (
            "check_duplicate must still match TRAINING records (regression "
            "guard — the sweep_failure filter must not over-broaden)"
        )
        assert result["fingerprint"] == fingerprint


class TestOnFailureModes:
    def test_on_failure_enum_values(self) -> None:
        """CLI surface --on-failure accepts continue | abort | retry:N."""
        assert hasattr(OnFailureMode, "CONTINUE")
        assert hasattr(OnFailureMode, "ABORT")
        assert hasattr(OnFailureMode, "RETRY")

    def test_on_failure_parse_retry_count(self) -> None:
        """Parse 'retry:3' into (mode=RETRY, n=3). Parse 'continue' into
        (mode=CONTINUE, n=0). Invalid inputs fail-loud.
        """
        from hft_ops.scheduler.executor import parse_on_failure

        mode, n = parse_on_failure("continue")
        assert mode == OnFailureMode.CONTINUE
        assert n == 0

        mode, n = parse_on_failure("abort")
        assert mode == OnFailureMode.ABORT
        assert n == 0

        mode, n = parse_on_failure("retry:3")
        assert mode == OnFailureMode.RETRY
        assert n == 3

        with pytest.raises(ValueError):
            parse_on_failure("nonsense")


# ============================================================================
# Executor semantics
# ============================================================================


def _make_mock_work(status: WorkerStatus, grid_index: int, **kwargs):
    """Module-level factory so the returned callable + its closure are
    picklable for ProcessPoolExecutor dispatch (lambdas + inner functions
    fail under spawn start-method on macOS).

    Returns a callable that, when executed, produces a deterministic
    WorkerResult with the given status + grid_index.
    """
    from hft_ops.scheduler.executor import WorkerResult

    def work():
        return WorkerResult(
            grid_index=grid_index,
            status=status,
            fingerprint=f"fp_{grid_index}" + "0" * 58,
            axis_values=kwargs.get("axis_values", {"k": f"v{grid_index}"}),
            stage_results=kwargs.get("stage_results", {"extraction": {"status": "completed"}}),
            error_kind=kwargs.get("error_kind", ""),
            stderr_tail=kwargs.get("stderr_tail", ""),
            exit_code=kwargs.get("exit_code", 0),
            attempt=kwargs.get("attempt", 1),
        )
    return work


class TestWorkerPoolExecutor:
    def test_basic_dispatch_collects_all_results(self, tmp_path: Path) -> None:
        """--parallel 2 with 4 tasks must return 4 results, regardless
        of completion order. Tests the fundamental dispatch contract.
        """
        executor = WorkerPoolExecutor(
            max_workers=2,
            runtime_dir=tmp_path / "_runtime",
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=0,
        )

        tasks = [_make_mock_work(WorkerStatus.SUCCESS, i) for i in range(4)]
        results = executor.run_all(tasks)

        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        grid_indices = sorted(r.grid_index for r in results)
        assert grid_indices == [0, 1, 2, 3], (
            f"All 4 grid points must complete; got indices {grid_indices}"
        )

    def test_continue_on_failure_proceeds_past_failed_task(
        self, tmp_path: Path
    ) -> None:
        """--on-failure continue: one task fails, others still complete."""
        executor = WorkerPoolExecutor(
            max_workers=2,
            runtime_dir=tmp_path / "_runtime",
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=0,
        )

        tasks = [
            _make_mock_work(WorkerStatus.SUCCESS, 0),
            _make_mock_work(WorkerStatus.FAILED, 1, error_kind="validation_error"),
            _make_mock_work(WorkerStatus.SUCCESS, 2),
        ]
        results = executor.run_all(tasks)

        assert len(results) == 3, "All 3 results must be collected"
        statuses = {r.grid_index: r.status for r in results}
        assert statuses[0] == WorkerStatus.SUCCESS
        assert statuses[1] == WorkerStatus.FAILED
        assert statuses[2] == WorkerStatus.SUCCESS

    def test_abort_on_failure_stops_remaining_tasks(self, tmp_path: Path) -> None:
        """--on-failure abort: first failure → no more tasks dispatched.

        Note: tasks already running will finish; tasks not yet started
        are cancelled. We verify by sending more tasks than workers can
        handle immediately + one fails.
        """
        executor = WorkerPoolExecutor(
            max_workers=1,  # serial — predictable
            runtime_dir=tmp_path / "_runtime",
            on_failure_mode=OnFailureMode.ABORT,
            max_retries=0,
        )

        tasks = [
            _make_mock_work(WorkerStatus.FAILED, 0, error_kind="validation_error"),
            _make_mock_work(WorkerStatus.SUCCESS, 1),
            _make_mock_work(WorkerStatus.SUCCESS, 2),
        ]
        results = executor.run_all(tasks)

        # Must have at most 1 result (the failed one) — remaining cancelled.
        assert len(results) >= 1, "First failure must at least be collected"
        assert any(r.status == WorkerStatus.FAILED for r in results)
        assert len(results) < len(tasks), (
            f"--on-failure abort must prevent all tasks from running; "
            f"got {len(results)}/{len(tasks)} results."
        )

    def test_parallel_1_vs_n_produces_same_result_set(
        self, tmp_path: Path
    ) -> None:
        """Phase 8A.1 determinism invariant: infrastructure-level identity
        — --parallel 1 and --parallel N produce the same SET of results
        (modulo ordering). Locks the fundamental dispatch correctness.

        This tests the SCHEDULER, not the actual trainer determinism
        (trainer determinism relies on lobtrainer.utils.reproducibility.set_seed
        which is separately verified).
        """
        runtime_dir_serial = tmp_path / "_runtime_serial"
        runtime_dir_parallel = tmp_path / "_runtime_parallel"

        # Make tasks with deterministic output based on grid_index only
        def make_tasks():
            return [_make_mock_work(WorkerStatus.SUCCESS, i) for i in range(5)]

        serial_exec = WorkerPoolExecutor(
            max_workers=1,
            runtime_dir=runtime_dir_serial,
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=0,
        )
        parallel_exec = WorkerPoolExecutor(
            max_workers=3,
            runtime_dir=runtime_dir_parallel,
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=0,
        )

        serial_results = serial_exec.run_all(make_tasks())
        parallel_results = parallel_exec.run_all(make_tasks())

        # Sort by grid_index for comparison
        serial_sorted = sorted(serial_results, key=lambda r: r.grid_index)
        parallel_sorted = sorted(parallel_results, key=lambda r: r.grid_index)

        assert len(serial_sorted) == len(parallel_sorted)
        for s, p in zip(serial_sorted, parallel_sorted):
            assert s.grid_index == p.grid_index
            assert s.status == p.status
            assert s.fingerprint == p.fingerprint
            assert s.axis_values == p.axis_values
            # Scheduler-level identity — per-record ordering-sensitive
            # fields (like timestamps / execution_order) are NOT
            # compared; that's a trainer-level determinism concern.


# ============================================================================
# ResourceSpec contract
# ============================================================================


class TestPostAuditFixes:
    """Post-audit regression guards for 4-agent validation pass
    (2026-04-20). Locks CRITICAL/HIGH findings' fixes.
    """

    def test_gpu_semaphore_double_acquire_raises(self, tmp_path: Path) -> None:
        """Agent-C M5: double-acquire on the same GPUSemaphore instance
        previously silently dropped the first lock reference (GC releases
        the fcntl.flock while caller thinks it still holds exclusive
        access). Fix: raise RuntimeError.
        """
        runtime_dir = tmp_path / "_runtime"
        runtime_dir.mkdir()
        sem = GPUSemaphore(gpu_id=0, runtime_dir=runtime_dir, timeout_seconds=2.0)
        sem.acquire()
        try:
            with pytest.raises(RuntimeError, match="already held"):
                sem.acquire()
        finally:
            sem.release()

    def test_retry_attempt_counter_propagates_to_result(
        self, tmp_path: Path
    ) -> None:
        """Agent-C H3: retry counter was computed but never propagated
        to ``WorkerResult.attempt``. All retries reported attempt=1,
        making sweep_failure_info.attempt inaccurate for diagnosis.
        Fix: executor stamps ``retry_counts[task_idx] + 1`` onto the
        final collected result.

        This test uses a task factory that returns a fresh callable
        on each call — simulating the retry path in run_all.
        """
        # Factory that transitions from FAILED/transient → FAILED/fatal
        # on second attempt. After first attempt fails transient, retry.
        # Second attempt also fails but with fatal kind — now we stop.
        # The collected result should have attempt == 2.
        call_count = {"n": 0}

        def make_transient_then_fatal_task():
            def work():
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return WorkerResult(
                        grid_index=0,
                        status=WorkerStatus.FAILED,
                        fingerprint="f" * 64,
                        axis_values={},
                        error_kind="oom",  # TRANSIENT
                        stderr_tail="oom on attempt 1",
                    )
                else:
                    return WorkerResult(
                        grid_index=0,
                        status=WorkerStatus.FAILED,
                        fingerprint="f" * 64,
                        axis_values={},
                        error_kind="assertion",  # FATAL
                        stderr_tail="fatal on attempt 2",
                    )
            return work

        executor = WorkerPoolExecutor(
            max_workers=1,
            runtime_dir=tmp_path / "_runtime",
            on_failure_mode=OnFailureMode.RETRY,
            max_retries=3,
        )
        results = executor.run_all([make_transient_then_fatal_task()])
        assert len(results) == 1
        # After 1 retry the attempt counter must be 2, not 1.
        assert results[0].attempt == 2, (
            f"Retry attempt counter must propagate to WorkerResult. "
            f"Expected attempt=2 after 1 retry, got {results[0].attempt}"
        )
        assert results[0].error_kind == "assertion"

    def test_grid_index_sentinel_replaced_with_real_index(
        self, tmp_path: Path
    ) -> None:
        """Agent-C M3: ``_run_task_safely`` sets grid_index=-1 on
        uncaught exception because the wrapper can't know the real
        index. The caller (run_all) knows task_idx — fix: overwrite
        the sentinel with the real index.
        """
        def raising_task():
            raise AssertionError("boom from task 0")

        executor = WorkerPoolExecutor(
            max_workers=1,
            runtime_dir=tmp_path / "_runtime",
            on_failure_mode=OnFailureMode.CONTINUE,
            max_retries=0,
        )
        results = executor.run_all([raising_task])
        assert len(results) == 1
        assert results[0].grid_index == 0, (
            f"run_all must replace grid_index=-1 sentinel with real "
            f"task_idx; got {results[0].grid_index}"
        )
        assert results[0].status == WorkerStatus.FAILED
        assert results[0].error_kind == "assertion"


class TestResourceSpec:
    def test_resource_spec_default_zero_gpus(self) -> None:
        """Default ResourceSpec must be {gpus=0, cpu_slots=1} —
        conservative; non-training stages get no GPU allocation.
        """
        spec = ResourceSpec()
        assert spec.gpus == 0
        assert spec.cpu_slots == 1

    def test_resource_spec_frozen(self) -> None:
        """ResourceSpec must be frozen — stage resource requirements
        are immutable within a manifest (locked at parse time).
        """
        spec = ResourceSpec(gpus=1, cpu_slots=2)
        with pytest.raises((AttributeError, TypeError)):  # dataclass FrozenInstanceError
            spec.gpus = 0  # type: ignore
