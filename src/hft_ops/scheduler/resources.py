"""Phase 8A.1 — Resource coordination primitives for parallel sweep execution.

Provides:
  - ``ResourceSpec``: per-stage resource requirements (gpus, cpu_slots).
  - ``GPUSemaphore``: filelock-backed exclusive-GPU lock that persists
    across worker crashes (fcntl.flock releases on FD close per POSIX).

Design rationale:
  - GPU memory is EXCLUSIVE (2 concurrent trainers on 1 GPU → OOM).
    Semaphore serializes access to each GPU independently.
  - CPU is preemptively scheduled by the OS. No runtime lock needed;
    parent computes per-worker thread budget and injects
    ``RAYON_NUM_THREADS`` / ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``
    env vars into the subprocess environment.

Fingerprint-stability invariant (Invariant 4): ``ResourceSpec`` NEVER
enters ``dedup.py::compute_fingerprint`` — resources are EXECUTION
POLICY, not TREATMENT. Running the same config with 1 GPU vs 2 GPUs
must produce the same fingerprint (reproducibility).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import filelock

logger = logging.getLogger(__name__)


__all__ = [
    "ResourceSpec",
    "GPUSemaphore",
]


# ---------------------------------------------------------------------------
# ResourceSpec — frozen per-stage declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSpec:
    """Resource requirements for a single pipeline stage.

    Frozen — stage resource requirements are locked at manifest-parse
    time and immutable thereafter.

    Args:
      gpus: Number of exclusive GPUs needed (0 for CPU-only stages like
        extraction / analysis / backtest; 1 for training / signal_export).
      cpu_slots: Nominal CPU-slot count for the stage. Parent divides
        ``cpu_budget`` (from ``--cpu-budget`` flag) across running workers
        weighted by their ``cpu_slots``. Subprocess thread counts
        (RAYON_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS) are
        injected accordingly. Default 1 = "one share of the CPU budget".
    """

    gpus: int = 0
    cpu_slots: int = 1


# ---------------------------------------------------------------------------
# GPUSemaphore — filelock-backed exclusive-GPU acquire/release
# ---------------------------------------------------------------------------


class GPUSemaphore:
    """Filelock-backed GPU semaphore. One lock file per GPU id.

    Guarantees AT MOST ONE worker holds a given GPU at a time —
    critical because Python-side multi-tenant GPU use (without MPS)
    silently OOMs the slower caller.

    Lock file lives under ``<runtime_dir>/gpu-<gpu_id>.lock``. The
    ``filelock`` package uses ``fcntl.flock`` on POSIX, which releases
    the lock automatically when the holding process dies / the lock
    file descriptor is closed (kernel-enforced). This means a
    SIGKILL'd worker CANNOT leak a GPU reservation — a cleaner
    alternative to lease timers or heartbeats.

    Usage:
        sem = GPUSemaphore(gpu_id=0, runtime_dir=ledger_dir / "_runtime",
                           timeout_seconds=1800.0)
        with sem:
            # Launch trainer subprocess pinned to GPU 0
            subprocess.run(["python", "train.py"], env={"CUDA_VISIBLE_DEVICES": "0"})

    Raises:
      filelock.Timeout: on acquire, if no permit is available within
        ``timeout_seconds`` (default 30 min — long enough to queue behind
        one in-progress training run; shorter prevents deadlock).
    """

    def __init__(
        self,
        gpu_id: int,
        runtime_dir: Path,
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.gpu_id = gpu_id
        self.runtime_dir = runtime_dir
        self.timeout_seconds = timeout_seconds
        self.lock_path = runtime_dir / f"gpu-{gpu_id}.lock"
        self._lock: Optional[filelock.FileLock] = None

    def acquire(self) -> None:
        """Acquire the GPU lock or raise filelock.Timeout."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._lock = filelock.FileLock(str(self.lock_path))
        self._lock.acquire(timeout=self.timeout_seconds)
        logger.debug("Acquired GPU %d lock (%s)", self.gpu_id, self.lock_path)

    def release(self) -> None:
        """Release the lock if held. Idempotent."""
        if self._lock is not None:
            try:
                self._lock.release()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "GPU %d lock release failed: %s", self.gpu_id, exc
                )
            finally:
                self._lock = None
                logger.debug("Released GPU %d lock", self.gpu_id)

    def __enter__(self) -> "GPUSemaphore":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
