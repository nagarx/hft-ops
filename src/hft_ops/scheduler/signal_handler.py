"""Phase 8A.1 — SIGINT cascade + subprocess PID tracker.

Parent process installs a SIGINT handler that:
  1. Sets the ``WorkerPoolExecutor.cancel_event``.
  2. Iterates tracked subprocess PIDs and sends SIGTERM.
  3. After 5s grace, sends SIGKILL to any survivors.
  4. Restores the prior signal handler on exit.

Threads (not processes) execute the grid-point tasks, so we cannot use
``os.setsid()`` for per-worker process groups. Instead, each stage
runner that spawns a subprocess registers the subprocess PID in a
global tracker; on Ctrl-C we iterate all tracked PIDs.

The tracker also handles natural process exit: on normal completion
we call ``subprocess_pid_tracker.untrack(pid)`` so the global list
doesn't grow unbounded.

Scoped via ``install_scheduler_signal_handler(executor)``; context-
manager usage unwinds the prior handler on exit (important for pytest
compatibility — pytest installs its own SIGINT handler and we must
not clobber it across tests).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional, Set

logger = logging.getLogger(__name__)


__all__ = [
    "SubprocessPidTracker",
    "subprocess_pid_tracker",
    "install_scheduler_signal_handler",
    "SIGNAL_GRACE_SECONDS",
]

SIGNAL_GRACE_SECONDS: float = 5.0


class SubprocessPidTracker:
    """Thread-safe set of subprocess PIDs for SIGINT cascade.

    Stage runners that spawn long-running subprocesses (cargo, lobtrainer,
    lobbacktest) should track their PID at spawn and untrack at
    completion:

        import subprocess
        proc = subprocess.Popen(cmd)
        subprocess_pid_tracker.track(proc.pid)
        try:
            proc.wait()
        finally:
            subprocess_pid_tracker.untrack(proc.pid)

    The module-level singleton ``subprocess_pid_tracker`` is consumed by
    ``install_scheduler_signal_handler``'s SIGINT cascade.
    """

    def __init__(self) -> None:
        self._pids: Set[int] = set()
        self._lock = threading.Lock()

    def track(self, pid: int) -> None:
        with self._lock:
            self._pids.add(pid)

    def untrack(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(pid)

    def snapshot(self) -> Set[int]:
        with self._lock:
            return set(self._pids)


subprocess_pid_tracker = SubprocessPidTracker()


@contextmanager
def install_scheduler_signal_handler(
    executor,
    *,
    tracker: Optional[SubprocessPidTracker] = None,
) -> Iterator[None]:
    """Install a SIGINT cascade handler for the duration of the context.

    On Ctrl-C:
      1. Set ``executor.cancel_event`` (workers check between stages).
      2. SIGTERM every tracked subprocess PID.
      3. After ``SIGNAL_GRACE_SECONDS``, SIGKILL any survivors.
      4. Re-raise KeyboardInterrupt in the main thread so the CLI exits.

    The prior SIGINT handler is restored on context exit. Safe to
    nest with pytest's handler.

    Only installs when the current thread is the MAIN thread (signal
    handlers can only be installed from main thread). Silently no-ops
    otherwise — useful for test harnesses.
    """
    if tracker is None:
        tracker = subprocess_pid_tracker

    if threading.current_thread() is not threading.main_thread():
        # Non-main thread (e.g., pytest sub-thread) — cannot install
        # signal handler. Proceed without; cancel_event can still be
        # set manually by tests.
        yield
        return

    prior_handler = signal.getsignal(signal.SIGINT)

    def _cascade(signum: int, frame) -> None:
        logger.warning(
            "SIGINT received — cancelling executor + SIGTERM'ing %d "
            "tracked subprocesses",
            len(tracker.snapshot()),
        )
        executor.cancel_event.set()

        tracked = tracker.snapshot()
        for pid in tracked:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already dead
            except OSError as exc:
                logger.debug("SIGTERM to PID %d failed: %s", pid, exc)

        if tracked:
            time.sleep(SIGNAL_GRACE_SECONDS)
            survivors = [
                pid for pid in tracker.snapshot()
                if _pid_alive(pid)
            ]
            for pid in survivors:
                logger.warning("PID %d survived SIGTERM; sending SIGKILL", pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        # Re-raise so the CLI's own KeyboardInterrupt handler (if any)
        # fires and the process exits cleanly.
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _cascade)
        yield
    finally:
        signal.signal(signal.SIGINT, prior_handler)


def _pid_alive(pid: int) -> bool:
    """Returns True if the PID is still running.

    ``os.kill(pid, 0)`` sends signal 0 which is a no-op but error-checks
    process existence.
    """
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
