"""
Stage runner protocol and result types.

Every pipeline stage (extraction, analysis, training, backtesting) implements
the StageRunner protocol. Stages are thin wrappers that invoke module CLIs
as subprocesses, capturing output and validating results.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest


class StageStatus(str, Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class StageResult:
    """Structured result from a stage execution.

    Attributes:
        stage_name: Name of the stage (extraction, training, etc.).
        status: Final execution status.
        duration_seconds: Wall-clock time for execution.
        output_dir: Path to stage output directory, if any.
        captured_metrics: Key metrics captured from output.
        stdout: Captured stdout (last N lines).
        stderr: Captured stderr (last N lines).
        error_message: Human-readable error if status == FAILED.
    """

    stage_name: str
    status: StageStatus = StageStatus.PENDING
    duration_seconds: float = 0.0
    output_dir: str = ""
    captured_metrics: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


class StageRunner(Protocol):
    """Protocol for pipeline stage runners.

    Each stage runner:
    1. Validates inputs before execution
    2. Runs the module as a subprocess
    3. Validates outputs after execution
    4. Returns a structured StageResult
    """

    @property
    def stage_name(self) -> str:
        """Human-readable stage name."""
        ...

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        """Return list of validation errors, empty if valid."""
        ...

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        """Execute the stage. Returns structured result."""
        ...

    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        """Validate stage outputs exist and are correct. Returns errors."""
        ...


_MAX_CAPTURED_LINES = 200


def run_subprocess(
    cmd: List[str],
    *,
    cwd: Path,
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with optional streaming output.

    Always injects ``HFT_OPS_ORCHESTRATED=1`` into the subprocess env so that
    bypass-detection in trainer/backtester scripts (Phase 1.4 deprecation
    warnings) suppresses the warning when the script runs under hft-ops.

    Phase 8A.1 Part 1 post-audit (agent-C H4 fix): routes through
    ``subprocess.Popen`` + ``subprocess_pid_tracker`` so the scheduler
    SIGINT cascade can SIGTERM cargo/lobtrainer/lobbacktest subprocesses
    on Ctrl-C. Previously ``subprocess.run`` was opaque to the tracker
    and the SIGINT cascade would find an empty tracker set (iterating
    over zero PIDs — silent no-op). This helper preserves the original
    API signature but tracks the Popen PID for the duration of the call.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        verbose: If True, stream output to console.
        env: Additional environment variable overrides (merged on top of the
            current process env + the HFT_OPS_ORCHESTRATED marker).
        timeout_seconds: Maximum execution time.

    Returns:
        CompletedProcess with captured stdout and stderr.

    Raises:
        subprocess.TimeoutExpired: If timeout is exceeded.
    """
    import os
    from hft_ops.scheduler.signal_handler import subprocess_pid_tracker

    # Always inject the orchestrated marker so trainer scripts (and any other
    # bypass-detecting scripts) know they are running under hft-ops.
    full_env = {**os.environ, "HFT_OPS_ORCHESTRATED": "1"}
    if env is not None:
        full_env.update(env)

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "text": True,
        "env": full_env,
    }
    if verbose:
        # Stream to parent's stdout/stderr — no capture.
        popen_kwargs["stdout"] = None
        popen_kwargs["stderr"] = None
    else:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE

    # Phase 8A.1 Part 1 post-audit: track PID for SIGINT cascade.
    proc = subprocess.Popen(cmd, **popen_kwargs)
    subprocess_pid_tracker.track(proc.pid)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise
    finally:
        subprocess_pid_tracker.untrack(proc.pid)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _tail(text: str, max_lines: int = _MAX_CAPTURED_LINES) -> str:
    """Keep only the last max_lines of text."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


_DEFAULT_STDERR_TAIL_LINES = 20


def _format_subprocess_failure(
    proc: subprocess.CompletedProcess,
    script_basename: str,
    *,
    stderr_tail_lines: int = _DEFAULT_STDERR_TAIL_LINES,
) -> str:
    """Build an actionable error message from a failed subprocess.

    Phase α-2 / #PY-80 (2026-05-10) — closes "argparse error in 0.1s with
    no diagnostic" class of bugs across all 6 stage runners.

    Augments the generic ``f"{script_basename} exited with code {N}"`` pattern
    by tailing the last ``stderr_tail_lines`` of stderr — surfaces argparse
    errors, Python tracebacks, and missing-dependency messages that previously
    were captured into ``result.stderr`` but never displayed by the
    orchestrator's main loop (cli.py only prints ``result.error_message``).

    Per hft-rules §8 ("never silently drop, clamp, or 'fix' data without
    recording diagnostics"): pre-#PY-80, a 0.1s argparse exit produced a
    generic message and the stderr "the following arguments are required:
    --checkpoint" was buried in ``result.stderr`` (truncated by ``_tail``)
    and never displayed.

    Args:
        proc: Completed subprocess result with ``returncode`` and ``stderr``.
        script_basename: Display name for the failed script (e.g.,
            ``"export_signals.py"``, ``"train.py"``).
        stderr_tail_lines: Number of trailing stderr lines to include
            (default: 20). Bounded so the resulting message stays digestible
            in the orchestrator UI; full stderr remains in ``result.stderr``.

    Returns:
        Multi-line string: the generic exit-code line, optionally followed
        by ``--- last N stderr lines ---`` and the tail. If stderr is empty,
        returns only the generic line (no fanciness).
    """
    base = f"{script_basename} exited with code {proc.returncode}"
    stderr = (proc.stderr or "").strip()
    if not stderr:
        return base
    lines = stderr.splitlines()[-stderr_tail_lines:]
    tail = "\n".join(lines)
    return f"{base}\n--- last {len(lines)} stderr lines ---\n{tail}"
