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

    kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "text": True,
        "timeout": timeout_seconds,
    }

    # Always inject the orchestrated marker so trainer scripts (and any other
    # bypass-detecting scripts) know they are running under hft-ops.
    full_env = {**os.environ, "HFT_OPS_ORCHESTRATED": "1"}
    if env is not None:
        full_env.update(env)
    kwargs["env"] = full_env

    if verbose:
        kwargs["stdout"] = None
        kwargs["stderr"] = None
    else:
        kwargs["capture_output"] = True

    return subprocess.run(cmd, **kwargs)


def _tail(text: str, max_lines: int = _MAX_CAPTURED_LINES) -> str:
    """Keep only the last max_lines of text."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)
