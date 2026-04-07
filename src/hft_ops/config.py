"""
Global hft-ops configuration.

Provides a single OpsConfig that holds the pipeline paths and global settings.
Constructed once at CLI entry and threaded through all operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hft_ops.paths import PipelinePaths


@dataclass(frozen=True)
class OpsConfig:
    """Top-level configuration for an hft-ops session.

    Args:
        paths: Resolved pipeline paths.
        verbose: Enable verbose subprocess output.
        dry_run: Validate without executing stages.
    """

    paths: PipelinePaths
    verbose: bool = False
    dry_run: bool = False

    @classmethod
    def from_pipeline_root(
        cls,
        pipeline_root: Optional[Path] = None,
        *,
        verbose: bool = False,
        dry_run: bool = False,
    ) -> OpsConfig:
        """Build config from an explicit or auto-detected pipeline root."""
        if pipeline_root is not None:
            paths = PipelinePaths(pipeline_root=Path(pipeline_root).resolve())
        else:
            paths = PipelinePaths.auto_detect()
        return cls(paths=paths, verbose=verbose, dry_run=dry_run)
