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
        cache_extraction: Phase 8A.0 — when True (default), extraction stage
            consults ``data/exports/_cache/`` for a content-addressed match
            before invoking the extractor subprocess. Cache miss falls
            through to normal extraction; post-success populates the cache.
            Set False via ``--no-cache-extraction`` to disable (e.g. for
            debugging a specific extractor invocation).
    """

    paths: PipelinePaths
    verbose: bool = False
    dry_run: bool = False
    cache_extraction: bool = True

    @classmethod
    def from_pipeline_root(
        cls,
        pipeline_root: Optional[Path] = None,
        *,
        verbose: bool = False,
        dry_run: bool = False,
        cache_extraction: bool = True,
    ) -> OpsConfig:
        """Build config from an explicit or auto-detected pipeline root."""
        if pipeline_root is not None:
            paths = PipelinePaths(pipeline_root=Path(pipeline_root).resolve())
        else:
            paths = PipelinePaths.auto_detect()
        return cls(
            paths=paths,
            verbose=verbose,
            dry_run=dry_run,
            cache_extraction=cache_extraction,
        )
