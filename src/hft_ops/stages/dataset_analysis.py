"""
Dataset analysis stage runner.

Invokes lob-dataset-analyzer/scripts/run_analysis.py with the specified
profile, split, and data directory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)


class DatasetAnalysisRunner:
    """Runs lob-dataset-analyzer via its CLI."""

    @property
    def stage_name(self) -> str:
        return "dataset_analysis"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.dataset_analysis

        analyzer_dir = config.paths.dataset_analyzer_dir
        if not analyzer_dir.exists():
            errors.append(
                f"lob-dataset-analyzer directory not found: {analyzer_dir}"
            )

        script = analyzer_dir / "scripts" / "run_analysis.py"
        if not script.exists():
            errors.append(f"run_analysis.py not found: {script}")

        data_dir = stage.data_dir or manifest.stages.extraction.output_dir
        if not data_dir:
            errors.append(
                "dataset_analysis.data_dir is required when "
                "extraction.output_dir is not set"
            )

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.dataset_analysis
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = "dry-run: would run dataset analysis"
            return result

        data_dir = stage.data_dir or manifest.stages.extraction.output_dir
        if data_dir:
            data_dir = str(config.paths.resolve(data_dir))

        script = config.paths.dataset_analyzer_dir / "scripts" / "run_analysis.py"
        cmd = [
            sys.executable, str(script),
            "--data-dir", data_dir,
        ]

        if stage.analyzers:
            cmd.extend(["--analyzers", ",".join(stage.analyzers)])
        elif stage.profile:
            cmd.extend(["--profile", stage.profile])

        if stage.split:
            cmd.extend(["--split", stage.split])

        if stage.output_dir:
            out_dir = str(config.paths.resolve(stage.output_dir))
            cmd.extend(["--output-dir", out_dir])

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.dataset_analyzer_dir,
                verbose=config.verbose,
                env=config.env_overrides or None,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"lob-dataset-analyzer exited with code {proc.returncode}"
                )
        except Exception as e:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        return result

    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        return []
