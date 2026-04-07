"""
Feature extraction stage runner.

Invokes the Rust feature-extractor-MBO-LOB binary (export_dataset) as a
subprocess. Supports skip_if_exists to reuse validated exports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)


class ExtractionRunner:
    """Runs feature extraction via cargo run --bin export_dataset."""

    @property
    def stage_name(self) -> str:
        return "extraction"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.extraction
        if not stage.config:
            errors.append("extraction.config is required")
        else:
            config_path = config.paths.resolve(stage.config)
            if not config_path.exists():
                errors.append(f"Extractor config not found: {config_path}")

        if not config.paths.extractor_dir.exists():
            errors.append(f"Extractor directory not found: {config.paths.extractor_dir}")

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.extraction
        result = StageResult(stage_name=self.stage_name)

        if stage.skip_if_exists and stage.output_dir:
            output_dir = config.paths.resolve(stage.output_dir)
            if output_dir.exists() and any(output_dir.glob("*_metadata.json")):
                result.status = StageStatus.SKIPPED
                result.output_dir = str(output_dir)
                return result

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = "dry-run: would run extraction"
            return result

        config_path = config.paths.resolve(stage.config)
        cmd = [
            "cargo", "run", "--release",
            "--bin", "export_dataset",
            "--features", "parallel",
            "--",
            "--config", str(config_path),
        ]

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.extractor_dir,
                verbose=config.verbose,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                if stage.output_dir:
                    result.output_dir = str(config.paths.resolve(stage.output_dir))
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"export_dataset exited with code {proc.returncode}"
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
        errors: List[str] = []
        stage = manifest.stages.extraction
        if not stage.output_dir:
            return errors

        output_dir = config.paths.resolve(stage.output_dir)
        if not output_dir.exists():
            errors.append(f"Extraction output directory not found: {output_dir}")
            return errors

        meta_files = sorted(output_dir.glob("*_metadata.json"))
        if not meta_files:
            errors.append(f"No metadata JSON files in {output_dir}")

        seq_files = sorted(output_dir.glob("*_sequences.npy"))
        if not seq_files:
            errors.append(f"No sequence .npy files in {output_dir}")

        label_files = sorted(output_dir.glob("*_labels.npy"))
        if not label_files:
            errors.append(f"No label .npy files in {output_dir}")

        return errors
