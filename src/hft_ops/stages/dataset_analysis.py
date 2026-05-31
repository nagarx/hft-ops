"""
Dataset analysis stage runner.

Invokes lob-dataset-analyzer/scripts/run_analysis.py with the specified
profile, split, and data directory.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    _format_subprocess_failure,
    run_subprocess,
    _tail,
)

_logger = logging.getLogger(__name__)


def _summarize_dataset_health(stage: Any, config: OpsConfig) -> Dict[str, Any]:
    """Schema-light, FAIL-SOFT dataset-analysis summary for
    ``record.dataset_health``: the resolved report dir + the requested
    analyzers/profile/split. NOT per-analyzer scalars — run_analysis.py writes N
    idx-prefixed per-analyzer JSONs (``{idx:02d}_{snake}.json``) with no stable
    aggregate, and harvesting specific health scalars (NaN fractions, kurtosis, ...)
    would couple to 47 analyzer ``to_dict()`` schemas + a run-config-dependent
    filename set — a deliberate follow-on needing a pinned analyzer-output contract.

    Deliberately NO report-file COUNT: run_analysis.py never clears its output dir
    (``mkdir(exist_ok=True)``) and ALL manifests share the default ``outputs/analysis``,
    so a glob count would OVER-report (a 5-analyzer ``quick`` run after a 47-analyzer
    ``full`` run reads back 47 stale files) — a misleading scalar. The dir + config
    make the stage non-dark on the record (it links to its reports) without that
    hazard. Observation-tier: never raises.
    """
    try:
        if stage.output_dir:
            report_dir = config.paths.resolve(stage.output_dir)
        else:
            # run_analysis.py default is outputs/analysis relative to its cwd,
            # and DatasetAnalysisRunner.run runs it with cwd=dataset_analyzer_dir.
            report_dir = config.paths.dataset_analyzer_dir / "outputs" / "analysis"
        out: Dict[str, Any] = {"report_dir": str(report_dir)}
        if stage.analyzers:
            out["analyzers"] = list(stage.analyzers)
        elif stage.profile:
            out["profile"] = stage.profile
        if stage.split:
            out["split"] = stage.split
        return out
    except Exception as exc:  # noqa: BLE001 — fail-soft observation tier
        _logger.warning("dataset-health summary failed (dataset_health empty): %s", exc)
        return {}


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
                # Step 6 (2026-05-31): surface a dataset-health summary so the
                # stage is no longer dark on the record (record.dataset_health was
                # NEVER populated). Schema-light + FAIL-SOFT (report dir + config,
                # not per-analyzer scalars). Harvested by cli._record_experiment.
                result.captured_metrics["dataset_health"] = _summarize_dataset_health(
                    stage, config
                )
            else:
                result.status = StageStatus.FAILED
                # Phase α-2 / #PY-80 (2026-05-10) — surface stderr.
                result.error_message = _format_subprocess_failure(proc, "lob-dataset-analyzer")
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
