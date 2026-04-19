"""
Validation stage runner — Rule-13 pre-training IC gate.

This runner calls ``hft_evaluator.fast_gate.run_fast_gate`` **as a library**
(not a subprocess). Phase 2b architectural decision: library-import was
chosen over subprocess for:

- Lower latency (no interpreter startup / module import per call)
- Direct exception propagation (no stdout/stderr scraping)
- Simpler test harness (no subprocess mocking)
- Richer in-memory report (dataclasses vs. JSON round-trip)
- No reliance on evaluator CLI stability (which is a public surface for
  humans, not an ABI for hft-ops)

The evaluator ``evaluate`` CLI remains the interface for humans; this
runner bypasses it intentionally.

The gate runs BETWEEN ``dataset_analysis`` and ``training``. On failure,
the ``on_fail`` policy selects the disposition:

- ``warn`` (DEFAULT): log warning, record gate_report in ledger, proceed.
- ``abort``: raise StageFailure → pipeline stops → ledger record saved
  with ``status: failed`` and the full gate_report attached.
- ``record_only``: always pass; gate verdict is informational only.

Rationale for warn-default: evaluator CLAUDE.md §Known Limitations
explicitly warns against using DISCARD / IC-based gates as hard filters.
Context, interaction, and early-timestep-only features produce zero
pre-training IC but carry model-attention value. The gate SURFACES
failures for researcher review; it does not silently block valid
experiments.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import StageResult, StageStatus

logger = logging.getLogger(__name__)


class ValidationRunner:
    """Runs the Rule-13 pre-training IC gate (fast_gate)."""

    @property
    def stage_name(self) -> str:
        return "validation"

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.validation

        if stage.on_fail not in ("warn", "abort", "record_only"):
            errors.append(
                f"stages.validation.on_fail must be 'warn' | 'abort' | "
                f"'record_only'; got {stage.on_fail!r}"
            )

        # The gate reads the extractor's output_dir to access sequences.
        # Either extraction is enabled (will produce output) OR output_dir
        # already exists.
        if not manifest.stages.extraction.output_dir:
            errors.append(
                "stages.extraction.output_dir must be set so validation "
                "knows where to load sequences from."
            )

        if stage.min_ic <= 0:
            errors.append(
                f"stages.validation.min_ic must be > 0; got {stage.min_ic}"
            )

        return errors

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.validation
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = "dry-run: would run IC gate"
            return result

        # Resolve data dir + horizon
        try:
            data_dir = config.paths.resolve(manifest.stages.extraction.output_dir)
        except Exception as exc:
            result.status = StageStatus.FAILED
            result.error_message = (
                f"Cannot resolve extraction.output_dir: {exc}"
            )
            return result

        if not data_dir.exists():
            result.status = StageStatus.FAILED
            result.error_message = (
                f"Export directory does not exist yet: {data_dir}. "
                f"Enable stages.extraction or point validation.data_dir "
                f"to an existing export."
            )
            return result

        horizon_idx = _resolve_horizon_idx_for_validation(manifest, data_dir)
        if horizon_idx is None:
            result.status = StageStatus.FAILED
            result.error_message = (
                "Cannot resolve horizon_idx for validation. Set either "
                "training.horizon_value or validation.target_horizon."
            )
            return result

        # Resolve output dir (default to runs/<experiment>/validation/)
        output_dir = stage.output_dir
        if not output_dir:
            output_dir_path = (
                config.paths.runs_dir
                / manifest.experiment.name
                / "validation"
            )
        else:
            output_dir_path = config.paths.resolve(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Resolve profile_ref if given
        profile_ref_path = None
        if stage.profile_ref:
            profile_ref_path = config.paths.resolve(stage.profile_ref)

        start = time.monotonic()
        gate_report: Dict[str, Any]
        try:
            # Import lazily so the evaluator is a soft dependency
            # (keeps hft-ops installable without hft-feature-evaluator in
            # narrow CI environments).
            from hft_evaluator.fast_gate import (  # noqa: WPS433
                GateThresholds,
                run_fast_gate,
            )

            thresholds = GateThresholds(
                min_ic=stage.min_ic,
                min_ic_count=stage.min_ic_count,
                min_return_std_bps=stage.min_return_std_bps,
                min_stability=stage.min_stability,
            )

            report = run_fast_gate(
                data_dir=data_dir,
                horizon_idx=horizon_idx,
                split="train",
                horizon_value=manifest.stages.training.horizon_value,
                thresholds=thresholds,
                sample_size=stage.sample_size,
                n_folds=stage.n_folds,
                allow_zero_ic_names=tuple(stage.allow_zero_ic_names),
                profile_ref=profile_ref_path,
            )

            # Write the full report to disk
            report_path = output_dir_path / "gate_report.json"
            report.to_json(report_path)

            gate_report = report.as_dict()
            result.duration_seconds = time.monotonic() - start
            result.output_dir = str(output_dir_path)

        except ImportError as exc:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = (
                "hft-feature-evaluator not installed; install the evaluator "
                f"or set stages.validation.enabled=false. ({exc})"
            )
            return result
        except Exception as exc:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = f"fast_gate failed: {exc}"
            logger.exception("fast_gate raised an unexpected exception")
            return result

        # Persist the verdict + key metrics on the stage result for the
        # orchestrator / ledger to pick up. The full report is on disk.
        result.captured_metrics["validation_verdict"] = gate_report["verdict"]
        result.captured_metrics["best_feature_ic"] = gate_report[
            "best_feature_ic"
        ]
        result.captured_metrics["best_feature_name"] = gate_report[
            "best_feature_name"
        ]
        result.captured_metrics["ic_count"] = gate_report["ic_count"]
        result.captured_metrics["return_std_bps"] = gate_report[
            "return_std_bps"
        ]
        result.captured_metrics["stability"] = gate_report["stability"]
        result.captured_metrics["n_folds_used"] = gate_report["n_folds_used"]
        result.captured_metrics["gate_report_path"] = str(
            output_dir_path / "gate_report.json"
        )
        # Embed the full serialized report so the ledger can store it
        # without a second disk read. Phase 7 Stage 7.4 Round 4
        # (2026-04-20): renamed "validation_report" → "gate_report" for
        # uniform cross-stage harvesting in cli.py::_record_experiment.
        # Phase 7 Stage 7.4 Round 5 (2026-04-20): inject a ``status``
        # field (lower-case verdict) so the dict conforms to the
        # ``hft_contracts.gate_report.GateReportDict`` convention.
        # ``fast_gate.GateReport`` historically used ``verdict: "PASS"|"FAIL"``
        # (upper-case); post_training_gate uses ``status: "pass"|"warn"|"abort"``
        # (lower-case). Unifying at the ADAPTER layer avoids breaking
        # fast_gate's public ``verdict`` field (preserved intact below)
        # while giving the ledger consumer one uniform key.
        # The legacy aliases ("gate_report_path", "best_feature", etc.)
        # above remain as top-level scalars for backward compatibility
        # with consumers that expected flat access.
        if "verdict" in gate_report and "status" not in gate_report:
            gate_report["status"] = str(gate_report["verdict"]).lower()
        result.captured_metrics["gate_report"] = gate_report

        # Disposition: apply on_fail policy
        verdict_pass = gate_report["verdict"] == "PASS"
        if verdict_pass:
            result.status = StageStatus.COMPLETED
        else:
            if stage.on_fail == "abort":
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"IC gate FAILED: {gate_report['reason']}. "
                    f"on_fail=abort → pipeline stops."
                )
            elif stage.on_fail == "warn":
                # Pipeline continues, but surface the warning prominently.
                result.status = StageStatus.COMPLETED
                result.error_message = (
                    f"[WARN] IC gate FAILED but on_fail=warn, continuing: "
                    f"{gate_report['reason']}"
                )
                logger.warning(
                    "IC gate FAILED for experiment %s: %s",
                    manifest.experiment.name,
                    gate_report["reason"],
                )
            elif stage.on_fail == "record_only":
                # Always pass; verdict remains in captured_metrics for review.
                result.status = StageStatus.COMPLETED
            else:
                # Loader should already have rejected this; defensive.
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"Unknown on_fail policy: {stage.on_fail!r}"
                )

        return result

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------
    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.validation

        output_dir = stage.output_dir
        if output_dir:
            output_dir_path = config.paths.resolve(output_dir)
        else:
            output_dir_path = (
                config.paths.runs_dir
                / manifest.experiment.name
                / "validation"
            )

        if not output_dir_path.exists():
            errors.append(
                f"Validation output directory not produced: {output_dir_path}"
            )
            return errors

        report_path = output_dir_path / "gate_report.json"
        if not report_path.exists():
            errors.append(f"gate_report.json not produced: {report_path}")

        return errors


def _resolve_horizon_idx_for_validation(
    manifest: ExperimentManifest,
    data_dir: Path,
) -> int | None:
    """Resolve horizon_idx for the gate from manifest or export metadata.

    Priority:
    1. If ``validation.target_horizon`` is set and numeric-like, treat it as
       a horizon VALUE and look up its index in the export.
    2. If backtesting.horizon_idx is set (via ``apply_resolved_context``),
       use it directly.
    3. If training.horizon_value is set, resolve via export metadata.
    """
    stage = manifest.stages.validation

    # Load export metadata for horizon lookup
    metadata_files = sorted(data_dir.glob("**/*_metadata.json"))
    horizons: List[int] = []
    if metadata_files:
        try:
            with open(metadata_files[0]) as f:
                md = json.load(f)
            horizons = list(md.get("horizons", [])) or list(
                md.get("max_horizons", [])
            )
            if not horizons:
                labeling = md.get("labeling", {})
                horizons = list(labeling.get("horizons", []))
        except (json.JSONDecodeError, OSError):
            horizons = []

    # (1) target_horizon explicit
    if stage.target_horizon:
        th = stage.target_horizon.strip().lstrip("Hh")
        try:
            value = int(th)
            if value in horizons:
                return horizons.index(value)
            # Could also be an index already
            if 0 <= value < max(len(horizons), 1000) and value < len(horizons):
                return value
        except ValueError:
            pass

    # (2) backtesting.horizon_idx (populated by orchestrator)
    if manifest.stages.backtesting.horizon_idx is not None:
        return manifest.stages.backtesting.horizon_idx

    # (3) training.horizon_value
    hv = manifest.stages.training.horizon_value
    if hv is not None and horizons and hv in horizons:
        return horizons.index(hv)

    # Single-horizon exports default to index 0
    if len(horizons) == 1:
        return 0

    return None
