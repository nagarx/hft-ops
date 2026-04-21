"""
Signal export stage runner.

Invokes a trainer-side signal export script (e.g., ``export_signals.py``,
``export_hmhp_signals.py``) to materialize trained-model predictions from
a checkpoint into a ``signals/`` directory that the backtester consumes.

Runs BETWEEN training and backtesting. If disabled, backtesting stage
must reuse pre-existing signals or run its own export.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)

# Phase 6 6A.9 (2026-04-17) + post-validation DRY (2026-04-18) + REV 2
# pre-push (2026-04-20): producer-consumer consistency — harvester applies
# the SAME content_hash format gate as the backtester-facing
# `hft_contracts.signal_manifest` (which defines the canonical regex).
# Importing from the SSoT module eliminates the parallel inline definition
# that existed in Phase 6 6A.9 and prevents producer/consumer regex drift.
# REV 2 renamed the module-private `_CONTENT_HASH_RE` to the public
# `CONTENT_HASH_RE`; `_CONTENT_HASH_RE` is kept as an alias through
# 2026-10-31. Contract source: pipeline_contract.toml:1211 (64-lowercase-
# hex pattern) + hft_contracts.canonical_hash.sha256_hex output format.
from hft_contracts.signal_manifest import CONTENT_HASH_RE

logger = logging.getLogger(__name__)


def _harvest_compatibility_fingerprint(
    output_dir: Optional[Path],
) -> Optional[str]:
    """Harvest ``compatibility_fingerprint`` from signal_metadata.json
    (Phase V.A.4, 2026-04-21).

    The trainer's Phase II exporter embeds a ``CompatibilityContract``
    block + its SHA-256 ``fingerprint`` into signal_metadata.json. This
    helper reads that fingerprint so hft-ops can attach it to the
    ``ExperimentRecord`` — giving the ledger a "verifiable trust column"
    that surfaces via ``hft-ops ledger list --compatibility-fp <hex>``.

    Best-effort: returns None on any failure (missing file, malformed
    JSON, absent field, non-64-hex value) rather than raising. Mirrors
    the contract of ``_harvest_feature_set_ref`` — the ExperimentRecord
    stores None gracefully in that case, matching the dataclass default.

    Searches for signal_metadata.json at ``<output_dir>/signal_metadata.json``
    AND ``<output_dir>/*/signal_metadata.json`` (split subdirs) — same
    dual-layout support as ``_harvest_feature_set_ref``.

    Returns:
        64-char lowercase hex string (SHA-256 digest) on success;
        None otherwise. Value is validated against
        ``hft_contracts.signal_manifest.CONTENT_HASH_RE`` before return —
        malformed values are silently dropped (fail-loud at the harvest
        boundary; same defense-in-depth gate as feature_set_ref harvest).
    """
    if output_dir is None or not output_dir.exists():
        return None

    # Direct and one-level-deep search (covers both `signals/` and
    # `signals/<split>/` layouts). Same pattern as _harvest_feature_set_ref.
    candidates = [output_dir / "signal_metadata.json"]
    try:
        for sub in output_dir.iterdir():
            if sub.is_dir():
                candidates.append(sub / "signal_metadata.json")
    except OSError:
        return None

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        # The Phase II exporter emits a top-level `compatibility_fingerprint`
        # alongside the `compatibility` block. Accept both locations for
        # forward-compat: future exporters may nest fingerprint under
        # compatibility.{fingerprint,} — check there too.
        raw = meta.get("compatibility_fingerprint")
        if raw is None and isinstance(meta.get("compatibility"), dict):
            raw = meta["compatibility"].get("fingerprint")
        if isinstance(raw, str) and CONTENT_HASH_RE.match(raw):
            return raw

    return None


def _harvest_feature_set_ref(
    output_dir: Optional[Path],
) -> Optional[Dict[str, str]]:
    """Harvest `feature_set_ref` from signal_metadata.json (Phase 4 Batch 4c.4).

    Best-effort: returns None on any failure (missing file, malformed JSON,
    absent field) rather than raising, because signal-export may succeed
    even when feature_set_ref cannot be harvested (e.g., legacy XGBoost
    script bypass writes a different metadata format). The ExperimentRecord
    gracefully stores None in that case, matching the dataclass default.

    Searches for signal_metadata.json at ``<output_dir>/signal_metadata.json``
    AND ``<output_dir>/*/signal_metadata.json`` (split subdirs) — SignalExporter
    may write to either shape depending on the trainer's export config.

    Returns:
        ``{"name": str, "content_hash": str}`` dict on success; None otherwise.
    """
    if output_dir is None or not output_dir.exists():
        return None

    # Direct and one-level-deep search (covers both `signals/` and
    # `signals/<split>/` layouts). No recursion beyond that — performance
    # + avoids unrelated signal_metadata.json from sibling runs.
    candidates = [output_dir / "signal_metadata.json"]
    try:
        for sub in output_dir.iterdir():
            if sub.is_dir():
                candidates.append(sub / "signal_metadata.json")
    except OSError:
        return None

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        raw = meta.get("feature_set_ref") if isinstance(meta, dict) else None
        if isinstance(raw, dict):
            name = raw.get("name")
            content_hash = raw.get("content_hash")
            # Phase 6 6A.9 hardening: validate content_hash format matches
            # backtester's gate. Malformed refs are silently dropped rather
            # than poisoning the ledger record (backtester would later reject
            # what hft-ops accepted — producer/consumer asymmetry).
            if (
                isinstance(name, str)
                and isinstance(content_hash, str)
                and CONTENT_HASH_RE.match(content_hash)
            ):
                return {"name": name, "content_hash": content_hash}

    return None


class SignalExportRunner:
    """Runs trainer-side signal export before backtesting."""

    @property
    def stage_name(self) -> str:
        return "signal_export"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.signal_export

        trainer_dir = config.paths.trainer_dir
        if not trainer_dir.exists():
            errors.append(f"Trainer directory not found: {trainer_dir}")
            return errors

        # Validate the configured script exists
        script_path = trainer_dir / stage.script
        if not script_path.exists():
            errors.append(
                f"Signal export script not found: {script_path} "
                f"(configured via stages.signal_export.script='{stage.script}')"
            )

        # Checkpoint path may reference a variable that resolves at runtime;
        # only validate if it's a concrete path AND training is disabled
        # (i.e., we depend on a pre-existing checkpoint).
        if stage.checkpoint and "${" not in stage.checkpoint:
            if not manifest.stages.training.enabled:
                checkpoint = config.paths.resolve(stage.checkpoint)
                if not checkpoint.exists():
                    errors.append(
                        f"Signal export checkpoint not found: {checkpoint}"
                    )

        if stage.split not in ("train", "val", "test"):
            errors.append(
                f"Invalid signal_export.split: {stage.split!r} "
                f"(expected 'train' | 'val' | 'test')"
            )

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.signal_export
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = (
                f"dry-run: would run signal export via {stage.script}"
            )
            return result

        script = config.paths.trainer_dir / stage.script

        cmd = [sys.executable, str(script)]

        cmd.extend(["--experiment", manifest.experiment.name])

        if stage.checkpoint:
            checkpoint = str(config.paths.resolve(stage.checkpoint))
            cmd.extend(["--checkpoint", checkpoint])

        cmd.extend(["--split", stage.split])

        if stage.output_dir:
            output_dir = str(config.paths.resolve(stage.output_dir))
            cmd.extend(["--output-dir", output_dir])

        cmd.extend(stage.extra_args)

        script_basename = Path(stage.script).name

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.trainer_dir,
                verbose=config.verbose,
                env=config.env_overrides or None,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                result.output_dir = stage.output_dir
                # Phase 4 Batch 4c.4 (2026-04-16): harvest feature_set_ref
                # from signal_metadata.json so cli.py::_record_experiment
                # can attach it to the ExperimentRecord. Best-effort: signal
                # metadata may be absent (e.g., first-run, non-standard
                # export path) — treat as None.
                resolved_output_dir = (
                    config.paths.resolve(stage.output_dir)
                    if stage.output_dir
                    else None
                )
                ref = _harvest_feature_set_ref(resolved_output_dir)
                if ref is not None:
                    result.captured_metrics["feature_set_ref"] = ref
                # Phase V.A.4 (2026-04-21): harvest compatibility_fingerprint
                # alongside feature_set_ref so cli.py::_record_experiment
                # can attach the "trust column" to the ExperimentRecord.
                # None on legacy signals / absent compatibility block.
                compat_fp = _harvest_compatibility_fingerprint(resolved_output_dir)
                if compat_fp is not None:
                    result.captured_metrics["compatibility_fingerprint"] = compat_fp
                # Phase V.1 L1.2 (2026-04-21): persist the resolved absolute
                # signal-export output_dir into captured_metrics so
                # cli.py::_record_experiment can attach it to the
                # ExperimentRecord. Closes Agent 2 H1 (manifest-move
                # resilience): future consumers (`hft-ops sweep compare`,
                # `hft-ops ledger show`) read this field directly instead of
                # re-resolving the manifest at query time.
                if resolved_output_dir is not None:
                    result.captured_metrics["signal_export_output_dir"] = str(
                        resolved_output_dir
                    )
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"{script_basename} exited with code {proc.returncode}"
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
        stage = manifest.stages.signal_export
        errors: List[str] = []

        if not stage.output_dir:
            return errors

        output_path = config.paths.resolve(stage.output_dir)
        if not output_path.exists():
            errors.append(f"Signal export output dir not produced: {output_path}")
            return errors

        # Heuristic: signals/ directory should contain at least one .npy file
        npy_files = list(output_path.glob("*.npy"))
        if not npy_files:
            errors.append(
                f"Signal export output missing .npy files: {output_path}"
            )

        return errors
