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


# Phase A cutoff (2026-04-23): the date lob-model-trainer's Phase A producer-path
# fix shipped. After this cutoff, EVERY ``signal_metadata.json`` exported by a
# healthy trainer venv should carry a non-null ``compatibility_fingerprint``.
# Absence post-cutoff is a real regression signal (producer-path bug, trainer
# venv missing hft_contracts, or similar). The harvester emits a WARN log so
# operators can detect silent regressions via log monitoring, without
# disrupting ledger ingestion (the harvested value is still None, graceful).
FINGERPRINT_REQUIRED_AFTER_ISO = "2026-04-23"


def _harvest_compatibility_fingerprint(
    output_dir: Optional[Path],
) -> Optional[str]:
    """Harvest ``compatibility_fingerprint`` from signal_metadata.json
    (Phase V.A.4, 2026-04-21; V.1.5 follow-up WARN-log, 2026-04-23).

    The trainer's Phase II exporter embeds a ``CompatibilityContract``
    block + its SHA-256 ``fingerprint`` into signal_metadata.json. This
    helper reads that fingerprint so hft-ops can attach it to the
    ``ExperimentRecord`` — giving the ledger a "verifiable trust column"
    that surfaces via ``hft-ops ledger list --compatibility-fp <hex>``.

    Best-effort: returns None on any failure (missing file, malformed
    JSON, absent field, non-64-hex value) rather than raising. Mirrors
    the contract of ``_harvest_feature_set_ref`` — the ExperimentRecord
    stores None gracefully in that case, matching the dataclass default.

    Observability (V.1.5 follow-up, closes SDR-2 from the 3rd-round data-flow
    audit — hft-rules §8 "Never silently drop, clamp, or fix data without
    recording diagnostics"):

      * ABSENT field (legacy pre-V.A.4 exporter — no ``compatibility_fingerprint``
        key anywhere in signal_metadata.json) → silent None (expected; not a
        drift signal, just pre-Phase-V records).
      * MALFORMED field (key present but value fails ``CONTENT_HASH_RE``
        — e.g. uppercase hex, wrong length, non-hex chars) → ``warnings.warn``
        citing the path + value, then silent None. This is a real drift signal:
        the producer wrote something, but we can't validate it.
      * Single WARN per unique (path, value) via stdlib default dedup —
        no spam on 100-record sweep scans.

    Searches for signal_metadata.json at ``<output_dir>/signal_metadata.json``
    AND ``<output_dir>/*/signal_metadata.json`` (split subdirs) — same
    dual-layout support as ``_harvest_feature_set_ref``.

    Returns:
        64-char lowercase hex string (SHA-256 digest) on success;
        None otherwise.
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
        if raw is None:
            # Phase A (2026-04-23): distinguish pre-cutoff manifests (legacy,
            # silent) from post-cutoff manifests (drift signal, WARN). The
            # ``exported_at`` field on signal_metadata.json is an ISO-8601
            # UTC timestamp; string comparison against the cutoff date works
            # because ISO-8601 strings sort lexicographically. A missing
            # ``exported_at`` is treated as pre-cutoff — conservative: we
            # don't warn on ambiguous data.
            exported_at = meta.get("exported_at", "")
            if isinstance(exported_at, str) and exported_at >= FINGERPRINT_REQUIRED_AFTER_ISO:
                logger.warning(
                    "harvest_compatibility_fingerprint: post-Phase-A manifest at "
                    "%s has no compatibility_fingerprint (exported_at=%s). "
                    "Possible producer-path regression — check trainer venv for "
                    "hft_contracts availability + "
                    "SignalExporter._build_compatibility_contract. Record will "
                    "be stored with compatibility_fingerprint=None.",
                    path, exported_at,
                )
            # Field absent — legacy manifest (pre-cutoff) or post-cutoff with
            # diagnostic WARN above. Either way, continue to the next candidate.
            continue
        if isinstance(raw, str) and CONTENT_HASH_RE.match(raw):
            return raw
        # Field present but malformed — V.1.5 SDR-2 WARN path (hft-rules §8).
        import warnings
        display = repr(raw) if not isinstance(raw, str) else repr(raw[:80])
        warnings.warn(
            f"harvest_compatibility_fingerprint: rejected malformed value "
            f"at {path} — expected 64-lowercase-hex SHA-256, got {display}. "
            f"Record will be stored with compatibility_fingerprint=None. "
            f"Inspect the trainer's signal-export step for producer drift.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Do not try other candidates once we saw a malformed value —
        # more likely than not all candidates share the same producer bug,
        # and further WARNs would be duplicates.
        return None

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


def _resolve_signal_export_config(
    stage: "SignalExportStage",  # forward ref to avoid TYPE_CHECKING churn
    manifest: "ExperimentManifest",
    config: "OpsConfig",
) -> Optional[Path]:
    """Resolve the trainer YAML path passed as `--config` to export_signals.py.

    Phase 7.5-A (2026-04-23) — closes Bug #2 of the Frame 5 Task 1 audit.
    The canonical `lob-model-trainer/scripts/export_signals.py` REQUIRES
    `--config <trainer_yaml_path>` to reconstruct the Trainer pipeline
    (normalization, feature selection, model instantiation, checkpoint
    loading). Prior SignalExportRunner passed `--experiment <name>` which
    `export_signals.py` rejects (argparse error).

    3-tier resolution cascade (see `SignalExportStage` docstring):

      **Priority 1**: ``stages.signal_export.config`` (explicit escape
      hatch). If set, take precedence over auto-resolved paths. Use case:
      operator wants signal_export to run against a DIFFERENT trainer
      config than training (e.g., different split, different calibration).

      **Priority 2**: ``<training.output_dir>/config.yaml`` (Phase 7.5-A+
      auto-persisted by train.py after training). Handles BOTH legacy
      wrapper manifests (`stages.training.config:`) AND Phase 1 wrapper-
      less manifests (`stages.training.trainer_config:` inline dict) —
      because train.py persists the effective config regardless of source
      shape.

      **Priority 3**: ``manifest.stages.training.config`` (legacy wrapper
      fallback). Used when Priority 2 file isn't present yet (e.g.,
      stages ran out of order, or training stage disabled + operator
      re-using an old checkpoint without persisted config).

    Returns:
        Absolute `Path` to the resolved trainer YAML when one of the 3
        tiers succeeds; ``None`` when ALL fail (caller MUST fail-loud
        with a StageResult(FAILED) + actionable error message).

    Rationale for returning None vs raising: consistent with
    ``SignalExportRunner.run()`` error-handling contract — all failures
    are surfaced as StageResult(FAILED) with structured messages, not
    as exceptions out of the stage subsystem.
    """
    # Priority 1 — explicit escape hatch
    if stage.config:
        resolved = config.paths.resolve(stage.config)
        if resolved.exists():
            return resolved
        # If operator explicitly set a path that doesn't exist, fail-loud:
        # returning None triggers caller's actionable error message which
        # cites all 3 tiers.
        return None

    # Priority 2 — auto-persisted by train.py after training
    training = manifest.stages.training
    if training.output_dir:
        candidate = config.paths.resolve(training.output_dir) / "config.yaml"
        if candidate.exists():
            return candidate

    # Priority 3 — legacy wrapper manifest path
    if training.config:
        resolved = config.paths.resolve(training.config)
        if resolved.exists():
            return resolved

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

        # Validate the configured script exists.
        #
        # V.1.5 Frame-5 Task-1c fix (2026-04-23): script path is PIPELINE-ROOT-
        # RELATIVE by convention (matches `extraction.config`, `data.data_dir`,
        # `stage.checkpoint` — all resolved via `config.paths.resolve()`).
        # Previous `trainer_dir / stage.script` produced DOUBLED prefix
        # (`/...lob-model-trainer/lob-model-trainer/scripts/...`) when manifests
        # used the canonical pipeline-root-relative `lob-model-trainer/scripts/...`
        # path. Bug had never surfaced because signal_export stage had never
        # been exercised live. Unified with `config.paths.resolve(stage.script)`
        # to match the pipeline-wide convention.
        script_path = config.paths.resolve(stage.script)
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

        # Phase 7.5-A (2026-04-23) — restrict to splits export_signals.py accepts.
        # The canonical `lob-model-trainer/scripts/export_signals.py` argparse
        # declares `--split choices=["val", "test"]` — "train" is NOT accepted.
        # Runner previously accepted "train" at manifest-validate time but the
        # subprocess would crash; now fail-loud at validate time.
        if stage.split not in ("val", "test"):
            errors.append(
                f"Invalid signal_export.split: {stage.split!r} "
                f"(expected 'val' | 'test'; export_signals.py does not "
                f"accept 'train' — signal export is a post-training artifact)"
            )

        # Phase 7.5-A — cross-check the 3-tier config resolver can succeed.
        # This is the manifest-load-time gate that fails BEFORE stage execution
        # when the operator has not configured a resolvable trainer config.
        # See `_resolve_signal_export_config` docstring for the cascade.
        #
        # Gated on `stage.enabled` so a disabled signal_export doesn't force
        # operators to populate an unused config slot (e.g., training-only
        # manifests should not require a signal_export config).
        #
        # Note: Priority 2 (`<training.output_dir>/config.yaml`) is checked
        # conditionally — if the training stage is ENABLED in the same manifest,
        # the file will be auto-persisted at run time by train.py even though
        # it doesn't exist yet. We only fail-loud when BOTH Priorities 1 and 3
        # are absent AND training stage is disabled (meaning Priority 2 will
        # not be auto-populated).
        if stage.enabled:
            has_priority_1 = bool(stage.config)
            has_priority_3 = bool(manifest.stages.training.config)
            training_will_populate_priority_2 = manifest.stages.training.enabled
            if not (
                has_priority_1
                or has_priority_3
                or training_will_populate_priority_2
            ):
                errors.append(
                    "signal_export stage cannot resolve trainer config: set one of "
                    "(a) stages.signal_export.config (explicit escape hatch), "
                    "(b) stages.training.config (legacy wrapper manifest path), "
                    "OR (c) enable stages.training (auto-persists config.yaml to "
                    "<training.output_dir>/config.yaml). All three are UNSET/disabled. "
                    "See SignalExportStage docstring (§Config resolution) for details."
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

        # V.1.5 Frame-5 Task-1c fix: see matching comment at validate-site above.
        script = config.paths.resolve(stage.script)

        # Phase 7.5-A (2026-04-23) — resolve trainer config for `--config` arg.
        # Prior runner passed `--experiment <name>` which is NOT accepted by
        # the canonical `export_signals.py` (argparse requires `--config <path>`).
        # Bug existed since Phase 6 6D archived the per-model export scripts
        # (`export_hmhp_signals.py` accepted `--experiment`); `export_signals.py`
        # was the unified replacement but the runner was never migrated.
        # 3-tier resolver (see `SignalExportStage` docstring for full details):
        resolved_config = _resolve_signal_export_config(stage, manifest, config)
        if resolved_config is None:
            result.status = StageStatus.FAILED
            result.error_message = (
                "SignalExportRunner cannot resolve trainer config path required "
                "by `export_signals.py --config <path>`. Tried (in order): "
                "(1) stages.signal_export.config (explicit escape hatch, UNSET); "
                f"(2) <training.output_dir>/config.yaml (auto-persisted by "
                f"train.py after Phase 7.5-A+; file not present); "
                "(3) stages.training.config (legacy wrapper manifest path; UNSET). "
                "Fix: set `stages.signal_export.config:` in the manifest OR set "
                "`stages.training.config:` (legacy wrapper pattern) OR ensure the "
                "training stage ran successfully to persist "
                "<training.output_dir>/config.yaml."
            )
            return result

        cmd = [sys.executable, str(script)]
        cmd.extend(["--config", str(resolved_config)])

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
