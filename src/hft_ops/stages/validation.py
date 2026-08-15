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

⚠️ **THE GATE DOES NOT NECESSARILY SCORE THE ARRAY THE MODEL WILL FIT**
(2026-08-15, operator ruling R3). ``run_fast_gate`` reads the ON-DISK label
array (``{date}_labels.npy``, else ``{date}_regression_labels.npy`` — the
precedence in ``hft_evaluator.data.loader:150-162``). But the trainer DERIVES
labels at load time from ``forward_prices`` via ``LabelFactory`` whenever
``labels.source`` is ``forward_prices`` or ``auto`` and the export declares
``forward_prices.exported`` (``lobtrainer/data/dataset.py:687,700``), and
DISCARDS the on-disk array. Measured over the ledger: 132 of 189 records took
that path, and 88 carry a PASSING gate that scored a different dependent
variable than the model fitted.

Moving the gate BEHIND label derivation is **not implementable inside
hft-ops**: ``run_fast_gate(data_dir, horizon_idx, ...)`` accepts no label
array, only an export directory (signature at
``hft_evaluator/fast_gate.py:540-553``), and ``hft-feature-evaluator`` is
outside this round's editable surface. So this runner implements the
RECORDABLE half — it attaches, to every gate report, an identity for the
array it actually scored AND an identity for the array the model will fit,
plus an explicit divergence verdict.

**Divergence is recorded, NEVER blocked.** A deliberate label substitution IS
the experiment in some runs — the NX1 arc fitted ``twap_exit_return`` against
a smoothed on-disk label on purpose and produced the programme's most valuable
2026-08 output. Blocking it would have prevented that. Every step of the
identity computation degrades per-item to ``"unresolved"`` with a reason and
can never fail the stage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            errors.append(f"stages.validation.min_ic must be > 0; got {stage.min_ic}")

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
            result.error_message = f"Cannot resolve extraction.output_dir: {exc}"
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
                config.paths.runs_dir / manifest.experiment.name / "validation"
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
        result.captured_metrics["best_feature_ic"] = gate_report["best_feature_ic"]
        result.captured_metrics["best_feature_name"] = gate_report["best_feature_name"]
        result.captured_metrics["ic_count"] = gate_report["ic_count"]
        result.captured_metrics["return_std_bps"] = gate_report["return_std_bps"]
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

        # R3 (2026-08-15): record WHICH dependent variable this verdict is
        # about. Without this, a PASS is unattributable — 88 ledger records
        # carry one that scored an array the model never fitted. Attached to
        # the in-memory report AND re-written to disk so the artifact and the
        # ledger agree. Best-effort throughout; a failure here must not cost
        # the gate result that was already computed.
        try:
            label_identity = _build_label_identity(manifest, config, data_dir, "train")
            gate_report["label_identity"] = label_identity
            result.captured_metrics["label_divergence"] = label_identity["divergence"]
            result.captured_metrics["scored_label_content_hash"] = label_identity[
                "scored"
            ].get("content_hash")
            result.captured_metrics["fitted_label_strategy_hash"] = label_identity[
                "will_fit"
            ].get("label_strategy_hash")
            if label_identity["divergence"] == "derived_at_load":
                logger.warning(
                    "Gate for %s scored the ON-DISK label array, but the "
                    "trainer will DERIVE labels at load time (%s). The gate "
                    "verdict describes a different dependent variable than "
                    "the model will fit. Recorded, not blocked.",
                    manifest.experiment.name,
                    label_identity["divergence_detail"],
                )
            try:
                report_path.write_text(
                    json.dumps(gate_report, indent=2, sort_keys=True, default=str)
                    + "\n"
                )
            except OSError:
                logger.warning(
                    "gate_report.json written without label_identity (%s "
                    "not rewritable); the in-memory record still carries it.",
                    report_path,
                )
        except Exception:
            logger.warning(
                "label identity could not be recorded for %s; the gate "
                "verdict stands but is unattributed.",
                manifest.experiment.name,
                exc_info=True,
            )

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
                result.error_message = f"Unknown on_fail policy: {stage.on_fail!r}"

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
                config.paths.runs_dir / manifest.experiment.name / "validation"
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


# =============================================================================
# Label identity — what the gate scored vs. what the model will fit (R3)
# =============================================================================
#
# Every helper below is best-effort by construction. A missing file, an
# unreadable YAML, an uninstalled hft-contracts — each degrades to an
# ``"unresolved"`` marker carrying its own reason. NONE of them may raise:
# the identity block is an OBSERVATION attached to a gate report, and an
# observation that can abort the thing it observes is worse than no
# observation at all.

_DERIVING_SOURCES = ("forward_prices", "auto")


def _unresolved(reason: str) -> Dict[str, Any]:
    return {"resolved": False, "reason": reason}


def _scored_label_identity(data_dir: Path, split: str) -> Dict[str, Any]:
    """Identify the on-disk label array ``run_fast_gate`` actually read.

    Mirrors ``hft_evaluator.data.loader.ExportLoader.load_day`` precedence:
    ``{date}_labels.npy`` first, ``{date}_regression_labels.npy`` only as a
    fallback. That precedence is itself worth recording — an export carrying
    BOTH gets scored on the classification array, not the bps regression one.
    """
    try:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            return _unresolved(f"split dir absent: {split_dir}")

        from hft_contracts import hash_file

        picked: List[Path] = []
        kinds: set = set()
        for seq in sorted(split_dir.glob("*_sequences.npy")):
            date = seq.name[: -len("_sequences.npy")]
            cls = split_dir / f"{date}_labels.npy"
            reg = split_dir / f"{date}_regression_labels.npy"
            if cls.exists():
                picked.append(cls)
                kinds.add("labels.npy")
            elif reg.exists():
                picked.append(reg)
                kinds.add("regression_labels.npy")

        if not picked:
            return _unresolved(f"no label arrays under {split_dir}")

        # Content-address the exact files, in a deterministic order, via the
        # hft-contracts hashing SSoT (hft-rules §0 — never re-derive a hash).
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        per_file = {p.name: hash_file(p) for p in picked}
        return {
            "resolved": True,
            "origin": "on_disk_export",
            "file_kinds": sorted(kinds),
            "n_files": len(picked),
            "content_hash": sha256_hex(canonical_json_blob(per_file)),
        }
    except Exception as exc:  # never let an observation break the stage
        logger.debug("scored-label identity unresolved", exc_info=True)
        return _unresolved(f"{type(exc).__name__}: {exc}")


def _resolve_trainer_labels_config(
    manifest: ExperimentManifest,
    config: OpsConfig,
) -> Dict[str, Any]:
    """Resolve the ``labels`` config the TRAINER will use, torch-free.

    Sources, in the order the orchestrator itself resolves them:

    1. inline ``stages.training.trainer_config`` → ``data.labels`` (the
       wrapper-less pattern; 9 of 17 inline manifests declare it there).
    2. ``stages.training.config`` YAML → ``data.labels``. If that file
       composes via ``_base:`` and carries no own ``labels`` block, the value
       is INHERITED and resolving it here would mean re-implementing the
       trainer's OmegaConf composition — a §0 reuse-first violation and a
       correctness hazard. We record ``unresolved`` naming the bases instead.
       An honest "unknown" beats a fabricated identity.
    3. ``stages.training.overrides`` dotted keys (``data.labels.*`` /
       ``labels.*``) layered last, because the orchestrator applies them AFTER
       inheritance.
    """
    try:
        training = manifest.stages.training
        labels: Optional[Dict[str, Any]] = None
        origin = ""
        note = ""

        tc = training.trainer_config
        if isinstance(tc, dict):
            origin = "manifest.stages.training.trainer_config"
            cand = (tc.get("data") or {}).get("labels") or tc.get("labels")
            if isinstance(cand, dict):
                labels = dict(cand)
            elif tc.get("_base"):
                note = f"inline trainer_config inherits via _base={tc['_base']!r}"
        elif training.config:
            origin = f"manifest.stages.training.config={training.config}"
            try:
                import yaml  # noqa: WPS433 — soft, loader-local

                path = config.paths.resolve(training.config)
                doc = yaml.safe_load(Path(path).read_text()) or {}
                cand = (doc.get("data") or {}).get("labels") or doc.get("labels")
                if isinstance(cand, dict):
                    labels = dict(cand)
                elif doc.get("_base"):
                    note = f"trainer config inherits via _base={doc['_base']!r}"
            except Exception as exc:
                note = f"cannot read trainer config: {type(exc).__name__}: {exc}"

        if labels is None:
            return _unresolved(
                f"no labels block reachable from {origin or 'training stage'}"
                + (f" ({note})" if note else "")
            )

        applied = []
        for key, value in (training.overrides or {}).items():
            for prefix in ("data.labels.", "labels."):
                if key.startswith(prefix):
                    labels[key[len(prefix) :]] = value
                    applied.append(key)
                    break

        out: Dict[str, Any] = {
            "resolved": True,
            "origin": origin,
            "source": labels.get("source"),
            "return_type": labels.get("return_type"),
            "task": labels.get("task"),
            "primary_horizon_idx": labels.get("primary_horizon_idx"),
            "overrides_applied": sorted(applied),
            "config": labels,
        }
        try:
            from hft_contracts import compute_label_strategy_hash

            # The hash `hft_contracts.compatibility` defines and that NO
            # consumer has ever compared (0 call sites in hft-ops /
            # hft-feature-evaluator before today). Recording it is the
            # precondition for ever comparing it.
            out["label_strategy_hash"] = compute_label_strategy_hash(labels)
        except Exception as exc:
            out["label_strategy_hash"] = None
            out["label_strategy_hash_reason"] = f"{type(exc).__name__}: {exc}"
        if note:
            out["note"] = note
        return out
    except Exception as exc:
        logger.debug("trainer-label identity unresolved", exc_info=True)
        return _unresolved(f"{type(exc).__name__}: {exc}")


def _export_declares_forward_prices(data_dir: Path, split: str) -> Optional[bool]:
    """Does the export declare ``forward_prices.exported``? None = unknown.

    This is the second half of the trainer's ``auto`` predicate
    (``dataset.py:683-700``): ``auto`` derives ONLY when the export says
    forward prices are present. Without this check an ``auto`` manifest could
    be reported as diverging when it will in fact read the on-disk array.
    """
    try:
        for md in sorted((data_dir / split).glob("*_metadata.json")):
            with open(md) as fh:
                meta = json.load(fh)
            fp = meta.get("forward_prices")
            return bool(isinstance(fp, dict) and fp.get("exported", False))
        return None
    except Exception:
        return None


def _build_label_identity(
    manifest: ExperimentManifest,
    config: OpsConfig,
    data_dir: Path,
    split: str,
) -> Dict[str, Any]:
    """Assemble scored-vs-fitted label identity + an explicit verdict."""
    scored = _scored_label_identity(data_dir, split)
    will_fit = _resolve_trainer_labels_config(manifest, config)
    declares_fp = _export_declares_forward_prices(data_dir, split)

    verdict = "unknown"
    detail = "trainer labels config could not be resolved"
    if will_fit.get("resolved"):
        source = will_fit.get("source")
        if source == "forward_prices":
            verdict = "derived_at_load"
            detail = (
                "labels.source='forward_prices' — the trainer recomputes "
                "labels from forward_prices via LabelFactory and DISCARDS the "
                "on-disk array the gate scored above"
            )
        elif source == "auto":
            if declares_fp is True:
                verdict = "derived_at_load"
                detail = (
                    "labels.source='auto' and the export declares "
                    "forward_prices.exported=true — the trainer will derive"
                )
            elif declares_fp is False:
                verdict = "scores_fitted_array"
                detail = (
                    "labels.source='auto' but the export declares no "
                    "forward_prices — the trainer reads the on-disk array"
                )
            else:
                detail = (
                    "labels.source='auto' and export metadata is unreadable; "
                    "derivation cannot be predicted"
                )
        elif source is None:
            detail = "labels block carries no 'source' key"
        else:
            verdict = "scores_fitted_array"
            detail = f"labels.source={source!r} — no load-time derivation"

    return {
        "schema": "label_identity/1.0",
        "scored": scored,
        "will_fit": will_fit,
        "export_declares_forward_prices": declares_fp,
        "divergence": verdict,
        "divergence_detail": detail,
        # Stated so nobody reads a non-blocking record as a silent pass:
        "policy": (
            "RECORDED, NEVER BLOCKING (ruling R3) — a deliberate label "
            "substitution is a legitimate experiment."
        ),
    }


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
            horizons = list(md.get("horizons", [])) or list(md.get("max_horizons", []))
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
