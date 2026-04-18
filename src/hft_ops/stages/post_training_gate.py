"""Post-training regression-detection gate runner (Phase 7 Stage 7.4).

Runs between ``training`` and ``signal_export``. Evaluates the completed
experiment against three checks:

1. **Floor check** — primary metric above a configured floor (rule §13:
   meaningful predictive signal must exist for any model worth shipping).
2. **Prior-best ratio check** — new metric within ``min_ratio_vs_prior_best``
   of the best prior experiment for the same ``(model_type, labeling_strategy,
   horizon)`` signature. Catches silent regressions as the pipeline evolves.
3. **Cost-breakeven check** (optional, regression only) — model's prediction
   magnitude plausibly clears the IBKR-calibrated cost floor (1.4 bps Deep
   ITM per CLAUDE.md §IBKR 0DTE Cost Model). Informational only by default.

On regression, disposition is per ``stage.on_regression``:
- ``warn`` (default): annotate + proceed (researcher can still inspect
  backtest + signal outputs to diagnose the regression).
- ``abort``: raise StageFailure → pipeline halts with the gate report
  attached to the ledger record.
- ``record_only``: silent; report written to disk only.

Design choices:

- **Disk-based metric read**: the gate reads ``test_metrics.json`` +
  ``training_history.json`` from the training stage's output_dir rather
  than depending on the cli.py orchestration passing prior StageResult
  state. Keeps the Runner interface uniform across stages.
- **Ledger-based prior-best lookup**: uses ``ExperimentLedger.filter()``
  to find matching historical experiments. Runs in O(N) over the index;
  ledger size well under SQLite-migration threshold (Phase 8) so this
  is fine for now.
- **Primary-metric auto-infer**: if ``stage.primary_metric`` is empty,
  try ``test_ic`` → ``test_directional_accuracy`` → ``best_val_macro_f1``
  → ``best_val_accuracy``. First present wins.
- **NaN/Inf safety** (rule §2): all metric reads guarded by
  ``math.isfinite``; missing / non-finite values PASS the floor check
  (no false abort) but are reported in the gate_report.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import StageResult, StageStatus

logger = logging.getLogger(__name__)


# Metric keys tried in order when PostTrainingGateStage.primary_metric is
# empty. First finite-float value wins. The ordering matches the
# pipeline's validated priority: regression IC (E5 canonical) > directional
# accuracy (classification/regression crossover) > val_macro_f1
# (classification) > val_accuracy (classification fallback).
_PRIMARY_METRIC_FALLBACK_ORDER = (
    "test_ic",
    "test_directional_accuracy",
    "best_val_macro_f1",
    "best_val_accuracy",
)


@dataclass
class CheckResult:
    """Outcome of a single gate check."""

    name: str
    status: str  # "pass" | "warn" | "fail" | "skipped"
    message: str = ""
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class GateReport:
    """Full post-training gate report.

    Written to ``<output_dir>/gate_report.json``. Summary attached to
    ``StageResult.captured_metrics["post_training_gate"]`` so cli.py
    _record_experiment can propagate into ExperimentRecord.notes.
    """

    status: str  # "pass" | "warn" | "abort"
    primary_metric_name: str = ""
    primary_metric_value: Optional[float] = None
    prior_best_experiment_id: str = ""
    prior_best_metric_value: Optional[float] = None
    n_matching_prior_experiments: int = 0
    checks: List[CheckResult] = field(default_factory=list)
    match_signature: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "prior_best_experiment_id": self.prior_best_experiment_id,
            "prior_best_metric_value": self.prior_best_metric_value,
            "n_matching_prior_experiments": self.n_matching_prior_experiments,
            "checks": [asdict(c) for c in self.checks],
            "match_signature": self.match_signature,
        }

    def summary(self) -> str:
        """One-line human summary for logs + ledger notes."""
        primary_str = (
            f"{self.primary_metric_value:.4f}"
            if self.primary_metric_value is not None
            else "N/A"
        )
        fail_checks = [c.name for c in self.checks if c.status == "fail"]
        warn_checks = [c.name for c in self.checks if c.status == "warn"]
        parts = [f"post_training_gate: {self.status.upper()}"]
        parts.append(f"{self.primary_metric_name}={primary_str}")
        if self.prior_best_metric_value is not None:
            parts.append(f"prior_best={self.prior_best_metric_value:.4f}")
        if fail_checks:
            parts.append(f"failed=[{','.join(fail_checks)}]")
        if warn_checks:
            parts.append(f"warned=[{','.join(warn_checks)}]")
        return " | ".join(parts)


class PostTrainingGateRunner:
    """Runs the post-training regression-detection gate.

    See module docstring for design rationale.
    """

    @property
    def stage_name(self) -> str:
        return "post_training_gate"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.post_training_gate

        if stage.on_regression not in ("warn", "abort", "record_only"):
            errors.append(
                f"stages.post_training_gate.on_regression must be "
                f"'warn' | 'abort' | 'record_only'; got "
                f"{stage.on_regression!r}"
            )

        if stage.min_metric_floor < 0:
            errors.append(
                f"stages.post_training_gate.min_metric_floor must be >= 0; "
                f"got {stage.min_metric_floor}"
            )

        if not (0.0 <= stage.min_ratio_vs_prior_best <= 1.0):
            errors.append(
                f"stages.post_training_gate.min_ratio_vs_prior_best must be "
                f"in [0.0, 1.0]; got {stage.min_ratio_vs_prior_best}"
            )

        # Training must be enabled AND have an output_dir so we have
        # metrics to compare.
        if not manifest.stages.training.enabled:
            errors.append(
                "stages.post_training_gate requires stages.training.enabled=True "
                "(the gate reads test_metrics.json / training_history.json from "
                "the training output_dir)."
            )

        if not manifest.stages.training.output_dir:
            errors.append(
                "stages.post_training_gate requires stages.training.output_dir "
                "to be set (so the gate knows where to load metrics from)."
            )

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.post_training_gate
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = (
                "dry-run: would evaluate post-training regression gate"
            )
            return result

        # Locate training output
        try:
            training_output_dir = config.paths.resolve(
                manifest.stages.training.output_dir
            )
        except Exception as exc:
            result.status = StageStatus.FAILED
            result.error_message = (
                f"Cannot resolve stages.training.output_dir: {exc}"
            )
            return result

        if not training_output_dir.exists():
            result.status = StageStatus.FAILED
            result.error_message = (
                f"Training output_dir does not exist: {training_output_dir}. "
                f"post_training_gate requires training to have produced "
                f"metrics files."
            )
            return result

        # Read metrics from disk
        metrics = _read_training_metrics(training_output_dir)

        # Select primary metric
        primary_name, primary_value = _select_primary_metric(
            metrics, stage.primary_metric
        )

        # Set up output dir for gate_report.json
        output_dir_path = (
            config.paths.resolve(stage.output_dir) if stage.output_dir
            else config.paths.runs_dir / manifest.experiment.name / "post_training_gate"
        )
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Build match signature for prior-best query
        match_signature = _build_match_signature(
            manifest=manifest,
            match_fields=tuple(stage.match_on_signature),
        )

        # Query ledger for prior best (if ratio check is enabled)
        prior_best_id: str = ""
        prior_best_value: Optional[float] = None
        n_matching = 0
        if stage.min_ratio_vs_prior_best > 0.0 and primary_name:
            prior_best_id, prior_best_value, n_matching = (
                _find_prior_best_experiment(
                    ledger_dir=config.paths.ledger_dir,
                    match_signature=match_signature,
                    metric_name=primary_name,
                    exclude_experiment_name=manifest.experiment.name,
                )
            )

        # Run checks
        start = time.monotonic()
        checks: List[CheckResult] = []

        # Check 1: Floor
        checks.append(
            _check_floor(
                metric_name=primary_name,
                metric_value=primary_value,
                floor=stage.min_metric_floor,
            )
        )

        # Check 2: Prior-best ratio
        checks.append(
            _check_prior_best_ratio(
                metric_name=primary_name,
                current_value=primary_value,
                prior_best_value=prior_best_value,
                min_ratio=stage.min_ratio_vs_prior_best,
                n_matching=n_matching,
            )
        )

        # Check 3: Cost breakeven (regression only; informational)
        checks.append(
            _check_cost_breakeven(
                metrics=metrics,
                cost_breakeven_bps=stage.cost_breakeven_bps,
            )
        )

        # Aggregate
        any_fail = any(c.status == "fail" for c in checks)
        any_warn = any(c.status == "warn" for c in checks)
        gate_status = "pass"
        if any_fail:
            gate_status = "abort" if stage.on_regression == "abort" else "warn"
        elif any_warn:
            gate_status = "warn"
        # record_only mode: always "pass" regardless of check outcomes
        if stage.on_regression == "record_only":
            gate_status = "pass"

        report = GateReport(
            status=gate_status,
            primary_metric_name=primary_name,
            primary_metric_value=primary_value,
            prior_best_experiment_id=prior_best_id,
            prior_best_metric_value=prior_best_value,
            n_matching_prior_experiments=n_matching,
            checks=checks,
            match_signature=match_signature,
        )

        # Persist report
        report_path = output_dir_path / "gate_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
            f.write("\n")

        result.duration_seconds = time.monotonic() - start
        result.output_dir = str(output_dir_path)
        result.captured_metrics["post_training_gate"] = report.to_dict()
        result.captured_metrics["post_training_gate_summary"] = report.summary()

        logger.info("post_training_gate: %s", report.summary())

        if gate_status == "abort":
            result.status = StageStatus.FAILED
            result.error_message = (
                f"Post-training regression gate failed: {report.summary()}. "
                f"Full report: {report_path}"
            )
        else:
            # warn + record_only both proceed as COMPLETED (researcher
            # reviews the gate_report; pipeline continues).
            result.status = StageStatus.COMPLETED
            if gate_status == "warn":
                result.error_message = (
                    f"post_training_gate: regression detected (non-fatal). "
                    f"{report.summary()}"
                )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_training_metrics(training_output_dir: Path) -> Dict[str, float]:
    """Read flat metrics dict from test_metrics.json + training_history.json.

    Returns only FINITE float values. Non-finite (NaN/Inf) are dropped
    with a WARNING-level log; missing files are silent (no-op).
    """
    metrics: Dict[str, float] = {}

    test_metrics_file = training_output_dir / "test_metrics.json"
    if test_metrics_file.exists():
        try:
            with open(test_metrics_file, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        metrics[k] = float(v)
                    elif isinstance(v, (int, float)):
                        logger.warning(
                            "test_metrics.json: non-finite %s=%s skipped",
                            k, v,
                        )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", test_metrics_file, exc)

    history_file = training_output_dir / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
            if isinstance(history, dict):
                # Legacy dict-of-lists format
                val_accs = history.get("val_accuracy", [])
                if val_accs and all(isinstance(v, (int, float)) for v in val_accs):
                    finite = [v for v in val_accs if math.isfinite(v)]
                    if finite:
                        metrics.setdefault("best_val_accuracy", max(finite))
                val_f1s = history.get("val_macro_f1", [])
                if val_f1s and all(isinstance(v, (int, float)) for v in val_f1s):
                    finite = [v for v in val_f1s if math.isfinite(v)]
                    if finite:
                        metrics.setdefault("best_val_macro_f1", max(finite))
            elif isinstance(history, list):
                # Per-epoch list-of-dicts format
                for key in ("val_accuracy", "val_macro_f1"):
                    series = [
                        epoch.get(key)
                        for epoch in history
                        if isinstance(epoch, dict)
                        and isinstance(epoch.get(key), (int, float))
                        and math.isfinite(epoch.get(key))
                    ]
                    if series:
                        metrics.setdefault(f"best_{key}", max(series))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", history_file, exc)

    return metrics


def _select_primary_metric(
    metrics: Dict[str, float],
    configured_name: str,
) -> tuple[str, Optional[float]]:
    """Return (metric_name, metric_value) per stage config + fallback order."""
    if configured_name:
        # Configured explicitly — use it, even if missing (None value).
        return configured_name, metrics.get(configured_name)
    for candidate in _PRIMARY_METRIC_FALLBACK_ORDER:
        if candidate in metrics:
            return candidate, metrics[candidate]
    return "", None


def _build_match_signature(
    manifest: ExperimentManifest,
    match_fields: tuple,
) -> Dict[str, Any]:
    """Derive the match signature from the current manifest.

    Reads fields from the trainer config (inline or file) via the same
    extraction logic used by cli.py::_record_experiment so the signature
    is consistent with how ExperimentRecord.training_config was populated.

    Returns a dict {field_name: value_or_empty_string}. Empty string for
    missing fields so the dict shape is consistent across all experiments.
    """
    signature: Dict[str, Any] = {}

    # Pull the inline trainer_config if present (Phase 1 wrapper-less);
    # otherwise derive from manifest.stages.* where feasible. Fallback to
    # empty string.
    trainer_config: Dict[str, Any] = (
        manifest.stages.training.trainer_config
        if manifest.stages.training.trainer_config
        else {}
    )

    for field_name in match_fields:
        if field_name == "model_type":
            signature["model_type"] = (
                trainer_config.get("model", {}).get("model_type", "")
                if isinstance(trainer_config.get("model"), dict)
                else ""
            )
        elif field_name == "labeling_strategy":
            signature["labeling_strategy"] = (
                trainer_config.get("data", {}).get("labeling_strategy", "")
                if isinstance(trainer_config.get("data"), dict)
                else ""
            )
        elif field_name == "horizon_value":
            signature["horizon_value"] = manifest.stages.training.horizon_value
        else:
            # Unknown field — try inline trainer_config then manifest stages
            signature[field_name] = trainer_config.get(field_name, "")

    return signature


def _find_prior_best_experiment(
    ledger_dir: Path,
    match_signature: Dict[str, Any],
    metric_name: str,
    exclude_experiment_name: str,
) -> tuple[str, Optional[float], int]:
    """Query the ledger for the best prior experiment matching signature.

    Returns ``(experiment_id, metric_value, n_matching)``. If no match,
    returns ``("", None, 0)``.

    Phase 7 post-validation (2026-04-19): guard against degenerate
    match-signatures where every field is empty/zero. Without this guard,
    the signature-matching loop would SKIP every field (via the
    ``expected_value == ""`` continue), resulting in EVERY historical
    experiment being considered a match regardless of type. That would
    produce false regressions (e.g., comparing a classification experiment
    against the best-ever regression IC). The guard returns an empty
    result, which downstream the ``_check_prior_best_ratio`` treats as
    "no baseline → skipped / pass by default".
    """
    # Lazy-import to avoid circular module load
    from hft_ops.ledger import ExperimentLedger  # noqa: WPS433

    # Degenerate signature guard: if every field is empty/zero, there's
    # no basis for comparison — skip the query entirely.
    if not any(
        (v != "" and v != 0 and v is not None)
        for v in match_signature.values()
    ):
        logger.info(
            "post_training_gate: match_signature has no meaningful fields "
            "(%r); skipping prior-best lookup to avoid false positives",
            match_signature,
        )
        return "", None, 0

    try:
        ledger = ExperimentLedger(ledger_dir)
    except Exception as exc:
        logger.warning("Could not open ledger at %s: %s", ledger_dir, exc)
        return "", None, 0

    # Iterate the in-memory _index entries; each is a dict from
    # ExperimentRecord.index_entry() containing model_type, labeling_strategy,
    # etc., plus training_metrics.
    matching: List[Dict[str, Any]] = []
    for entry in ledger._index:
        if entry.get("name") == exclude_experiment_name:
            continue
        # Only consider completed training records (skip analysis, calibration,
        # evaluation, sweep_aggregate).
        if entry.get("record_type", "training") != "training":
            continue
        # Match signature
        matches = True
        for field_name, expected_value in match_signature.items():
            # Empty expected means "not specified in this experiment" —
            # conservative: require match on specified fields only.
            if expected_value == "":
                continue
            actual = entry.get(field_name, "")
            if actual != expected_value:
                matches = False
                break
        if matches:
            matching.append(entry)

    n_matching = len(matching)
    if not matching:
        return "", None, 0

    # Find best by metric_name in training_metrics (or top-level for
    # index_entry keys like best_val_macro_f1).
    def _extract_metric(entry: Dict[str, Any]) -> Optional[float]:
        tm = entry.get("training_metrics", {})
        if isinstance(tm, dict) and metric_name in tm:
            v = tm[metric_name]
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        # Fall back to top-level entry key (index_entry surfaces some
        # classification keys at top level).
        v = entry.get(metric_name)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
        return None

    candidates_with_metric = [
        (entry, _extract_metric(entry))
        for entry in matching
    ]
    candidates_with_metric = [
        (e, v) for e, v in candidates_with_metric if v is not None
    ]
    if not candidates_with_metric:
        return "", None, n_matching

    best_entry, best_value = max(candidates_with_metric, key=lambda pair: pair[1])
    return best_entry.get("experiment_id", ""), best_value, n_matching


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_floor(
    metric_name: str,
    metric_value: Optional[float],
    floor: float,
) -> CheckResult:
    """Floor check: primary metric >= configured floor."""
    if metric_value is None:
        return CheckResult(
            name="floor",
            status="skipped",
            message=(
                f"primary metric {metric_name!r} not present in captured "
                f"metrics; floor check skipped"
            ),
            threshold=floor,
        )
    if metric_value >= floor:
        return CheckResult(
            name="floor",
            status="pass",
            message=f"{metric_name}={metric_value:.4f} >= floor={floor:.4f}",
            metric_value=metric_value,
            threshold=floor,
        )
    return CheckResult(
        name="floor",
        status="fail",
        message=(
            f"{metric_name}={metric_value:.4f} below floor={floor:.4f} "
            f"(rule §13 — meaningful predictive signal required)"
        ),
        metric_value=metric_value,
        threshold=floor,
    )


def _check_prior_best_ratio(
    metric_name: str,
    current_value: Optional[float],
    prior_best_value: Optional[float],
    min_ratio: float,
    n_matching: int,
) -> CheckResult:
    """Prior-best ratio check: current >= min_ratio * prior_best."""
    if min_ratio <= 0.0:
        return CheckResult(
            name="prior_best_ratio",
            status="skipped",
            message="min_ratio_vs_prior_best=0; check disabled",
        )
    if current_value is None:
        return CheckResult(
            name="prior_best_ratio",
            status="skipped",
            message=(
                f"current metric {metric_name!r} not captured; "
                f"ratio check skipped"
            ),
        )
    if prior_best_value is None:
        return CheckResult(
            name="prior_best_ratio",
            status="skipped",
            message=(
                f"no prior experiments with signature match AND metric "
                f"{metric_name!r} (n_matching={n_matching}); no baseline "
                f"— ratio check skipped (PASS by default for first-of-kind)"
            ),
        )
    if prior_best_value <= 0:
        # Ratio-against-zero-or-negative is ill-defined. Pass.
        return CheckResult(
            name="prior_best_ratio",
            status="pass",
            message=(
                f"prior_best={prior_best_value:.4f} <= 0; ratio check "
                f"vacuously passes"
            ),
            metric_value=current_value,
            threshold=prior_best_value,
        )
    ratio = current_value / prior_best_value
    threshold_value = min_ratio * prior_best_value
    if ratio >= min_ratio:
        return CheckResult(
            name="prior_best_ratio",
            status="pass",
            message=(
                f"{metric_name}={current_value:.4f}; ratio_vs_best="
                f"{ratio:.3f} >= {min_ratio:.2f} "
                f"(prior_best={prior_best_value:.4f})"
            ),
            metric_value=current_value,
            threshold=threshold_value,
        )
    return CheckResult(
        name="prior_best_ratio",
        status="fail",
        message=(
            f"REGRESSION: {metric_name}={current_value:.4f} < "
            f"{threshold_value:.4f} = {min_ratio:.2f} × "
            f"prior_best={prior_best_value:.4f} (ratio={ratio:.3f}). "
            f"See ExperimentLedger for prior-best experiment_id."
        ),
        metric_value=current_value,
        threshold=threshold_value,
    )


def _check_cost_breakeven(
    metrics: Dict[str, float],
    cost_breakeven_bps: float,
) -> CheckResult:
    """Cost-breakeven check (informational).

    For regression experiments where ``test_mae`` or ``test_rmse`` is
    captured (trained prediction magnitude in bps), compare against the
    cost floor. Models whose prediction magnitudes are below the cost
    floor cannot realistically be traded.

    **Informational only** — returns ``warn`` (never ``fail``) so it
    doesn't abort pipelines; researcher decides whether to pursue.
    """
    if cost_breakeven_bps <= 0.0:
        return CheckResult(
            name="cost_breakeven",
            status="skipped",
            message="cost_breakeven_bps=0; check disabled",
        )
    # Use test_rmse or test_mae as a rough proxy for typical prediction
    # magnitude. Better: quantile-95 of |pred| from a held-out set, but
    # the current training pipeline doesn't surface that.
    magnitude = metrics.get("test_rmse") or metrics.get("test_mae")
    if magnitude is None:
        return CheckResult(
            name="cost_breakeven",
            status="skipped",
            message=(
                "no test_rmse or test_mae in metrics (not a regression run?); "
                "cost-breakeven check skipped"
            ),
        )
    if magnitude >= cost_breakeven_bps:
        return CheckResult(
            name="cost_breakeven",
            status="pass",
            message=(
                f"prediction magnitude={magnitude:.2f} bps >= "
                f"{cost_breakeven_bps:.2f} bps cost floor"
            ),
            metric_value=magnitude,
            threshold=cost_breakeven_bps,
        )
    return CheckResult(
        name="cost_breakeven",
        status="warn",
        message=(
            f"INFORMATIONAL: prediction magnitude={magnitude:.2f} bps < "
            f"{cost_breakeven_bps:.2f} bps cost floor. Model's signal may "
            f"not clear IBKR-calibrated trading costs (see CLAUDE.md "
            f"§IBKR 0DTE Cost Model)."
        ),
        metric_value=magnitude,
        threshold=cost_breakeven_bps,
    )
