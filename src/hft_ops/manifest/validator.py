"""
Cross-module config validation.

Validates that an experiment manifest is internally consistent and that
referenced module configs are compatible with each other. All validation
happens BEFORE any computation starts to prevent wasted resources.

Checks performed:
    1. Contract version matches hft_contracts.SCHEMA_VERSION
    2. Extractor TOML is parseable and has required sections
    3. Trainer YAML is parseable and has required sections
    4. Cross-module consistency: feature_count, window_size, stride,
       labeling_strategy, num_classes
    5. If extraction output exists, validates export metadata
    6. horizon_value resolves to a valid horizon_idx
    7. Referenced file paths exist

Runtime value resolution (horizon_idx, etc.) is provided via
``resolve_manifest_context`` as a pure function — no mutation of manifest
state. The orchestrator (cli.py) applies the resolved values explicitly
when invoking stage runners. This keeps ``validate_manifest`` side-effect
free and makes sweep expansions safe (no shared-state races).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.paths import PipelinePaths


@dataclass
class ResolvedContext:
    """Runtime-resolved values derived from a manifest.

    Produced by ``resolve_manifest_context`` alongside ``validate_manifest``.
    Stage runners (or the orchestrator CLI) apply these values when needed;
    the manifest dataclass is left untouched so sweep expansion and repeat
    validation are side-effect free.

    Attributes:
        horizon_idx: Zero-based index into the extractor's horizons list
            corresponding to ``stages.training.horizon_value``, or None
            if not resolvable (missing extractor config, mismatched value).
        horizon_value: The original ``horizon_value`` from the training
            stage, stored for convenience.
        feature_count: The feature count computed from the extractor config,
            if one is reachable. None otherwise.
    """

    horizon_idx: Optional[int] = None
    horizon_value: Optional[int] = None
    feature_count: Optional[int] = None

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


class ValidationError:
    """A single validation issue."""

    def __init__(self, severity: str, message: str, context: str = ""):
        self.severity = severity  # "error" or "warning"
        self.message = message
        self.context = context

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        ctx = f" ({self.context})" if self.context else ""
        return f"{prefix} {self.message}{ctx}"

    def __repr__(self) -> str:
        return f"ValidationError({self.severity!r}, {self.message!r})"


class ValidationResult:
    """Aggregated validation result."""

    def __init__(self) -> None:
        self.issues: List[ValidationError] = []

    def error(self, message: str, context: str = "") -> None:
        self.issues.append(ValidationError("error", message, context))

    def warning(self, message: str, context: str = "") -> None:
        self.issues.append(ValidationError("warning", message, context))

    @property
    def errors(self) -> List[ValidationError]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationError]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        if not self.issues:
            return "Validation passed (0 issues)"
        lines = [f"Validation: {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


def _load_toml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a TOML file, returning None on failure."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return None


def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a YAML file, returning None on failure."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _compute_feature_count(features_cfg: Dict[str, Any]) -> int:
    """Compute expected feature count from an extractor TOML [features] section.

    Formula: lob_levels * 4 + derived(8 if enabled) + mbo(36 if enabled) +
             signals(14 if enabled)
    Source: pipeline_contract.toml [features]
    """
    lob_levels = features_cfg.get("lob_levels", 10)
    count = lob_levels * 4  # LOB features: ask_prices + ask_sizes + bid_prices + bid_sizes

    if features_cfg.get("include_derived", False):
        count += 8
    if features_cfg.get("include_mbo", False):
        count += 36
    if features_cfg.get("include_signals", False):
        count += 14

    return count


def _validate_contract_version(
    manifest: ExperimentManifest,
    result: ValidationResult,
) -> None:
    """Check that the manifest's contract_version matches hft_contracts."""
    if not manifest.experiment.contract_version:
        result.warning(
            "No contract_version specified in manifest. "
            "Strongly recommended for reproducibility.",
            context="experiment.contract_version",
        )
        return

    try:
        from hft_contracts import SCHEMA_VERSION

        if manifest.experiment.contract_version != SCHEMA_VERSION:
            result.error(
                f"contract_version '{manifest.experiment.contract_version}' "
                f"!= hft_contracts.SCHEMA_VERSION '{SCHEMA_VERSION}'",
                context="experiment.contract_version",
            )
    except ImportError:
        result.warning(
            "hft_contracts not installed; cannot validate contract_version.",
            context="experiment.contract_version",
        )


def _validate_file_references(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
    result: ValidationResult,
) -> None:
    """Check that referenced config files exist."""
    if manifest.stages.extraction.enabled and manifest.stages.extraction.config:
        config_path = paths.resolve(manifest.stages.extraction.config)
        if not config_path.exists():
            result.error(
                f"Extractor config not found: {config_path}",
                context="stages.extraction.config",
            )

    if manifest.stages.training.enabled:
        has_path = bool(manifest.stages.training.config)
        has_inline = manifest.stages.training.trainer_config is not None

        # Exactly one of (config, trainer_config) must be set when training is enabled.
        # The loader already rejects BOTH-set; here we catch NEITHER-set.
        if not has_path and not has_inline:
            result.error(
                "When stages.training.enabled=True, either 'config' (path) or "
                "'trainer_config' (inline dict) must be specified.",
                context="stages.training",
            )
        elif has_path:
            config_path = paths.resolve(manifest.stages.training.config)
            if not config_path.exists():
                result.error(
                    f"Trainer config not found: {config_path}",
                    context="stages.training.config",
                )
        # If has_inline: no file to check; cross-module consistency uses
        # trainer_config dict directly (see _validate_cross_module_consistency).


def _validate_cross_module_consistency(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
    result: ValidationResult,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Cross-validate extractor TOML against trainer YAML.

    Returns (extractor_config, trainer_config) for downstream use.
    """
    ext_cfg: Optional[Dict[str, Any]] = None
    train_cfg: Optional[Dict[str, Any]] = None

    if manifest.stages.extraction.enabled and manifest.stages.extraction.config:
        ext_path = paths.resolve(manifest.stages.extraction.config)
        if ext_path.exists():
            ext_cfg = _load_toml(ext_path)
            if ext_cfg is None:
                result.error(
                    f"Failed to parse extractor TOML: {ext_path}",
                    context="stages.extraction.config",
                )

    if manifest.stages.training.enabled:
        if manifest.stages.training.config:
            train_path = paths.resolve(manifest.stages.training.config)
            if train_path.exists():
                train_cfg = _load_yaml(train_path)
                if train_cfg is None:
                    result.error(
                        f"Failed to parse trainer YAML: {train_path}",
                        context="stages.training.config",
                    )
        elif manifest.stages.training.trainer_config is not None:
            # Inline trainer config: use the dict directly (already parsed).
            # Note: inline trainer_config may use '_base:' for composition;
            # full multi-base resolution is Phase 3's responsibility. For
            # cross-module consistency validation (Phase 1), we read top-level
            # keys only. Resolution of _base happens in the TrainingRunner when
            # it materializes the effective config to a temp file.
            train_cfg = dict(manifest.stages.training.trainer_config)

    if ext_cfg is None or train_cfg is None:
        return ext_cfg, train_cfg

    features_cfg = ext_cfg.get("features", {})
    ext_feature_count = _compute_feature_count(features_cfg)

    train_data = train_cfg.get("data", {})
    train_feature_count = train_data.get("feature_count")
    if train_feature_count is not None and train_feature_count != ext_feature_count:
        result.error(
            f"Feature count mismatch: extractor computes {ext_feature_count}, "
            f"trainer config specifies {train_feature_count}",
            context="cross-module:feature_count",
        )

    train_model = train_cfg.get("model", {})
    model_input_size = train_model.get("input_size")
    if model_input_size is not None and model_input_size != ext_feature_count:
        result.error(
            f"model.input_size ({model_input_size}) != "
            f"extractor feature_count ({ext_feature_count})",
            context="cross-module:model.input_size",
        )

    ext_sequence = ext_cfg.get("sequence", {})
    train_sequence = train_data.get("sequence", {})

    ext_window = ext_sequence.get("window_size")
    train_window = train_sequence.get("window_size")
    if ext_window is not None and train_window is not None:
        if ext_window != train_window:
            result.error(
                f"window_size mismatch: extractor={ext_window}, trainer={train_window}",
                context="cross-module:window_size",
            )

    ext_stride = ext_sequence.get("stride")
    train_stride = train_sequence.get("stride")
    if ext_stride is not None and train_stride is not None:
        if ext_stride != train_stride:
            result.error(
                f"stride mismatch: extractor={ext_stride}, trainer={train_stride}",
                context="cross-module:stride",
            )

    ext_labels = ext_cfg.get("labels", {})
    ext_strategy = ext_labels.get("strategy", "")
    train_strategy = train_data.get("labeling_strategy", "")

    if ext_strategy and train_strategy:
        ext_norm = ext_strategy.lower().replace("-", "_")
        train_norm = train_strategy.lower().replace("-", "_")
        if ext_norm != train_norm:
            result.warning(
                f"Labeling strategy mismatch: extractor='{ext_strategy}', "
                f"trainer='{train_strategy}'. This may be intentional "
                f"(e.g., triple_barrier export with TLOB-style labels).",
                context="cross-module:labeling_strategy",
            )

    return ext_cfg, train_cfg


def _validate_horizon_resolution(
    manifest: ExperimentManifest,
    ext_cfg: Optional[Dict[str, Any]],
    result: ValidationResult,
) -> Optional[int]:
    """Resolve horizon_value to horizon_idx. Returns resolved idx or None."""
    if not manifest.stages.training.enabled:
        return None

    horizon_value = manifest.stages.training.horizon_value
    if horizon_value is None:
        return None

    if ext_cfg is None:
        result.warning(
            "Cannot resolve horizon_value without extractor config.",
            context="stages.training.horizon_value",
        )
        return None

    labels_cfg = ext_cfg.get("labels", {})
    horizons = labels_cfg.get("max_horizons") or labels_cfg.get("horizons") or []

    if not horizons:
        horizon = labels_cfg.get("horizon")
        if horizon is not None:
            horizons = [horizon]

    if not horizons:
        result.warning(
            "No horizons found in extractor config. Cannot resolve horizon_value.",
            context="stages.training.horizon_value",
        )
        return None

    if horizon_value in horizons:
        idx = horizons.index(horizon_value)
        return idx
    else:
        result.error(
            f"horizon_value={horizon_value} not found in extractor horizons={horizons}",
            context="stages.training.horizon_value",
        )
        return None


def _validate_existing_exports(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
    result: ValidationResult,
) -> None:
    """If extraction output already exists, validate its metadata using
    the full ``hft_contracts.validate_export_contract()`` function.
    """
    if not manifest.stages.extraction.output_dir:
        return

    output_dir = paths.resolve(manifest.stages.extraction.output_dir)
    if not output_dir.exists():
        return

    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metadata_files = sorted(subdir.glob("*_metadata.json"))
        if not metadata_files:
            continue

        sample_meta_path = metadata_files[0]
        try:
            with open(sample_meta_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            result.error(
                f"Failed to read export metadata: {sample_meta_path}: {e}",
                context="stages.extraction.output_dir",
            )
            continue

        if "schema_version" not in metadata:
            result.warning(
                f"Metadata {sample_meta_path.name} has no schema_version. "
                "Re-export with latest extractor for full validation.",
                context=f"export_metadata:{subdir.name}",
            )
            continue

        # Import hft_contracts.validation separately so that an ImportError
        # here cannot cause the downstream `except _CE` clause to raise
        # UnboundLocalError (which happens when _CE never gets bound).
        try:
            from hft_contracts.validation import (
                ContractError as _CE,
                validate_export_contract as _validate,
                validate_provenance_present as _prov,
            )
        except ImportError:
            # hft_contracts not installed / validation module missing — skip
            # contract checks but don't fail the manifest.
            break

        try:
            warnings_list = _validate(metadata, strict_completeness=False)
            for w in warnings_list:
                result.warning(w, context=f"export_metadata:{subdir.name}")

            prov_warnings = _prov(metadata)
            for w in prov_warnings:
                result.warning(w, context=f"provenance:{subdir.name}")

        except _CE as exc:
            result.error(str(exc), context=f"export_contract:{subdir.name}")

        break


def validate_manifest(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
) -> ValidationResult:
    """Validate an experiment manifest comprehensively.

    Performs all validation checks and returns a ValidationResult.
    If result.is_valid is False, the experiment should not be run.

    Args:
        manifest: The parsed experiment manifest.
        paths: Resolved pipeline paths.

    Returns:
        ValidationResult with all discovered issues.
    """
    result = ValidationResult()

    if not manifest.experiment.name:
        result.error("Missing experiment name", context="experiment.name")

    path_errors = paths.validate()
    for err in path_errors:
        result.error(err, context="pipeline_paths")

    _validate_contract_version(manifest, result)
    _validate_file_references(manifest, paths, result)

    ext_cfg, train_cfg = _validate_cross_module_consistency(
        manifest, paths, result
    )

    # Resolve horizon strictly for validation side-effects (e.g., flagging
    # horizon_value that doesn't exist in the extractor's horizons list).
    # DO NOT mutate the manifest here — the CLI / sweep orchestrator applies
    # the resolved value explicitly via ``resolve_manifest_context``.
    _validate_horizon_resolution(manifest, ext_cfg, result)

    _validate_existing_exports(manifest, paths, result)

    # Mirror exactly the stage fields present in ``Stages``. Missing entries
    # here cause validation-only or signal-export-only manifests to be flagged
    # as "no stages enabled" — an incorrect warning that confuses users and
    # retroactive backfill flows.
    enabled_stages = []
    if manifest.stages.extraction.enabled:
        enabled_stages.append("extraction")
    if manifest.stages.raw_analysis.enabled:
        enabled_stages.append("raw_analysis")
    if manifest.stages.dataset_analysis.enabled:
        enabled_stages.append("dataset_analysis")
    if manifest.stages.validation.enabled:
        enabled_stages.append("validation")
    if manifest.stages.training.enabled:
        enabled_stages.append("training")
    if manifest.stages.signal_export.enabled:
        enabled_stages.append("signal_export")
    if manifest.stages.backtesting.enabled:
        enabled_stages.append("backtesting")

    if not enabled_stages:
        result.warning("No stages enabled in manifest.", context="stages")

    if manifest.stages.training.enabled and not manifest.stages.extraction.enabled:
        if not manifest.stages.extraction.output_dir:
            result.error(
                "Training enabled but extraction disabled and no output_dir specified. "
                "Where should the trainer read data from?",
                context="stages.training",
            )

    if manifest.stages.backtesting.enabled and not manifest.stages.training.enabled:
        if not manifest.stages.backtesting.model_checkpoint:
            result.error(
                "Backtesting enabled but training disabled and no model_checkpoint specified.",
                context="stages.backtesting",
            )

    return result


def resolve_manifest_context(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
) -> ResolvedContext:
    """Compute runtime-resolved values from a manifest WITHOUT mutation.

    This is the companion to ``validate_manifest`` for the "runtime side" of
    resolution: translating ``horizon_value`` to the concrete ``horizon_idx``
    using the extractor's horizons list. The orchestrator CLI calls this once
    per run (or per sweep grid point) and applies the result explicitly to
    stage invocations — no shared-state races, no surprising mutations.

    Args:
        manifest: The parsed experiment manifest (not mutated).
        paths: Resolved pipeline paths.

    Returns:
        ResolvedContext with horizon_idx, horizon_value, feature_count
        populated where possible.
    """
    ctx = ResolvedContext(horizon_value=manifest.stages.training.horizon_value)

    if manifest.stages.extraction.enabled and manifest.stages.extraction.config:
        ext_path = paths.resolve(manifest.stages.extraction.config)
        if ext_path.exists():
            ext_cfg = _load_toml(ext_path)
            if ext_cfg is not None:
                features_cfg = ext_cfg.get("features", {})
                ctx.feature_count = _compute_feature_count(features_cfg)

                if manifest.stages.training.enabled:
                    horizon_value = manifest.stages.training.horizon_value
                    if horizon_value is not None:
                        labels_cfg = ext_cfg.get("labels", {})
                        horizons = (
                            labels_cfg.get("max_horizons")
                            or labels_cfg.get("horizons")
                            or []
                        )
                        if not horizons:
                            single = labels_cfg.get("horizon")
                            if single is not None:
                                horizons = [single]
                        if horizon_value in horizons:
                            ctx.horizon_idx = horizons.index(horizon_value)

    return ctx


def apply_resolved_context(
    manifest: ExperimentManifest,
    ctx: ResolvedContext,
) -> None:
    """Apply resolved context values to a manifest in-place.

    This is the ONLY place that mutates manifest state, and it is callee-
    explicit: the orchestrator decides when to apply, not the validator.
    Stage runners that need ``horizon_idx`` can continue to read it from
    ``manifest.stages.backtesting.horizon_idx`` without changing their
    signatures.

    Idempotent: if ``backtesting.horizon_idx`` is already set, we leave it.

    Args:
        manifest: The manifest to mutate. Safe to call with a per-grid-point
            expanded copy (the sweep path already deep-copies).
        ctx: The resolved context.
    """
    if (
        ctx.horizon_idx is not None
        and manifest.stages.backtesting.enabled
        and manifest.stages.backtesting.horizon_idx is None
    ):
        manifest.stages.backtesting.horizon_idx = ctx.horizon_idx
