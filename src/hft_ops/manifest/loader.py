"""
Manifest YAML loader with variable resolution.

Loads an experiment manifest YAML, resolves ${...} variable references,
and returns a typed ExperimentManifest dataclass.

Variable resolution supports:
    ${experiment.name}              -- from manifest header
    ${stages.extraction.output_dir} -- cross-reference within manifest
    ${timestamp}                    -- execution timestamp (ISO 8601)
    ${date}                         -- execution date (YYYY-MM-DD)
    ${resolved.horizon_idx}         -- deferred: computed at runtime

Unresolvable references (e.g., ${resolved.*}) are left as-is for
runtime resolution by the stage runners.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from hft_ops.manifest.schema import (
    BacktestingStage,
    BacktestParams,
    DatasetAnalysisStage,
    ExperimentHeader,
    ExperimentManifest,
    ExtractionStage,
    RawAnalysisStage,
    SignalExportStage,
    Stages,
    SweepAxis,
    SweepAxisValue,
    SweepConfig,
    TrainingStage,
    ValidationStage,
)

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

_DEFERRED_PREFIXES = ("resolved.",)


from hft_ops.utils import get_nested as _get_nested


def _resolve_variables(
    raw: Dict[str, Any],
    *,
    now: datetime,
    max_passes: int = 5,
    extra_vars: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve ${...} variable references in a nested dict.

    Performs multiple passes to handle transitive references
    (e.g., A references B which references C). Deferred variables
    (${resolved.*}) are left for runtime resolution.

    Args:
        raw: The raw parsed YAML dict.
        now: Current timestamp for ${timestamp} and ${date}.
        max_passes: Maximum resolution passes to prevent infinite loops.
        extra_vars: Optional extra variable bindings consulted BEFORE falling
            back to dotted-path lookup in ``raw``. Introduced Phase 5 FULL-A
            (2026-04-17) to let sweep expansion inject ``{"sweep": {"point_name":
            ..., "axis_values": {...}}}`` without having to mutate ``raw``
            (which would require post-resolution key stripping). Lookup
            precedence: deferred prefixes → builtin_vars → extra_vars → raw.

    Returns:
        The dict with all resolvable variables substituted.
    """
    builtin_vars: Dict[str, str] = {
        "timestamp": now.strftime("%Y%m%dT%H%M%S"),
        "date": now.strftime("%Y-%m-%d"),
    }
    extras: Dict[str, Any] = extra_vars or {}

    def _substitute(value: Any) -> Any:
        if isinstance(value, str):
            def _replacer(match: re.Match) -> str:
                key = match.group(1)
                if any(key.startswith(p) for p in _DEFERRED_PREFIXES):
                    return match.group(0)
                if key in builtin_vars:
                    return builtin_vars[key]
                # Extra-vars lookup: dotted-path walk into the extras dict
                if extras:
                    resolved_extra = _get_nested(extras, key)
                    if resolved_extra is not None and isinstance(resolved_extra, (str, int, float)):
                        return str(resolved_extra)
                resolved = _get_nested(raw, key)
                if resolved is not None and isinstance(resolved, (str, int, float)):
                    return str(resolved)
                return match.group(0)

            return _VAR_PATTERN.sub(_replacer, value)
        elif isinstance(value, dict):
            return {k: _substitute(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute(v) for v in value]
        return value

    resolved = raw
    for _ in range(max_passes):
        prev = str(resolved)
        resolved = _substitute(resolved)
        if str(resolved) == prev:
            break

    return resolved


def _build_backtest_params(raw: Dict[str, Any]) -> BacktestParams:
    """Build BacktestParams from a raw dict, ignoring unknown keys."""
    known_fields = {f.name for f in BacktestParams.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return BacktestParams(**filtered)


def _build_extraction(raw: Dict[str, Any]) -> ExtractionStage:
    return ExtractionStage(
        enabled=raw.get("enabled", True),
        skip_if_exists=raw.get("skip_if_exists", True),
        config=raw.get("config", ""),
        output_dir=raw.get("output_dir", ""),
    )


def _build_raw_analysis(raw: Dict[str, Any]) -> RawAnalysisStage:
    return RawAnalysisStage(
        enabled=raw.get("enabled", False),
        profile=raw.get("profile", "standard"),
        symbol=raw.get("symbol", ""),
        data_dir=raw.get("data_dir", ""),
        analyzers=raw.get("analyzers", []),
        output_dir=raw.get("output_dir", ""),
    )


def _build_dataset_analysis(raw: Dict[str, Any]) -> DatasetAnalysisStage:
    return DatasetAnalysisStage(
        enabled=raw.get("enabled", True),
        profile=raw.get("profile", "quick"),
        split=raw.get("split", "train"),
        data_dir=raw.get("data_dir", ""),
        analyzers=raw.get("analyzers", []),
        output_dir=raw.get("output_dir", ""),
    )


def _build_training(raw: Dict[str, Any]) -> TrainingStage:
    trainer_config = raw.get("trainer_config")
    # Normalize empty dicts/lists to None so "unset" is unambiguous
    if trainer_config is not None and not isinstance(trainer_config, dict):
        raise ValueError(
            f"stages.training.trainer_config must be a dict, got {type(trainer_config).__name__}"
        )
    if trainer_config == {}:
        trainer_config = None

    config_path = raw.get("config", "")

    # Fail-fast on the loader side for the impossible combo.
    # (Additional semantic check lives in validate_manifest for enabled=True.)
    if config_path and trainer_config is not None:
        raise ValueError(
            "stages.training: specify EITHER 'config' (path) OR 'trainer_config' "
            "(inline dict), not both. See TrainingStage docstring for guidance."
        )

    return TrainingStage(
        enabled=raw.get("enabled", True),
        config=config_path,
        trainer_config=trainer_config,
        overrides=raw.get("overrides", {}),
        horizon_value=raw.get("horizon_value"),
        output_dir=raw.get("output_dir", ""),
        extra_args=raw.get("extra_args", []),
    )


def _build_backtesting(raw: Dict[str, Any]) -> BacktestingStage:
    params_raw = raw.get("params", {})
    horizon_idx = raw.get("horizon_idx")
    if isinstance(horizon_idx, str) and "${" in horizon_idx:
        horizon_idx = None
    elif horizon_idx is not None:
        horizon_idx = int(horizon_idx)

    return BacktestingStage(
        enabled=raw.get("enabled", True),
        script=raw.get("script", "scripts/backtest_deeplob.py"),
        model_checkpoint=raw.get("model_checkpoint", ""),
        data_dir=raw.get("data_dir", ""),
        signals_dir=raw.get("signals_dir", ""),
        horizon_idx=horizon_idx,
        params=_build_backtest_params(params_raw) if params_raw else BacktestParams(),
        params_file=raw.get("params_file", ""),
        extra_args=raw.get("extra_args", []),
    )


def _build_signal_export(raw: Dict[str, Any]) -> SignalExportStage:
    return SignalExportStage(
        enabled=raw.get("enabled", False),
        script=raw.get("script", "scripts/export_signals.py"),
        checkpoint=raw.get("checkpoint", ""),
        split=raw.get("split", "test"),
        output_dir=raw.get("output_dir", ""),
        extra_args=raw.get("extra_args", []),
    )


def _build_validation(raw: Dict[str, Any]) -> ValidationStage:
    """Parse stages.validation from raw YAML.

    Validates ``on_fail`` is one of the three accepted values; any other
    input raises early at load time (fail-fast — researchers shouldn't
    discover typos at gate-invocation time).
    """
    on_fail = raw.get("on_fail", "warn")
    if on_fail not in ("warn", "abort", "record_only"):
        raise ValueError(
            f"stages.validation.on_fail must be one of "
            f"{{'warn', 'abort', 'record_only'}}, got {on_fail!r}"
        )

    allow_zero = raw.get("allow_zero_ic_names", [])
    if not isinstance(allow_zero, list):
        raise ValueError(
            f"stages.validation.allow_zero_ic_names must be a list, "
            f"got {type(allow_zero).__name__}"
        )

    return ValidationStage(
        enabled=raw.get("enabled", True),
        on_fail=on_fail,
        target_horizon=str(raw.get("target_horizon", "")),
        min_ic=float(raw.get("min_ic", 0.05)),
        min_ic_count=int(raw.get("min_ic_count", 2)),
        min_return_std_bps=float(raw.get("min_return_std_bps", 5.0)),
        min_stability=float(raw.get("min_stability", 2.0)),
        sample_size=int(raw.get("sample_size", 200_000)),
        n_folds=int(raw.get("n_folds", 20)),
        allow_zero_ic_names=[str(x) for x in allow_zero],
        profile_ref=raw.get("profile_ref", ""),
        output_dir=raw.get("output_dir", ""),
    )


def _build_sweep(raw: Dict[str, Any]) -> SweepConfig:
    """Build SweepConfig from raw YAML dict."""
    axes_raw = raw.get("axes", [])
    axes = []
    for axis_raw in axes_raw:
        values = []
        for val_raw in axis_raw.get("values", []):
            # Phase 5 FULL-A post-audit fix (Agent 1 H2): YAML `overrides:`
            # with no value parses to None → `.get("overrides", {})` returns
            # None (key IS present). Downstream iteration would crash. Normalize.
            if "overrides" in val_raw:
                raw_overrides = val_raw.get("overrides")
            else:
                raw_overrides = {k: v for k, v in val_raw.items() if k != "label"}
            if raw_overrides is None:
                raw_overrides = {}
            if not isinstance(raw_overrides, dict):
                raise ValueError(
                    f"sweep.axes[*].values[*].overrides must be a dict or null, "
                    f"got {type(raw_overrides).__name__} for label "
                    f"{val_raw.get('label', '<unnamed>')!r}"
                )
            values.append(
                SweepAxisValue(
                    label=val_raw.get("label", ""),
                    overrides=raw_overrides,
                )
            )
        axes.append(
            SweepAxis(
                name=axis_raw.get("name", ""),
                values=values,
            )
        )
    return SweepConfig(
        name=raw.get("name", ""),
        strategy=raw.get("strategy", "grid"),
        axes=axes,
    )


def load_manifest(
    manifest_path: str | Path,
    *,
    now: datetime | None = None,
) -> ExperimentManifest:
    """Load and resolve an experiment manifest from a YAML file.

    Args:
        manifest_path: Path to the manifest YAML file.
        now: Override timestamp for deterministic testing.

    Returns:
        A fully resolved ExperimentManifest.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if now is None:
        now = datetime.now(timezone.utc)

    with open(manifest_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Manifest is empty or not a dict: {manifest_path}")

    raw = _resolve_variables(raw, now=now)

    experiment_raw = raw.get("experiment", {})
    if not experiment_raw.get("name"):
        raise ValueError(
            f"Manifest missing required field: experiment.name ({manifest_path})"
        )

    header = ExperimentHeader(
        name=experiment_raw["name"],
        description=experiment_raw.get("description", ""),
        hypothesis=experiment_raw.get("hypothesis", ""),
        contract_version=experiment_raw.get("contract_version", ""),
        tags=experiment_raw.get("tags", []),
    )

    stages_raw = raw.get("stages", {})
    stages = Stages(
        extraction=_build_extraction(stages_raw.get("extraction", {})),
        raw_analysis=_build_raw_analysis(stages_raw.get("raw_analysis", {})),
        dataset_analysis=_build_dataset_analysis(
            stages_raw.get("dataset_analysis", {})
        ),
        validation=_build_validation(stages_raw.get("validation", {})),
        training=_build_training(stages_raw.get("training", {})),
        signal_export=_build_signal_export(stages_raw.get("signal_export", {})),
        backtesting=_build_backtesting(stages_raw.get("backtesting", {})),
    )

    # Parse optional sweep section
    sweep = None
    sweep_raw = raw.get("sweep")
    if sweep_raw and isinstance(sweep_raw, dict):
        sweep = _build_sweep(sweep_raw)

    return ExperimentManifest(
        experiment=header,
        pipeline_root=raw.get("pipeline_root", ".."),
        stages=stages,
        sweep=sweep,
        manifest_path=str(manifest_path),
    )
