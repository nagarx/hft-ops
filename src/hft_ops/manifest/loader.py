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

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

if TYPE_CHECKING:
    from hft_ops.paths import PipelinePaths

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

_DEFERRED_PREFIXES = ("resolved.",)
# Note: `resolved.*` IS short-circuited in `_substitute` (returns literal without
# attempting resolution) because runtime-resolved variables MUST NOT be touched at
# load OR expand time — they're populated by stage runners during execution.

_F1_LOAD_EXCLUDED_PREFIXES = ("resolved.", "sweep.")
# F-1 (audit §4.1, 2026-05-19): F-1 walker excludes these from fail-loud at LOAD
# time per pre-commit gate hft-architect B-1 CRITICAL finding. Sweep manifests use
# `${sweep.point_name}` / `${sweep.axis_values.*}` literals that resolve at sweep
# EXPAND time (`manifest/resolver.py:157` via extra_vars), NOT at load time. The
# walker treats them as legitimate-literal AT LOAD. At expand time, `_substitute`
# does NOT short-circuit `sweep.*` (only `resolved.*` per `_DEFERRED_PREFIXES`),
# so `sweep.*` resolves via the extras → builtin → raw fallback chain.


from hft_ops.utils import get_nested as _get_nested


def _maybe_rebase_path(
    resolved_str: str,
    *,
    source_key: str,
    target_key: str,
    paths: Optional["PipelinePaths"],
) -> str:
    """Rebase a substituted path value when source slot and target slot live in
    different path-base coordinate systems.

    Phase α-1 / #PY-78 (2026-05-10) — fixes path-relativity contract mismatch
    between pipeline-root-relative source slots (e.g., ``stages.extraction.output_dir``
    consumed via ``paths.resolve(...)``) and trainer-cwd-relative target slots
    (e.g., ``stages.training.trainer_config.data.data_dir`` consumed bare by
    the trainer subprocess with cwd=``$pipeline_root/lob-model-trainer/``).

    When ``paths`` is None (legacy callers / test fixtures without a pipeline
    root), no rebasing happens — substitution behaves exactly like before this
    fix landed. This preserves backward-compat with the prior behavior.

    Path-base classification is delegated to ``slot_taxonomy.detect_slot_path_base``
    (the SSoT). New slot path-bases must be registered there per hft-rules §0.
    """
    if paths is None:
        return resolved_str
    # Local import to avoid circular dependency at module load time.
    from hft_ops.manifest.slot_taxonomy import PathBase, detect_slot_path_base

    src_base = detect_slot_path_base(source_key)
    tgt_base = detect_slot_path_base(target_key)
    if src_base == PathBase.PIPELINE_ROOT and tgt_base == PathBase.TRAINER_CWD:
        # HIGH-1 short-circuit: absolute source paths are cwd-independent.
        # Skip rebasing — the trainer subprocess will interpret an absolute
        # path identically regardless of cwd, and forcing a long
        # `../../../...` cross-mount relative path is operationally fragile.
        if os.path.isabs(resolved_str):
            return resolved_str
        # Convert pipeline-root-relative source → trainer-cwd-relative literal
        # so trainer subprocess (cwd=trainer_dir) sees the correct path.
        # HIGH-2 fail-loud: surface filesystem errors with actionable context
        # per hft-rules §8 (never silently drop/clamp/fix data; explicit
        # diagnostic citing slot keys + source value).
        try:
            abs_path = paths.resolve(resolved_str)
            return os.path.relpath(abs_path, paths.trainer_dir)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot rebase manifest path: source slot "
                f"'{source_key}' = {resolved_str!r} (pipeline-root-relative) → "
                f"target slot '{target_key}' (trainer-cwd-relative). "
                f"Resolution failed: {type(exc).__name__}: {exc}"
            ) from exc
    return resolved_str


def _resolve_variables(
    raw: Dict[str, Any],
    *,
    now: datetime,
    max_passes: int = 5,
    extra_vars: Dict[str, Any] | None = None,
    paths: Optional["PipelinePaths"] = None,
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
        paths: Optional :class:`PipelinePaths` for path-base aware substitution
            (Phase α-1 / #PY-78). When provided AND a substitution crosses
            path-bases (source slot is pipeline-root-relative, target slot is
            trainer-cwd-relative), the value is rebased via
            ``paths.resolve()`` + ``os.path.relpath(..., trainer_dir)``. When
            ``None`` (legacy/test callers), substitution behaves exactly like
            before — preserving backward-compat. Slot path-base classification
            lives in :mod:`hft_ops.manifest.slot_taxonomy` (SSoT per
            hft-rules §0).

    Returns:
        The dict with all resolvable variables substituted.
    """
    builtin_vars: Dict[str, str] = {
        "timestamp": now.strftime("%Y%m%dT%H%M%S"),
        "date": now.strftime("%Y-%m-%d"),
    }
    extras: Dict[str, Any] = extra_vars or {}

    def _substitute(value: Any, current_dotted_path: str = "") -> Any:
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
                        return _maybe_rebase_path(
                            str(resolved_extra),
                            source_key=key,
                            target_key=current_dotted_path,
                            paths=paths,
                        )
                resolved = _get_nested(raw, key)
                if resolved is not None and isinstance(resolved, (str, int, float)):
                    return _maybe_rebase_path(
                        str(resolved),
                        source_key=key,
                        target_key=current_dotted_path,
                        paths=paths,
                    )
                return match.group(0)

            return _VAR_PATTERN.sub(_replacer, value)
        elif isinstance(value, dict):
            return {
                k: _substitute(
                    v,
                    f"{current_dotted_path}.{k}".lstrip("."),
                )
                for k, v in value.items()
            }
        elif isinstance(value, list):
            # Lists pass through current_dotted_path unchanged. List indices
            # are NOT part of slot taxonomy (slots classify by dotted-name path).
            return [_substitute(v, current_dotted_path) for v in value]
        return value

    resolved = raw
    for _ in range(max_passes):
        prev = str(resolved)
        resolved = _substitute(resolved, "")
        if str(resolved) == prev:
            break

    # F-1 (audit §4.1; 2026-05-19): fail-loud on any unresolved ${...} literal
    # left over post-fixed-point (e.g., cyclic ${A.x} ↔ ${B.x}, unresolvable
    # dotted-path refs, or stuck-state from partially-resolved chains). Walks
    # the entire resolved tree + collects every (dotted_path, key) pair; raises
    # ValueError listing ALL with actionable diagnostics. Per hft-rules §8:
    # `${...}` literals propagating downstream as silent broken state is a
    # CRITICAL contract violation. _DEFERRED_PREFIXES are LEGITIMATELY left as
    # literals (resolved at stage-runner time); excluded from the gate.
    unresolved_refs: List[Tuple[str, str]] = []
    _collect_unresolved_refs(resolved, "", unresolved_refs)
    if unresolved_refs:
        lines = [f"  ${{{key}}} at {path or '<root>'}" for path, key in unresolved_refs]
        raise ValueError(
            f"Manifest variable resolution failed after {max_passes} passes; "
            f"{len(unresolved_refs)} unresolved reference(s) remain (excluding "
            f"deferred prefixes {_F1_LOAD_EXCLUDED_PREFIXES}). Either cyclic refs "
            f"(e.g., ${{A.x}} ↔ ${{B.x}}) or unresolvable dotted-paths. "
            f"Unresolved references:\n" + "\n".join(lines)
        )

    return resolved


def _collect_unresolved_refs(
    value: Any,
    current_path: str,
    accumulator: List[Tuple[str, str]],
) -> None:
    """F-1 (audit §4.1, 2026-05-19): walk resolved tree + collect every
    unresolved ``${...}`` reference NOT covered by ``_F1_LOAD_EXCLUDED_PREFIXES``.

    Mutates ``accumulator`` in-place with ``(dotted_path, var_key)`` pairs.
    Pre-impl gate APPROVE-WITH-MICRO-FIX: collects ALL unresolved refs (not
    just first); excludes ``_F1_LOAD_EXCLUDED_PREFIXES`` (resolved later at
    expand/stage-runner time, not manifest LOAD time). Pre-commit gate
    hft-architect B-1: ``_F1_LOAD_EXCLUDED_PREFIXES`` is a SUPERSET of
    ``_DEFERRED_PREFIXES`` — substitute short-circuits only `resolved.*`
    (runtime-only), but F-1 LOAD walker additionally excludes `sweep.*`
    (expand-time-only).
    """
    if isinstance(value, str):
        for match in _VAR_PATTERN.finditer(value):
            key = match.group(1)
            if not any(key.startswith(p) for p in _F1_LOAD_EXCLUDED_PREFIXES):
                accumulator.append((current_path, key))
    elif isinstance(value, dict):
        for k, v in value.items():
            sub_path = f"{current_path}.{k}".lstrip(".")
            _collect_unresolved_refs(v, sub_path, accumulator)
    elif isinstance(value, (list, tuple)):
        # List/tuple indices not part of dotted-path taxonomy; descend with
        # current_path unchanged (matches _substitute() convention). Tuple
        # included defensively per Wave 2Y D1 (2026-05-19) — YAML loader emits
        # dict/list only, but Python callers (test fixtures, sweep expansion,
        # future Pydantic models) may pre-populate the tree with tuples;
        # silent skip would let ${...} literals propagate as silent broken
        # state (hft-rules §8 violation).
        for v in value:
            _collect_unresolved_refs(v, current_path, accumulator)


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

    # Phase X.1 v2 post-validation fix (Agent 4 Q8 2026-05-04): strict
    # type-validation of strict_checkpoint_fingerprint. Pre-fix, a YAML
    # string `"false"` would silently activate strict mode (string-truthy).
    # Now any non-bool value raises ValueError with actionable message.
    strict_ckpt_fp = raw.get("strict_checkpoint_fingerprint", False)
    if not isinstance(strict_ckpt_fp, bool):
        raise ValueError(
            f"stages.training.strict_checkpoint_fingerprint must be bool "
            f"(true/false), got {type(strict_ckpt_fp).__name__}={strict_ckpt_fp!r}. "
            f"YAML 'true'/'false' parse as bool; quoted '\"true\"' is a string "
            f"and is rejected."
        )

    return TrainingStage(
        enabled=raw.get("enabled", True),
        config=config_path,
        trainer_config=trainer_config,
        overrides=raw.get("overrides", {}),
        horizon_value=raw.get("horizon_value"),
        output_dir=raw.get("output_dir", ""),
        extra_args=raw.get("extra_args", []),
        strict_checkpoint_fingerprint=strict_ckpt_fp,
    )


_KNOWN_BACKTESTING_KEYS = frozenset({
    "enabled",
    "script",
    "model_checkpoint",
    "data_dir",
    "signals_dir",
    "horizon_idx",
    "params",
    "params_file",
    "extra_args",
})


def _build_backtesting(raw: Dict[str, Any]) -> BacktestingStage:
    # Phase 7.5-B.3 (2026-04-23) — Final-validation-round closure per
    # hft-rules §8 "Never silently drop, clamp, or 'fix' data without
    # recording diagnostics." 5-agent adversarial audit of the Phase 7.5
    # cycle (5th agent's Chain B SDR-1/SDR-2) surfaced live-fs evidence
    # that 3 production HMHP manifests declare `readability:` / `holding:` /
    # `costs:` BLOCKS at the `backtesting:` top-level that BacktestingStage
    # does NOT declare. Loader silently dropped these.
    #
    # Operator-facing impact: `nvda_hmhp_128feat_arcx_h10.yaml` declares
    # `costs: {exchange: ARCX}` but runner never sees it → subprocess
    # defaults to `--exchange XNAS` → ARCX experiment runs on XNAS cost
    # model → silent-wrong-result. Same risk class for `readability.min_agreement`,
    # `holding.hold_events`, etc. — coincidentally aligned with defaults
    # TODAY but silently breaks whenever operator wants non-default values.
    #
    # This fix: emit WARN on unknown top-level keys. Converts silent-drop
    # to operator-visible diagnostic. Full typed-dataclass migration (adding
    # `ReadabilityConfig` / `HoldingConfig` / `CostsConfig` sub-fields +
    # runner-side flag emission) is Phase 8+ scope; this surgical fix is
    # sufficient to UNBLOCK Task 1d first-live-run without silent-wrong
    # results.
    import warnings
    unknown_keys = set(raw.keys()) - _KNOWN_BACKTESTING_KEYS
    if unknown_keys:
        warnings.warn(
            f"BacktestingStage loader: silently dropping unknown top-level "
            f"keys {sorted(unknown_keys)!r} (not declared on BacktestingStage "
            f"schema). If you intended these as backtester-script flags, "
            f"move the values into `stages.backtesting.extra_args:` list "
            f"(e.g., `extra_args: [\"--exchange\", \"ARCX\", "
            f"\"--min-agreement\", \"1.0\"]`). Current schema declares: "
            f"{sorted(_KNOWN_BACKTESTING_KEYS)!r}. Phase 8+ will add "
            f"typed sub-dataclasses for `readability`/`holding`/`costs` "
            f"blocks with structured cmd-flag routing.",
            RuntimeWarning,
            stacklevel=2,
        )

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


def _build_post_training_gate(raw: Dict[str, Any]) -> "PostTrainingGateStage":
    """Parse stages.post_training_gate from raw YAML.

    Phase 7 Stage 7.4 Round 6 (2026-04-20, post-push-audit fix): this
    helper was missing despite the stage being declared on ``Stages``
    at ``schema.py:356`` with ``default_factory=PostTrainingGateStage``.
    Missing kwarg in ``load_manifest`` silently filled the stage with
    ``enabled=False`` defaults regardless of user YAML — every
    ``stages.post_training_gate.enabled: true`` was ignored, defeating
    the opt-in mechanism of the just-shipped Stage 7.4. This violated
    hft-rules §5: "If a config option exists but is not fully
    supported, it must fail fast with a precise error — never silently
    degrade."

    Validates ``on_regression`` is one of ``{"warn", "abort",
    "record_only"}`` (matches the schema's PostTrainingGateStage.
    on_regression attribute documentation) and coerces ``match_on_signature``
    to ``List[str]``. Other fields are numeric / string and coerced
    via the standard constructor; invalid types raise early at load
    time (fail-fast).
    """
    # Lazy import to avoid circular dependency with schema.py
    from hft_ops.manifest.schema import PostTrainingGateStage

    on_regression = raw.get("on_regression", "warn")
    if on_regression not in ("warn", "abort", "record_only"):
        raise ValueError(
            f"stages.post_training_gate.on_regression must be one of "
            f"{{'warn', 'abort', 'record_only'}}, got {on_regression!r}"
        )

    match_sig = raw.get(
        "match_on_signature",
        ["model_type", "labeling_strategy", "horizon_value"],
    )
    if not isinstance(match_sig, list):
        raise ValueError(
            f"stages.post_training_gate.match_on_signature must be a list, "
            f"got {type(match_sig).__name__}"
        )

    return PostTrainingGateStage(
        enabled=bool(raw.get("enabled", False)),
        on_regression=on_regression,
        primary_metric=str(raw.get("primary_metric", "")),
        min_metric_floor=float(raw.get("min_metric_floor", 0.05)),
        min_ratio_vs_prior_best=float(raw.get("min_ratio_vs_prior_best", 0.9)),
        match_on_signature=[str(x) for x in match_sig],
        cost_breakeven_bps=float(raw.get("cost_breakeven_bps", 1.4)),
        output_dir=str(raw.get("output_dir", "")),
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
    paths: Optional["PipelinePaths"] = None,
) -> ExperimentManifest:
    """Load and resolve an experiment manifest from a YAML file.

    Args:
        manifest_path: Path to the manifest YAML file.
        now: Override timestamp for deterministic testing.
        paths: Optional :class:`PipelinePaths` for path-base aware ``${...}``
            substitution (Phase α-1 / #PY-78). When ``None``, the loader
            attempts ``PipelinePaths.auto_detect()`` so production callers
            transparently get correct path-rebasing without signature changes.
            If auto-detect raises (test fixtures running outside a pipeline
            root), substitution falls back to legacy behavior (no rebasing) —
            preserves backward compatibility.

    Returns:
        A fully resolved ExperimentManifest.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing.
    """
    # Phase α-1.3 / #PY-83-cluster (2026-05-10): use Path.absolute() not
    # Path.resolve() to preserve symlink-source lineage. The walk-up at
    # line 547 below looks for `contracts/pipeline_contract.toml` in
    # ancestors; if `manifest_path` is under a symlinked checkout dir,
    # `.resolve()` derefs and walk-up jumps off-tree — same defect class
    # as α-3 / #PY-79 in lob-model-trainer/feature_set_resolver.py:442.
    manifest_path = Path(manifest_path).absolute()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if now is None:
        now = datetime.now(timezone.utc)

    if paths is None:
        # Auto-detect pipeline root by walking up from the MANIFEST path
        # (not from this source file's location). This ensures test fixtures
        # in tmp dirs auto-detect their OWN tmp pipeline root if present,
        # OR fall through to legacy no-rebasing if the fixture lacks
        # `contracts/pipeline_contract.toml`. Production callers transparently
        # get the path-base-aware substitution.
        try:
            from hft_ops.paths import PipelinePaths
            for ancestor in [manifest_path, *manifest_path.parents]:
                if (ancestor / "contracts" / "pipeline_contract.toml").exists():
                    paths = PipelinePaths(pipeline_root=ancestor)
                    break
            else:
                paths = None
        except Exception:
            paths = None

    with open(manifest_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Manifest is empty or not a dict: {manifest_path}")

    raw = _resolve_variables(raw, now=now, paths=paths)

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
        post_training_gate=_build_post_training_gate(
            stages_raw.get("post_training_gate", {})
        ),
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
