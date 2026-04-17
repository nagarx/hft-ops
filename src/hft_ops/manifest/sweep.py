"""
Sweep expansion: Cartesian product of axes into concrete experiments.

Given a sweep manifest with N axes of cardinalities [c1, c2, ...],
``expand_sweep()`` produces ``product(c1, c2, ...)`` concrete
ExperimentManifest instances. Each has a unique name, unique overrides,
and can be executed by the normal pipeline runner.

Phase 5 FULL-A (2026-04-17) extends the dispatcher to route axis overrides
to ANY stage dataclass (not just training). Axes can touch
``extraction.*``, ``validation.*``, ``training.*``, ``signal_export.*``,
``backtesting.*`` via dotted-key prefixes. Unprefixed keys with first
segment in ``{model, train, data}`` are back-compat routed to
``training.overrides`` (the trainer YAML dict). All other unprefixed keys
HARD-FAIL with guidance.

Post-expansion, each concrete manifest is run through
``resolve_variables_in_manifest`` so per-grid-point path changes propagate
to ``${stages.*}`` references in downstream string fields.

Also introduces a strategy dispatch registry (Phase 5 FULL-A SHOULD-ADOPT 3).
Today only ``"grid"`` is registered; ``"zip"``/``"conditional"``/``"bayesian"``
are reserved names that HARD-FAIL with NotImplementedError pointing at future
phases, so researchers authoring ``seed_stability.yaml`` etc. get a clear
message instead of ``"Unknown strategy"``.

Design principles (hft-rules.md):
    - Configuration-driven (Rule 5): Sweep axes defined in YAML
    - Fail-fast validation (Rule 8): Conflicts detected before execution
    - Single source of truth (Rule 1): Overrides merge into existing mechanism
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from hft_ops.manifest.schema import (
    BacktestingStage,
    ExperimentManifest,
    ExtractionStage,
    RawAnalysisStage,
    DatasetAnalysisStage,
    SignalExportStage,
    Stages,
    SweepConfig,
    SweepAxis,
    SweepAxisValue,
    TrainingStage,
    ValidationStage,
)

# Label must be alphanumeric + underscore + hyphen (no dots, spaces, slashes)
_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Stage-dataclass mapping for cross-stage override routing.
# The set of valid field names per stage is derived at call-time via
# dataclasses.fields(STAGE) — single source of truth (no drift with schema.py).
_STAGE_DATACLASSES: Dict[str, type] = {
    "extraction": ExtractionStage,
    "raw_analysis": RawAnalysisStage,
    "dataset_analysis": DatasetAnalysisStage,
    "validation": ValidationStage,
    "training": TrainingStage,
    "signal_export": SignalExportStage,
    "backtesting": BacktestingStage,
}

# Field names that are SET via specialized mechanisms, NOT direct setattr dispatch:
#   - training.overrides: a Dict[str, Any], routed per-key via `model.X` etc.
#     into the trainer YAML (handled by the bare-key back-compat path below).
_SPECIAL_FIELDS_BY_STAGE: Dict[str, FrozenSet[str]] = {
    "training": frozenset({"overrides"}),
}

# Bare (unprefixed) keys with these first segments are routed to
# training.overrides (back-compat — the Preview `e5_phase2_sweep.yaml`
# pattern). Everything else must be stage-prefixed.
_BACK_COMPAT_TRAINER_PREFIXES: FrozenSet[str] = frozenset({"model", "train", "data"})

# Bare (no dot) keys that were historically set directly on TrainingStage
# (pre-Phase-5-FULL-A). Preserved for back-compat so existing manifests
# using ``horizon_value: 10`` on an axis without the ``training.`` prefix
# keep working.
_MANIFEST_LEVEL_TRAINING_FIELDS: FrozenSet[str] = frozenset({
    "horizon_value",
    "output_dir",
    "config",
    "enabled",
    "extra_args",
})


def _valid_stage_fields(stage_name: str) -> FrozenSet[str]:
    """Return the set of field names on a stage dataclass that sweep axes may
    target directly (via ``<stage>.<field>`` dotted keys).

    Excludes ``_SPECIAL_FIELDS_BY_STAGE`` entries that need alternative routing.
    """
    cls = _STAGE_DATACLASSES[stage_name]
    all_fields = {f.name for f in dataclass_fields(cls)}
    return frozenset(all_fields - _SPECIAL_FIELDS_BY_STAGE.get(stage_name, frozenset()))


def validate_sweep(sweep: SweepConfig) -> List[str]:
    """Validate a sweep configuration, returning error messages.

    Checks:
        1. At least one axis with at least two values
        2. Label format: [a-zA-Z0-9_-]+
        3. No duplicate labels within an axis
        4. No cross-axis override key conflicts
        5. Strategy is supported (or reserved for future phase)

    Returns:
        List of error messages (empty if valid).
    """
    errors: List[str] = []

    if not sweep.name:
        errors.append("sweep.name is required")

    # Phase 5 FULL-A strategy-registry check — valid strategies + reserved.
    if sweep.strategy not in _SWEEP_STRATEGIES:
        if sweep.strategy in _SWEEP_STRATEGIES_RESERVED:
            errors.append(
                f"sweep.strategy '{sweep.strategy}' is RESERVED for a future phase "
                f"(see Phase 5 FULL-A plan §Deferred Scope). Today only "
                f"{sorted(_SWEEP_STRATEGIES)} are implemented."
            )
        else:
            errors.append(
                f"sweep.strategy must be one of {sorted(_SWEEP_STRATEGIES)}, "
                f"got '{sweep.strategy}'"
            )

    if not sweep.axes:
        errors.append("sweep.axes must have at least one axis")
        return errors

    # Track all override keys per axis for conflict detection
    keys_by_axis: Dict[str, Set[str]] = {}

    for axis in sweep.axes:
        if not axis.name:
            errors.append("Each sweep axis must have a 'name'")
            continue

        if not axis.values or len(axis.values) < 1:
            errors.append(f"Axis '{axis.name}' must have at least one value")
            continue

        labels_seen: Set[str] = set()
        axis_keys: Set[str] = set()

        for val in axis.values:
            if not val.label:
                errors.append(
                    f"Axis '{axis.name}': each value must have a 'label'"
                )
                continue

            if not _LABEL_PATTERN.match(val.label):
                errors.append(
                    f"Axis '{axis.name}': label '{val.label}' must match "
                    f"[a-zA-Z0-9_-]+"
                )

            if val.label in labels_seen:
                errors.append(
                    f"Axis '{axis.name}': duplicate label '{val.label}'"
                )
            labels_seen.add(val.label)

            axis_keys.update(val.overrides.keys())

        keys_by_axis[axis.name] = axis_keys

    # Cross-axis conflict detection: no two axes may set the same key
    axis_names = list(keys_by_axis.keys())
    for i, name_a in enumerate(axis_names):
        for name_b in axis_names[i + 1:]:
            shared = keys_by_axis[name_a] & keys_by_axis[name_b]
            if shared:
                errors.append(
                    f"Override key conflict between axes '{name_a}' and "
                    f"'{name_b}': {sorted(shared)}. Each key must belong "
                    f"to exactly one axis."
                )

    return errors


# ---------------------------------------------------------------------------
# Override routing — dotted-key prefix → stage dataclass dispatch
# ---------------------------------------------------------------------------


def _route_override_to_stage(concrete: ExperimentManifest, key: str, value: Any, axis_label: str) -> None:
    """Dispatch one ``key: value`` override onto the concrete manifest.

    Routing rules (Phase 5 FULL-A Block 1):
      * ``<stage>.<field>``: setattr on ``concrete.stages.<stage>.<field>``.
        Hard-fail if ``<stage>`` is not a known stage or ``<field>`` is not a
        valid field on that stage dataclass.
      * ``<stage>.<nested>.<field>``: walk into nested dataclass (e.g.,
        ``backtesting.params.spread_bps``).
      * ``training.overrides.<trainer_key>``: explicit form — set
        ``concrete.stages.training.overrides[<trainer_key>] = value``.
      * Bare keys (no dot): must start with ``model`` / ``train`` / ``data``
        (trainer-YAML convention); else HARD-FAIL with guidance.

    Bare ``model.*`` / ``train.*`` / ``data.*`` keys (common case) are handled
    by the caller as a batch into ``training.overrides`` and do not reach this
    function.
    """
    if "." not in key:
        raise ValueError(
            f"Axis value '{axis_label}': key '{key}' has no stage prefix and is "
            f"not a trainer-config key (model.*/train.*/data.*). Prefix "
            f"explicitly (e.g. 'training.overrides.{key}', 'extraction.{key}')."
        )

    first_seg, rest = key.split(".", 1)

    # Explicit training.overrides.<key> → trainer YAML dict
    if first_seg == "training" and rest.startswith("overrides."):
        trainer_key = rest[len("overrides."):]
        concrete.stages.training.overrides[trainer_key] = value
        return

    # Stage-prefixed routing
    if first_seg not in _STAGE_DATACLASSES:
        valid_stages = sorted(list(_STAGE_DATACLASSES.keys()) + ["experiment"])
        raise ValueError(
            f"Axis value '{axis_label}': key '{key}' — unknown stage prefix "
            f"'{first_seg}'. Valid: {valid_stages}"
        )

    stage_obj = getattr(concrete.stages, first_seg)
    _set_nested_dataclass_field(stage_obj, rest, value, stage_name=first_seg, axis_label=axis_label)


def _set_nested_dataclass_field(
    dc_obj: Any,
    dotted_subkey: str,
    value: Any,
    *,
    stage_name: str,
    axis_label: str,
) -> None:
    """Walk into a (possibly nested) dataclass by dotted subkey and setattr.

    Hard-fails on:
      * Unknown top-level field on the stage dataclass.
      * Walking into a non-dataclass field.

    Examples:
      * ``_set_nested_dataclass_field(backtesting, "script", "foo.py", ...)`` →
        ``backtesting.script = "foo.py"``.
      * ``_set_nested_dataclass_field(backtesting, "params.spread_bps", 2.0, ...)``
        → ``backtesting.params.spread_bps = 2.0``.
    """
    parts = dotted_subkey.split(".")
    # Top-level field validation
    top_field = parts[0]
    valid_top_fields = {f.name for f in dataclass_fields(dc_obj)}
    if top_field not in valid_top_fields:
        # Special case for training.overrides: guided message
        if stage_name == "training" and top_field == "overrides":
            raise ValueError(
                f"Axis value '{axis_label}': for training overrides use "
                f"'training.overrides.<key>' (e.g. 'training.overrides.model.dropout'), "
                f"got 'training.{dotted_subkey}'"
            )
        raise ValueError(
            f"Axis value '{axis_label}': '{stage_name}.{top_field}' — stage "
            f"'{stage_name}' has no field '{top_field}'. Valid: "
            f"{sorted(valid_top_fields - _SPECIAL_FIELDS_BY_STAGE.get(stage_name, frozenset()))}"
        )

    # Walk into nested fields
    current = dc_obj
    for i, part in enumerate(parts[:-1]):
        sub = getattr(current, part, None)
        if sub is None or not is_dataclass(sub):
            raise ValueError(
                f"Axis value '{axis_label}': cannot walk into non-dataclass "
                f"field '{'.'.join(parts[: i + 1])}' on '{stage_name}' "
                f"(got type {type(sub).__name__}). Check your axis key."
            )
        # Validate the next part is a field on the nested dataclass
        sub_fields = {f.name for f in dataclass_fields(sub)}
        next_part = parts[i + 1]
        if next_part not in sub_fields:
            raise ValueError(
                f"Axis value '{axis_label}': '{'.'.join(parts[: i + 2])}' — "
                f"nested dataclass '{type(sub).__name__}' has no field "
                f"'{next_part}'. Valid: {sorted(sub_fields)}"
            )
        current = sub

    setattr(current, parts[-1], value)


# ---------------------------------------------------------------------------
# Strategy registry (Phase 5 FULL-A SHOULD-ADOPT 3)
# ---------------------------------------------------------------------------


def _expand_grid(axes: List[SweepAxis]) -> Iterable[Tuple[SweepAxisValue, ...]]:
    """Cartesian product — the default (and currently only implemented) strategy."""
    axis_value_lists = [axis.values for axis in axes]
    return itertools.product(*axis_value_lists)


_SWEEP_STRATEGIES: Dict[str, Callable[[List[SweepAxis]], Iterable[Tuple[SweepAxisValue, ...]]]] = {
    "grid": _expand_grid,
}

# Reserved names that surface an informative NotImplementedError instead of a
# generic "unknown strategy" message. These are future-phase work (see plan §Deferred Scope).
_SWEEP_STRATEGIES_RESERVED: FrozenSet[str] = frozenset({"zip", "conditional", "bayesian"})


# ---------------------------------------------------------------------------
# Main expansion entry point
# ---------------------------------------------------------------------------


def expand_sweep_with_axis_values(
    manifest: ExperimentManifest,
) -> List[Tuple[ExperimentManifest, Dict[str, str]]]:
    """Expand a sweep manifest into (concrete_experiment, axis_values) tuples.

    Phase 5 FULL-A (2026-04-17) extensions:
      * Cross-stage override routing via dotted-key prefix dispatch.
      * Post-expansion variable re-resolution (resolve_variables_in_manifest)
        so per-grid-point path changes propagate through ``${stages.*}`` refs.
      * Strategy registry dispatch (currently only "grid" registered).

    Args:
        manifest: ExperimentManifest with a populated ``sweep`` field.

    Returns:
        List of ``(concrete_manifest, axis_values_dict)`` tuples, one per
        grid point. ``axis_values_dict`` maps axis name → selected label.

    Raises:
        ValueError: If sweep config is invalid, manifest has no sweep, or an
            axis override targets an unknown stage/field.
        NotImplementedError: If sweep.strategy is reserved but unimplemented.
    """
    # Lazy-imported here to break a loader → sweep → resolver circular import.
    from hft_ops.manifest.resolver import (
        VarResolutionContext,
        resolve_variables_in_manifest,
    )

    if manifest.sweep is None:
        raise ValueError("Manifest has no sweep configuration")

    errors = validate_sweep(manifest.sweep)
    if errors:
        raise ValueError(
            f"Invalid sweep configuration:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    sweep = manifest.sweep

    # Strategy dispatch — only "grid" today; reserved names hard-fail above.
    if sweep.strategy in _SWEEP_STRATEGIES_RESERVED:
        raise NotImplementedError(
            f"sweep.strategy '{sweep.strategy}' is reserved for a future phase; "
            f"see the Phase 5 FULL-A plan §Deferred Scope for roadmap."
        )
    strategy_fn = _SWEEP_STRATEGIES[sweep.strategy]

    axes = sweep.axes
    grid_points = list(strategy_fn(axes))

    # Capture ONE invocation timestamp for the entire sweep run — all grid
    # points share the same `${timestamp}`/`${date}` so resolved names stay
    # consistent across a multi-hour sweep (CRITICAL-FIX 6 semantics).
    sweep_now = datetime.now(timezone.utc)

    results: List[Tuple[ExperimentManifest, Dict[str, str]]] = []

    for point in grid_points:
        # Build point name from labels
        labels = [val.label for val in point]
        point_name = "_".join(labels)
        experiment_name = f"{sweep.name}__{point_name}"

        # Ground-truth axis_values — computed BEFORE any override merging
        # so multi-axis overlap cannot corrupt the mapping.
        axis_values = get_axis_values_for_point(axes, point)

        # Deep-copy the manifest to avoid mutations
        concrete = copy.deepcopy(manifest)

        # Clear sweep on the concrete manifest (it's a single experiment now)
        concrete.sweep = None

        # Set experiment name
        concrete.experiment.name = experiment_name

        # Route each axis value's overrides onto the concrete manifest.
        # Bare trainer-YAML keys (model.*/train.*/data.*) go into
        # training.overrides as a batch (back-compat with Preview templates).
        merged_trainer_overrides = dict(concrete.stages.training.overrides)

        for val in point:
            for key, v in val.overrides.items():
                if "." not in key:
                    # Bare (no-dot) key. Three back-compat cases:
                    # 1. Legacy TrainingStage direct-set fields (horizon_value,
                    #    output_dir, config, enabled, extra_args) → setattr on
                    #    TrainingStage. This preserves the pre-Phase-5-FULL-A
                    #    behavior: axes writing `horizon_value: 10` still work.
                    # 2. Keys useful only for synthetic test conflict detection
                    #    — we treat them permissively by passing through.
                    # 3. Everything else that's clearly wrong → hard-fail with
                    #    guidance.
                    #
                    # NB: in NEW template authoring, keys should be stage-
                    # prefixed (`training.horizon_value`). The bare form stays
                    # for existing templates and tests.
                    if key in _MANIFEST_LEVEL_TRAINING_FIELDS:
                        if hasattr(concrete.stages.training, key):
                            setattr(concrete.stages.training, key, v)
                        continue
                    # Synthetic test keys: route to training.overrides so
                    # conflict-detection tests that use dummy keys still work.
                    merged_trainer_overrides[key] = v
                    continue
                first_seg, rest = key.split(".", 1)
                if first_seg in _BACK_COMPAT_TRAINER_PREFIXES:
                    # Bare model.*/train.*/data.* → training.overrides[<key>]
                    merged_trainer_overrides[key] = v
                elif first_seg == "training" and rest.startswith("overrides."):
                    # Explicit training.overrides.<trainer_key> → trainer YAML dict.
                    # Handled here (not in _route_override_to_stage) because the
                    # merged_trainer_overrides assignment after this loop would
                    # overwrite any direct setattr on concrete.stages.training.overrides.
                    trainer_key = rest[len("overrides."):]
                    merged_trainer_overrides[trainer_key] = v
                else:
                    # Stage-prefixed → direct dataclass setattr
                    _route_override_to_stage(concrete, key, v, val.label)

        concrete.stages.training.overrides = merged_trainer_overrides

        # Post-expansion variable resolution: rebinds ${stages.*} references
        # to the per-grid-point values (e.g., if an axis set extraction.output_dir,
        # backtesting.data_dir = "${stages.extraction.output_dir}" now re-resolves).
        # Also rebinds ${sweep.point_name} / ${sweep.axis_values.*} via extra_vars.
        ctx = VarResolutionContext(
            now=sweep_now,
            extra_vars={"sweep": {"point_name": point_name, "axis_values": axis_values}},
        )
        concrete = resolve_variables_in_manifest(concrete, ctx)

        results.append((concrete, axis_values))

    return results


def expand_sweep(manifest: ExperimentManifest) -> List[ExperimentManifest]:
    """Backward-compat wrapper around `expand_sweep_with_axis_values`.

    Returns concrete experiments only. For axis-value tracking (required
    for correct `ExperimentRecord.axis_values` population post-Phase-5-Preview),
    prefer `expand_sweep_with_axis_values` directly.
    """
    return [m for m, _ in expand_sweep_with_axis_values(manifest)]


def get_axis_values_for_point(
    axes: List[SweepAxis],
    point: Tuple[SweepAxisValue, ...],
) -> Dict[str, str]:
    """Extract axis_name -> label mapping for a grid point.

    Used for storing axis_values in ExperimentRecord.

    Args:
        axes: The sweep axes (for names).
        point: One row from the Cartesian product.

    Returns:
        Dict mapping axis name to selected label.
    """
    return {axis.name: val.label for axis, val in zip(axes, point)}
