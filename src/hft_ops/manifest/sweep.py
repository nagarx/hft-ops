"""
Sweep expansion: Cartesian product of axes into concrete experiments.

Given a sweep manifest with N axes of cardinalities [c1, c2, ...],
``expand_sweep()`` produces ``product(c1, c2, ...)`` concrete
ExperimentManifest instances. Each has a unique name, unique overrides,
and can be executed by the normal pipeline runner.

Design principles (hft-rules.md):
    - Configuration-driven (Rule 5): Sweep axes defined in YAML
    - Fail-fast validation (Rule 8): Conflicts detected before execution
    - Single source of truth (Rule 1): Overrides merge into existing mechanism
"""

from __future__ import annotations

import copy
import itertools
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from hft_ops.manifest.schema import (
    ExperimentManifest,
    SweepConfig,
    SweepAxis,
    SweepAxisValue,
)

# Label must be alphanumeric + underscore + hyphen (no dots, spaces, slashes)
_LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Manifest-level fields on TrainingStage that are NOT trainer-YAML overrides.
# These are set directly on the TrainingStage dataclass, not merged into
# stages.training.overrides.
_MANIFEST_LEVEL_TRAINING_FIELDS = frozenset({
    "horizon_value",
    "output_dir",
    "config",
    "enabled",
    "extra_args",
})


def validate_sweep(sweep: SweepConfig) -> List[str]:
    """Validate a sweep configuration, returning error messages.

    Checks:
        1. At least one axis with at least two values
        2. Label format: [a-zA-Z0-9_-]+
        3. No duplicate labels within an axis
        4. No cross-axis override key conflicts
        5. Strategy is supported

    Returns:
        List of error messages (empty if valid).
    """
    errors: List[str] = []

    if not sweep.name:
        errors.append("sweep.name is required")

    if sweep.strategy not in ("grid",):
        errors.append(
            f"sweep.strategy must be 'grid', got '{sweep.strategy}'"
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


def expand_sweep(manifest: ExperimentManifest) -> List[ExperimentManifest]:
    """Expand a sweep manifest into concrete experiments.

    Computes the Cartesian product of all axes and produces one
    ExperimentManifest per grid point. Each has:
        - Unique experiment name: ``{sweep_name}__{label1}_{label2}``
        - Merged overrides (base + axis values)
        - Unique output_dir incorporating the point name

    Args:
        manifest: ExperimentManifest with a populated ``sweep`` field.

    Returns:
        List of concrete ExperimentManifest instances ready for execution.

    Raises:
        ValueError: If sweep config is invalid or manifest has no sweep.
    """
    if manifest.sweep is None:
        raise ValueError("Manifest has no sweep configuration")

    errors = validate_sweep(manifest.sweep)
    if errors:
        raise ValueError(
            f"Invalid sweep configuration:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    sweep = manifest.sweep
    axes = sweep.axes

    # Build the grid: Cartesian product of axis values
    axis_value_lists = [axis.values for axis in axes]
    grid_points = list(itertools.product(*axis_value_lists))

    results: List[ExperimentManifest] = []

    for point in grid_points:
        # Build point name from labels
        labels = [val.label for val in point]
        point_name = "_".join(labels)
        experiment_name = f"{sweep.name}__{point_name}"

        # Deep-copy the manifest to avoid mutations
        concrete = copy.deepcopy(manifest)

        # Clear sweep on the concrete manifest (it's a single experiment now)
        concrete.sweep = None

        # Set experiment name
        concrete.experiment.name = experiment_name

        # Merge overrides: base manifest overrides first, then axis values
        merged_overrides = dict(concrete.stages.training.overrides)
        manifest_level_overrides: Dict[str, Any] = {}

        for val in point:
            for key, v in val.overrides.items():
                if key in _MANIFEST_LEVEL_TRAINING_FIELDS:
                    manifest_level_overrides[key] = v
                else:
                    merged_overrides[key] = v

        # Apply manifest-level overrides directly to TrainingStage
        if "horizon_value" in manifest_level_overrides:
            concrete.stages.training.horizon_value = manifest_level_overrides[
                "horizon_value"
            ]
        if "output_dir" in manifest_level_overrides:
            concrete.stages.training.output_dir = manifest_level_overrides[
                "output_dir"
            ]
        if "config" in manifest_level_overrides:
            concrete.stages.training.config = manifest_level_overrides["config"]

        # Set merged trainer-YAML overrides
        concrete.stages.training.overrides = merged_overrides

        # Auto-generate output_dir if not explicitly set by an axis
        if "output_dir" not in manifest_level_overrides:
            base_output = concrete.stages.training.output_dir
            if base_output:
                # Replace any ${sweep.point_name} placeholder
                concrete.stages.training.output_dir = base_output.replace(
                    "${sweep.point_name}", point_name
                )

        results.append(concrete)

    return results


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
