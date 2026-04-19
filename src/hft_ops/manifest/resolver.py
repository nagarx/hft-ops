"""
Variable resolution for concrete (post-expansion) manifests.

Introduced in Phase 5 FULL-A (2026-04-17) Block 4. Addresses the silent bug
where ``expand_sweep_with_axis_values`` mutates a concrete manifest's stage
fields AFTER ``load_manifest`` ran ``_resolve_variables`` once at load time.
If an axis changes ``stages.extraction.output_dir``, downstream strings like
``stages.backtesting.data_dir = "${stages.extraction.output_dir}"`` that were
already interpolated against the BASE manifest's value silently point at the
wrong path.

This module provides a post-expansion re-resolution pass that walks the
concrete ``ExperimentManifest``, serializes it to a dict (``asdict``),
delegates to the existing ``_resolve_variables`` machinery in ``loader.py``
(SSoT for dotted-path lookup + regex substitution + deferred-prefix handling),
and rebuilds the ``ExperimentManifest`` from the resolved dict.

**Why round-trip (asdict → resolve → rebuild), not dataclass-tree walk?**
The existing ``_resolve_variables`` uses ``_get_nested(raw, dotted_key)`` which
walks DICTS. Re-implementing an equivalent walker that navigates dataclass
attributes + Dict[str, Any] overrides + List[str] extra_args + Optional[Stage]
None-checks would duplicate ~100 LOC of lookup semantics, drift over time, and
break when a researcher adds a new Union field. Round-trip reuses the SSoT
for the regex machinery AND the lookup semantics. Cost: one ``asdict`` +
rebuild per concrete manifest (cheap compared to extraction/training subprocess
startup).

**Phase 11 reuse**: the BQP Phase 10 design (§15.1a) will require a similar
variable-resolution capability on concrete manifests for envelope serialization.
This module is the single call-site for that logic; the future envelope
serializer imports ``resolve_variables_in_manifest`` rather than re-implementing
walker semantics.

The module also carefully preserves ``manifest.manifest_path`` (which is set
by ``load_manifest`` as a side-effect and is NOT a dataclass field that gets
round-tripped via asdict).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VarResolutionContext:
    """Context for post-expansion variable resolution.

    Named ``VarResolutionContext`` (not ``ResolutionContext``) to avoid
    collision with the existing ``ResolvedContext`` in
    ``hft_ops.manifest.validator`` which handles horizon_idx resolution
    (a different concern — that maps ``training.horizon_value`` to
    ``backtesting.horizon_idx``; this one substitutes ``${...}`` markers in
    string fields).

    Attributes:
        now: Timestamp for ``${timestamp}`` / ``${date}`` resolution.
            In sweep contexts, this should be the sweep-invocation time
            (bound ONCE for the entire sweep, not per grid point — so all
            grid points share the same timestamp-derived names).
        extra_vars: Additional variable bindings beyond ``timestamp``/``date``
            and the manifest's own fields. Primary use: injecting
            ``{"sweep": {"point_name": ..., "axis_values": {...}}}`` so
            templates can reference ``${sweep.point_name}`` and
            ``${sweep.axis_values.<axis_name>}``.

    Note on ``${timestamp}``/``${date}`` semantics:
        These are LOAD-TIME variables consumed by ``_resolve_variables`` at
        ``loader.py::load_manifest``. After load, they are already literal
        strings in the manifest dataclass. The post-expansion pass using
        ``now`` rebinds them only if they survived the load pass (e.g., if
        an axis override introduced a new string containing ``${timestamp}``).
        Existing already-resolved timestamp strings are preserved verbatim.
        This matches CRITICAL-FIX 6 in the Phase 5 FULL-A plan.
    """

    now: datetime
    extra_vars: Dict[str, Any] = field(default_factory=dict)


def resolve_variables_in_manifest(manifest, ctx: VarResolutionContext):
    """Re-resolve ``${...}`` references in a concrete ``ExperimentManifest``.

    Used by ``expand_sweep_with_axis_values`` AFTER applying axis overrides
    so per-grid-point path changes propagate to all string references.

    Strategy: ``asdict`` → ``_resolve_variables`` → ``_build_*`` rebuild.
    Reuses the SSoT regex + lookup machinery from ``loader.py``. Idempotent
    under repeated application (inherited from ``_resolve_variables`` which
    uses a fixed-point loop with ``max_passes=5``).

    Args:
        manifest: The concrete ``ExperimentManifest`` (typically the deep-
            copied per-grid-point manifest from ``expand_sweep``).
        ctx: Resolution context with timestamp + extra variable bindings.

    Returns:
        A new ``ExperimentManifest`` with string fields re-resolved.
        ``manifest.manifest_path`` is preserved from the input.

    Invariants preserved:
        * Deferred prefixes (``${resolved.*}``) remain unresolved (stage
          runners handle these at execution time).
        * Non-string fields (int, float, bool, None, List[dataclass]) pass
          through unchanged.
        * ``sweep`` field stays None (axis expansion clears it).
        * ``manifest_path`` is NOT a dataclass field that round-trips through
          asdict; preserved via explicit copy.
    """
    # Lazy import to avoid circular dep (loader imports schema; this module
    # is called from sweep which is called from cli; loader is also imported
    # by cli). Same-module lazy-import pattern used elsewhere (dedup.py).
    from hft_ops.manifest.loader import (  # noqa: F401 — helpers imported lazy
        _build_post_training_gate,
        _resolve_variables,
        _build_extraction,
        _build_raw_analysis,
        _build_dataset_analysis,
        _build_validation,
        _build_training,
        _build_signal_export,
        _build_backtesting,
        _build_sweep,
    )
    from hft_ops.manifest.schema import (
        ExperimentHeader,
        ExperimentManifest,
        Stages,
    )

    # Step 1: serialize concrete manifest to dict (drops manifest_path side-effect).
    raw: Dict[str, Any] = asdict(manifest)

    # Step 2: run the existing regex machinery. This is the SSoT: handles
    # _DEFERRED_PREFIXES, builtin vars (timestamp/date), dotted-path lookup
    # via _get_nested, and the max_passes=5 fixed-point loop for transitive refs.
    # extra_vars is passed through — _resolve_variables consults it BEFORE the
    # raw-dict lookup, enabling ${sweep.point_name} / ${sweep.axis_values.*}
    # without polluting the manifest dict.
    raw = _resolve_variables(raw, now=ctx.now, extra_vars=ctx.extra_vars)

    # Step 5: rebuild manifest via the canonical _build_* helpers (reuse).
    experiment_raw = raw.get("experiment", {})
    header = ExperimentHeader(
        name=experiment_raw.get("name", ""),
        description=experiment_raw.get("description", ""),
        hypothesis=experiment_raw.get("hypothesis", ""),
        contract_version=experiment_raw.get("contract_version", ""),
        tags=list(experiment_raw.get("tags", [])),
    )
    stages_raw = raw.get("stages", {})
    stages = Stages(
        extraction=_build_extraction(stages_raw.get("extraction", {})),
        raw_analysis=_build_raw_analysis(stages_raw.get("raw_analysis", {})),
        dataset_analysis=_build_dataset_analysis(stages_raw.get("dataset_analysis", {})),
        validation=_build_validation(stages_raw.get("validation", {})),
        training=_build_training(stages_raw.get("training", {})),
        post_training_gate=_build_post_training_gate(
            stages_raw.get("post_training_gate", {})
        ),
        signal_export=_build_signal_export(stages_raw.get("signal_export", {})),
        backtesting=_build_backtesting(stages_raw.get("backtesting", {})),
    )

    sweep_raw = raw.get("sweep")
    sweep = None
    if sweep_raw and isinstance(sweep_raw, dict) and sweep_raw.get("axes"):
        # Only rebuild a SweepConfig if the serialized dict actually has a sweep;
        # concrete per-grid-point manifests typically have sweep=None after
        # expansion (sweep.py explicitly sets concrete.sweep = None).
        sweep = _build_sweep(sweep_raw)

    rebuilt = ExperimentManifest(
        experiment=header,
        pipeline_root=raw.get("pipeline_root", ".."),
        stages=stages,
        sweep=sweep,
        manifest_path=getattr(manifest, "manifest_path", ""),  # preserve side-effect field
    )
    return rebuilt
