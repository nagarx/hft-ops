"""Phase 5 FULL-A Block 4: post-expansion variable-resolution pass.

Tests ``hft_ops.manifest.resolver.resolve_variables_in_manifest`` + the
``VarResolutionContext`` dataclass. These exercise the silent-bug fix from
CRITICAL-FIX 5: a sweep axis changing ``stages.extraction.output_dir`` must
cause downstream string references like ``stages.backtesting.data_dir =
"${stages.extraction.output_dir}"`` to re-bind per grid point.

Tests are isolated — no cross-module dependencies; they build concrete
``ExperimentManifest`` instances directly and call the resolver.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hft_ops.manifest.resolver import (
    VarResolutionContext,
    resolve_variables_in_manifest,
)
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
    TrainingStage,
    ValidationStage,
)


def _make_manifest(
    *,
    name: str = "test_exp",
    extraction_output_dir: str = "data/exports/base",
    backtesting_data_dir: str = "${stages.extraction.output_dir}",
    training_output_dir: str = "",
    pipeline_root: str = "..",
) -> ExperimentManifest:
    """Build a concrete manifest with a ${...} reference for re-resolution."""
    return ExperimentManifest(
        experiment=ExperimentHeader(
            name=name,
            description="",
            hypothesis="",
            contract_version="2.2",
            tags=[],
        ),
        pipeline_root=pipeline_root,
        stages=Stages(
            extraction=ExtractionStage(
                enabled=True,
                skip_if_exists=True,
                config="feature-extractor-MBO-LOB/configs/e5_timebased_60s.toml",
                output_dir=extraction_output_dir,
            ),
            raw_analysis=RawAnalysisStage(enabled=False),
            dataset_analysis=DatasetAnalysisStage(enabled=False),
            validation=ValidationStage(enabled=False),
            training=TrainingStage(
                enabled=True,
                config="",
                trainer_config={"model": {"model_type": "tlob"}},
                output_dir=training_output_dir,
            ),
            signal_export=SignalExportStage(enabled=False),
            backtesting=BacktestingStage(
                enabled=True,
                script="scripts/backtest_deeplob.py",
                data_dir=backtesting_data_dir,
                params=BacktestParams(),
            ),
        ),
        sweep=None,
        manifest_path="/dev/null/test_manifest.yaml",
    )


# -----------------------------------------------------------------------------
# Core contract: cross-stage ${...} re-resolution per grid point
# -----------------------------------------------------------------------------


class TestCrossStageReResolution:
    def test_extraction_output_dir_change_propagates_to_backtesting_data_dir(self):
        """CRITICAL-FIX 5 regression guard.

        A sweep axis changing ``stages.extraction.output_dir`` must cause
        ``stages.backtesting.data_dir = "${stages.extraction.output_dir}"`` to
        re-resolve per-point. Pre-fix this was a silent bug.
        """
        m = _make_manifest(
            extraction_output_dir="data/exports/e5_60s",
            backtesting_data_dir="${stages.extraction.output_dir}",
        )
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        resolved = resolve_variables_in_manifest(m, ctx)
        assert resolved.stages.backtesting.data_dir == "data/exports/e5_60s", (
            f"backtesting.data_dir should re-resolve to the (mutated) extraction path, "
            f"got {resolved.stages.backtesting.data_dir!r}"
        )

    def test_distinct_extraction_paths_produce_distinct_backtest_paths(self):
        """Two grid points with different extraction paths must see DIFFERENT
        backtest data paths after resolution."""
        m1 = _make_manifest(name="p1", extraction_output_dir="data/exports/e5_30s")
        m2 = _make_manifest(name="p2", extraction_output_dir="data/exports/e5_120s")
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r1 = resolve_variables_in_manifest(m1, ctx)
        r2 = resolve_variables_in_manifest(m2, ctx)
        assert r1.stages.backtesting.data_dir == "data/exports/e5_30s"
        assert r2.stages.backtesting.data_dir == "data/exports/e5_120s"
        assert r1.stages.backtesting.data_dir != r2.stages.backtesting.data_dir


# -----------------------------------------------------------------------------
# Sweep extra_vars injection
# -----------------------------------------------------------------------------


class TestSweepExtraVars:
    def test_sweep_point_name_injection(self):
        """${sweep.point_name} resolves via ctx.extra_vars without mutating the
        manifest dict (invariant of the extra_vars design)."""
        m = _make_manifest(
            extraction_output_dir="outputs/${sweep.point_name}/export",
        )
        ctx = VarResolutionContext(
            now=datetime(2026, 4, 17, tzinfo=timezone.utc),
            extra_vars={"sweep": {"point_name": "e5_phase2__on_d12_6"}},
        )
        r = resolve_variables_in_manifest(m, ctx)
        assert r.stages.extraction.output_dir == "outputs/e5_phase2__on_d12_6/export"

    def test_sweep_axis_values_nested_lookup(self):
        """${sweep.axis_values.cvml} resolves via extra_vars nested dict walk."""
        m = _make_manifest(
            extraction_output_dir="outputs/cvml_${sweep.axis_values.cvml}",
        )
        ctx = VarResolutionContext(
            now=datetime(2026, 4, 17, tzinfo=timezone.utc),
            extra_vars={"sweep": {"point_name": "x", "axis_values": {"cvml": "on"}}},
        )
        r = resolve_variables_in_manifest(m, ctx)
        assert r.stages.extraction.output_dir == "outputs/cvml_on"

    def test_sweep_key_stripped_from_rebuilt_manifest(self):
        """Injected `sweep` extras MUST NOT leak back into the rebuilt manifest
        (it would shadow the concrete manifest's `sweep: None` state)."""
        m = _make_manifest()
        ctx = VarResolutionContext(
            now=datetime(2026, 4, 17, tzinfo=timezone.utc),
            extra_vars={"sweep": {"point_name": "p1"}},
        )
        r = resolve_variables_in_manifest(m, ctx)
        assert r.sweep is None, "extras must not pollute manifest.sweep"


# -----------------------------------------------------------------------------
# Idempotency + deep chains (CRITICAL-FIX 5 T4g)
# -----------------------------------------------------------------------------


class TestIdempotencyAndDeepChains:
    def test_idempotent_under_repeated_application(self):
        """T4e: resolve twice → same result."""
        m = _make_manifest(extraction_output_dir="data/exports/e5")
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r1 = resolve_variables_in_manifest(m, ctx)
        r2 = resolve_variables_in_manifest(r1, ctx)
        assert r1.stages.extraction.output_dir == r2.stages.extraction.output_dir
        assert r1.stages.backtesting.data_dir == r2.stages.backtesting.data_dir

    def test_three_deep_variable_chain(self):
        """T4g: ${a} references ${b} which references ${c} — all resolve.

        Uses ${experiment.name} → name="x" AND training_output_dir
        containing ${stages.extraction.output_dir} which is literal. Direct
        3-deep chains aren't expressible in the manifest schema (no free
        string key tree), so we test that transitive resolution through the
        existing fixed-point loop works for the 2-level case chain that IS
        expressible: training.output_dir → extraction.output_dir → literal.
        """
        m = _make_manifest(
            extraction_output_dir="outputs/${experiment.name}/export",
            training_output_dir="${stages.extraction.output_dir}/trained",
        )
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r = resolve_variables_in_manifest(m, ctx)
        # Both levels resolve in a single call thanks to max_passes=5 fixed-point loop
        assert r.stages.extraction.output_dir == "outputs/test_exp/export"
        assert r.stages.training.output_dir == "outputs/test_exp/export/trained"


# -----------------------------------------------------------------------------
# Invariant preservation
# -----------------------------------------------------------------------------


class TestInvariantPreservation:
    def test_manifest_path_preserved(self):
        """T4h: asdict/rebuild drops manifest_path (it's set as a loader side-effect);
        resolver explicitly preserves it."""
        m = _make_manifest()
        assert m.manifest_path == "/dev/null/test_manifest.yaml"
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r = resolve_variables_in_manifest(m, ctx)
        assert r.manifest_path == "/dev/null/test_manifest.yaml"

    def test_deferred_prefix_untouched(self):
        """T4d: ${resolved.*} is left for runtime resolution (stage-runner handles)."""
        m = _make_manifest(
            backtesting_data_dir="outputs/${resolved.horizon_idx}/backtest",
        )
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r = resolve_variables_in_manifest(m, ctx)
        assert r.stages.backtesting.data_dir == "outputs/${resolved.horizon_idx}/backtest"

    def test_no_sweep_context_preserves_literal_timestamp_markers(self):
        """T4f: a resolver call without sweep context still works (just rebinds
        timestamp/date if present)."""
        m = _make_manifest(
            training_output_dir="outputs/runs/${date}_${experiment.name}",
        )
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r = resolve_variables_in_manifest(m, ctx)
        assert r.stages.training.output_dir == "outputs/runs/2026-04-17_test_exp"

    def test_non_string_fields_pass_through_unchanged(self):
        """Booleans, ints, enabled flags MUST NOT be regex-substituted."""
        m = _make_manifest()
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        r = resolve_variables_in_manifest(m, ctx)
        assert r.stages.extraction.enabled is True
        assert r.stages.backtesting.enabled is True
        assert r.stages.validation.enabled is False
