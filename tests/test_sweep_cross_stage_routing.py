"""Phase 5 FULL-A Block 1: cross-stage override routing.

Tests the new dotted-key prefix dispatch in
``hft_ops.manifest.sweep.expand_sweep_with_axis_values``:

  * Stage-prefixed keys (``extraction.*``, ``signal_export.*``,
    ``backtesting.*``, ``validation.*``, ``training.*``) route to the
    correct stage dataclass field.
  * Nested dataclass keys (``backtesting.params.spread_bps``) walk into
    the nested dataclass.
  * Bare ``model.*`` / ``train.*`` / ``data.*`` keys keep back-compat
    routing to ``stages.training.overrides``.
  * Bare legacy training-direct fields (``horizon_value``, ``output_dir``)
    keep back-compat to TrainingStage dataclass.
  * Unknown prefixes / unknown stage fields HARD-FAIL with guidance.
  * Post-expansion variable-resolution propagates per-grid-point changes.

Also covers the strategy registry shell (SHOULD-ADOPT 3).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    Stages,
    SweepAxis,
    SweepAxisValue,
    SweepConfig,
    TrainingStage,
)
from hft_ops.manifest.sweep import (
    expand_sweep,
    expand_sweep_with_axis_values,
    validate_sweep,
)


def _build_manifest(axes: List[SweepAxis], sweep_name: str = "t") -> ExperimentManifest:
    """Minimal manifest with a default base TrainingStage."""
    return ExperimentManifest(
        experiment=ExperimentHeader(name="t", contract_version="2.2"),
        stages=Stages(
            training=TrainingStage(
                enabled=True,
                config="configs/base.yaml",
                output_dir="out/${sweep.point_name}",
            ),
        ),
        sweep=SweepConfig(name=sweep_name, strategy="grid", axes=axes),
    )


# -----------------------------------------------------------------------------
# T1a-c: single stage-prefixed axis routes correctly
# -----------------------------------------------------------------------------


class TestSingleStagePrefix:
    def test_extraction_config_axis_routes_to_extraction_stage(self):
        """T1a: `extraction.config` axis → stages.extraction.config."""
        axes = [SweepAxis(name="bin", values=[
            SweepAxisValue(label="60s", overrides={"extraction.config": "configs/e5_60s.toml"}),
            SweepAxisValue(label="120s", overrides={"extraction.config": "configs/e5_120s.toml"}),
        ])]
        results = expand_sweep_with_axis_values(_build_manifest(axes))
        assert len(results) == 2
        m0, _ = results[0]
        m1, _ = results[1]
        assert m0.stages.extraction.config == "configs/e5_60s.toml"
        assert m1.stages.extraction.config == "configs/e5_120s.toml"

    def test_backtesting_params_nested_axis_routes_to_nested_dataclass(self):
        """T1b: `backtesting.params.spread_bps` walks into BacktestParams."""
        axes = [SweepAxis(name="cost", values=[
            SweepAxisValue(label="c_0_5", overrides={"backtesting.params.spread_bps": 0.5}),
            SweepAxisValue(label="c_2_0", overrides={"backtesting.params.spread_bps": 2.0}),
        ])]
        results = expand_sweep_with_axis_values(_build_manifest(axes))
        m0, _ = results[0]
        m1, _ = results[1]
        assert m0.stages.backtesting.params.spread_bps == 0.5
        assert m1.stages.backtesting.params.spread_bps == 2.0

    def test_signal_export_script_axis_routes_to_signal_export_stage(self):
        """T1c: `signal_export.script` axis → stages.signal_export.script."""
        axes = [SweepAxis(name="exporter", values=[
            SweepAxisValue(label="regular", overrides={"signal_export.script": "scripts/export_signals.py"}),
            SweepAxisValue(label="hmhp", overrides={"signal_export.script": "scripts/export_hmhp_signals.py"}),
        ])]
        results = expand_sweep_with_axis_values(_build_manifest(axes))
        m0, _ = results[0]
        m1, _ = results[1]
        assert m0.stages.signal_export.script == "scripts/export_signals.py"
        assert m1.stages.signal_export.script == "scripts/export_hmhp_signals.py"


# -----------------------------------------------------------------------------
# T1d: hard-fail with guidance
# -----------------------------------------------------------------------------


class TestHardFailOnUnknown:
    def test_unknown_stage_prefix_fails_with_stage_list(self):
        axes = [SweepAxis(name="x", values=[
            SweepAxisValue(label="a", overrides={"analytics.bogus": 1}),
            SweepAxisValue(label="b", overrides={"analytics.bogus": 2}),
        ])]
        with pytest.raises(ValueError, match="unknown stage prefix 'analytics'"):
            expand_sweep_with_axis_values(_build_manifest(axes))

    def test_unknown_stage_field_fails_with_field_list(self):
        axes = [SweepAxis(name="x", values=[
            SweepAxisValue(label="a", overrides={"extraction.does_not_exist": 1}),
            SweepAxisValue(label="b", overrides={"extraction.does_not_exist": 2}),
        ])]
        with pytest.raises(ValueError, match="stage 'extraction' has no field 'does_not_exist'"):
            expand_sweep_with_axis_values(_build_manifest(axes))

    def test_walking_into_non_dataclass_field_fails(self):
        """Attempting to walk into a str/int field (e.g. extraction.config.foo)
        produces a clear error."""
        axes = [SweepAxis(name="x", values=[
            SweepAxisValue(label="a", overrides={"extraction.config.foo": 1}),
            SweepAxisValue(label="b", overrides={"extraction.config.foo": 2}),
        ])]
        with pytest.raises(ValueError, match="cannot walk into non-dataclass"):
            expand_sweep_with_axis_values(_build_manifest(axes))

    def test_backtest_params_unknown_nested_field_fails(self):
        axes = [SweepAxis(name="x", values=[
            SweepAxisValue(label="a", overrides={"backtesting.params.nonexistent": 1}),
            SweepAxisValue(label="b", overrides={"backtesting.params.nonexistent": 2}),
        ])]
        with pytest.raises(ValueError, match="has no field 'nonexistent'"):
            expand_sweep_with_axis_values(_build_manifest(axes))


# -----------------------------------------------------------------------------
# T1e: explicit training.overrides.<key> equivalent to unprefixed model.<key>
# -----------------------------------------------------------------------------


class TestTrainerOverridesEquivalence:
    def test_explicit_training_overrides_prefix_equivalent_to_bare(self):
        """`training.overrides.model.dropout` sets the same trainer-YAML field
        as the bare `model.dropout` (Preview-style)."""
        axes_bare = [SweepAxis(name="d", values=[
            SweepAxisValue(label="d1", overrides={"model.dropout": 0.1}),
            SweepAxisValue(label="d2", overrides={"model.dropout": 0.2}),
        ])]
        axes_prefixed = [SweepAxis(name="d", values=[
            SweepAxisValue(label="d1", overrides={"training.overrides.model.dropout": 0.1}),
            SweepAxisValue(label="d2", overrides={"training.overrides.model.dropout": 0.2}),
        ])]
        r_bare = expand_sweep_with_axis_values(_build_manifest(axes_bare))
        r_prefixed = expand_sweep_with_axis_values(_build_manifest(axes_prefixed))

        m_bare_0, _ = r_bare[0]
        m_pref_0, _ = r_prefixed[0]
        assert m_bare_0.stages.training.overrides.get("model.dropout") == 0.1
        assert m_pref_0.stages.training.overrides.get("model.dropout") == 0.1


# -----------------------------------------------------------------------------
# T1f: two-axis cross-stage Cartesian with post-expansion re-resolution
# -----------------------------------------------------------------------------


class TestCrossStageCartesian:
    def test_extraction_axis_and_training_axis_full_grid(self):
        """Extraction axis × training axis → full 2×2 grid with correct
        per-point values on BOTH stages."""
        axes = [
            SweepAxis(name="bin", values=[
                SweepAxisValue(label="60s", overrides={"extraction.config": "configs/e5_60s.toml"}),
                SweepAxisValue(label="120s", overrides={"extraction.config": "configs/e5_120s.toml"}),
            ]),
            SweepAxis(name="cvml", values=[
                SweepAxisValue(label="off", overrides={"model.tlob_use_cvml": False}),
                SweepAxisValue(label="on", overrides={"model.tlob_use_cvml": True}),
            ]),
        ]
        results = expand_sweep_with_axis_values(_build_manifest(axes))
        assert len(results) == 4

        # Every combo present
        combos = set()
        for m, av in results:
            combos.add((m.stages.extraction.config, m.stages.training.overrides.get("model.tlob_use_cvml"),
                        av["bin"], av["cvml"]))
        assert combos == {
            ("configs/e5_60s.toml", False, "60s", "off"),
            ("configs/e5_60s.toml", True, "60s", "on"),
            ("configs/e5_120s.toml", False, "120s", "off"),
            ("configs/e5_120s.toml", True, "120s", "on"),
        }

    def test_per_grid_point_variable_rebinding(self):
        """CRITICAL-FIX 5 end-to-end: axis changes extraction.output_dir,
        backtesting.data_dir must re-resolve per grid point (not stay frozen
        at the base manifest's value)."""
        # Build manifest where backtesting.data_dir references extraction.output_dir
        m = ExperimentManifest(
            experiment=ExperimentHeader(name="t", contract_version="2.2"),
            stages=Stages(
                training=TrainingStage(enabled=True, config="base.yaml"),
            ),
            sweep=SweepConfig(name="t", strategy="grid", axes=[
                SweepAxis(name="bin", values=[
                    SweepAxisValue(label="60s", overrides={"extraction.output_dir": "data/exports/60s"}),
                    SweepAxisValue(label="120s", overrides={"extraction.output_dir": "data/exports/120s"}),
                ]),
            ]),
        )
        m.stages.backtesting.data_dir = "${stages.extraction.output_dir}/bt"

        results = expand_sweep_with_axis_values(m)
        assert len(results) == 2
        r0, _ = results[0]
        r1, _ = results[1]
        assert r0.stages.backtesting.data_dir == "data/exports/60s/bt"
        assert r1.stages.backtesting.data_dir == "data/exports/120s/bt"


# -----------------------------------------------------------------------------
# T1g: conflict detection
# -----------------------------------------------------------------------------


class TestConflictDetection:
    def test_two_axes_same_key_fail_validation(self):
        """Two axes setting the same dotted key → validate_sweep returns error."""
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="a1", overrides={"extraction.config": "c1.toml"}),
                SweepAxisValue(label="a2", overrides={"extraction.config": "c2.toml"}),
            ]),
            SweepAxis(name="b", values=[
                SweepAxisValue(label="b1", overrides={"extraction.config": "d1.toml"}),
                SweepAxisValue(label="b2", overrides={"extraction.config": "d2.toml"}),
            ]),
        ]
        errors = validate_sweep(SweepConfig(name="t", strategy="grid", axes=axes))
        assert any("Override key conflict" in e for e in errors), errors


# -----------------------------------------------------------------------------
# T1h: Preview back-compat (regression guard)
# -----------------------------------------------------------------------------


class TestPreviewBackCompat:
    def test_e5_phase2_pattern_still_works(self):
        """Preview's e5_phase2 template uses bare `model.tlob_use_cvml`
        (unprefixed). Block 1 must preserve this behavior — the sweep expansion
        should produce 4 grid points with correct trainer-overrides dict."""
        axes = [
            SweepAxis(name="cvml", values=[
                SweepAxisValue(label="off", overrides={"model.tlob_use_cvml": False}),
                SweepAxisValue(label="on", overrides={"model.tlob_use_cvml": True}),
            ]),
            SweepAxis(name="loss_delta", values=[
                SweepAxisValue(label="d12_6", overrides={"model.regression_loss_delta": 12.6}),
                SweepAxisValue(label="d15_1", overrides={"model.regression_loss_delta": 15.1}),
            ]),
        ]
        results = expand_sweep_with_axis_values(_build_manifest(axes, sweep_name="e5_phase2"))
        assert len(results) == 4
        for m, av in results:
            assert "model.tlob_use_cvml" in m.stages.training.overrides
            assert "model.regression_loss_delta" in m.stages.training.overrides


# -----------------------------------------------------------------------------
# Strategy registry shell (SHOULD-ADOPT 3)
# -----------------------------------------------------------------------------


class TestStrategyRegistryShell:
    def test_reserved_strategy_name_surfaces_informative_error(self):
        """Reserved strategy names → validation error with "RESERVED for a future phase"."""
        sweep = SweepConfig(name="t", strategy="zip", axes=[
            SweepAxis(name="a", values=[SweepAxisValue(label="a1", overrides={})]),
        ])
        errors = validate_sweep(sweep)
        assert any("RESERVED for a future phase" in e for e in errors)
        assert any("zip" in e for e in errors)

    def test_unknown_strategy_name_surfaces_valid_list(self):
        """Non-reserved + non-implemented strategy names → "must be one of [...]" error."""
        sweep = SweepConfig(name="t", strategy="random", axes=[
            SweepAxis(name="a", values=[SweepAxisValue(label="a1", overrides={})]),
        ])
        errors = validate_sweep(sweep)
        assert any("strategy must be one of" in e for e in errors)

    def test_grid_strategy_still_works(self):
        """Regression: grid (the only implemented strategy) continues working."""
        axes = [SweepAxis(name="a", values=[
            SweepAxisValue(label="a1", overrides={"model.X": 1}),
            SweepAxisValue(label="a2", overrides={"model.X": 2}),
        ])]
        results = expand_sweep_with_axis_values(_build_manifest(axes))
        assert len(results) == 2
