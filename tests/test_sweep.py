"""Tests for sweep/grid expansion."""

from __future__ import annotations

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
from hft_ops.manifest.sweep import expand_sweep, validate_sweep


# =============================================================================
# Fixtures
# =============================================================================


def _make_sweep(
    name: str = "test_sweep",
    axes: list | None = None,
    strategy: str = "grid",
) -> SweepConfig:
    if axes is None:
        axes = [
            SweepAxis(
                name="model",
                values=[
                    SweepAxisValue(label="tlob", overrides={"model.model_type": "tlob"}),
                    SweepAxisValue(label="ridge", overrides={"model.model_type": "temporal_ridge"}),
                ],
            ),
            SweepAxis(
                name="horizon",
                values=[
                    SweepAxisValue(label="H10", overrides={"horizon_value": 10}),
                    SweepAxisValue(label="H60", overrides={"horizon_value": 60}),
                ],
            ),
        ]
    return SweepConfig(name=name, strategy=strategy, axes=axes)


def _make_manifest(sweep: SweepConfig | None = None) -> ExperimentManifest:
    return ExperimentManifest(
        experiment=ExperimentHeader(name="base_exp", tags=["test"]),
        stages=Stages(
            training=TrainingStage(
                config="configs/test.yaml",
                overrides={"data.data_dir": "/data/test"},
            )
        ),
        sweep=sweep,
    )


# =============================================================================
# validate_sweep Tests
# =============================================================================


class TestValidateSweep:
    def test_valid_sweep(self):
        errors = validate_sweep(_make_sweep())
        assert errors == []

    def test_missing_name(self):
        errors = validate_sweep(_make_sweep(name=""))
        assert any("name is required" in e for e in errors)

    def test_invalid_strategy(self):
        errors = validate_sweep(_make_sweep(strategy="bayesian"))
        assert any("strategy must be 'grid'" in e for e in errors)

    def test_no_axes(self):
        errors = validate_sweep(SweepConfig(name="x", axes=[]))
        assert any("at least one axis" in e for e in errors)

    def test_duplicate_label(self):
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="dup", overrides={"x": 1}),
                SweepAxisValue(label="dup", overrides={"x": 2}),
            ]),
        ]
        errors = validate_sweep(_make_sweep(axes=axes))
        assert any("duplicate label 'dup'" in e for e in errors)

    def test_invalid_label_format(self):
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="has space", overrides={"x": 1}),
            ]),
        ]
        errors = validate_sweep(_make_sweep(axes=axes))
        assert any("[a-zA-Z0-9_-]+" in e for e in errors)

    def test_cross_axis_conflict(self):
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="v1", overrides={"model.type": "tlob"}),
            ]),
            SweepAxis(name="b", values=[
                SweepAxisValue(label="v2", overrides={"model.type": "ridge"}),
            ]),
        ]
        errors = validate_sweep(_make_sweep(axes=axes))
        assert any("conflict" in e.lower() for e in errors)

    def test_no_conflict_disjoint_keys(self):
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="v1", overrides={"model.type": "tlob"}),
            ]),
            SweepAxis(name="b", values=[
                SweepAxisValue(label="v2", overrides={"data.dir": "/x"}),
            ]),
        ]
        errors = validate_sweep(_make_sweep(axes=axes))
        assert errors == []


# =============================================================================
# expand_sweep Tests
# =============================================================================


class TestExpandSweep:
    def test_cartesian_product_count(self):
        """2 models x 2 horizons = 4 experiments."""
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        assert len(results) == 4

    def test_experiment_names(self):
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        names = [r.experiment.name for r in results]
        assert "test_sweep__tlob_H10" in names
        assert "test_sweep__tlob_H60" in names
        assert "test_sweep__ridge_H10" in names
        assert "test_sweep__ridge_H60" in names

    def test_sweep_cleared_on_concrete(self):
        """Concrete manifests should have sweep=None."""
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        for r in results:
            assert r.sweep is None

    def test_base_overrides_preserved(self):
        """Base training overrides should be in every concrete manifest."""
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        for r in results:
            assert r.stages.training.overrides["data.data_dir"] == "/data/test"

    def test_axis_overrides_merged(self):
        """Axis overrides should be merged into training overrides."""
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        tlob_h10 = [r for r in results if r.experiment.name == "test_sweep__tlob_H10"][0]
        assert tlob_h10.stages.training.overrides["model.model_type"] == "tlob"

    def test_manifest_level_override(self):
        """horizon_value should be set on TrainingStage, not in overrides dict."""
        manifest = _make_manifest(sweep=_make_sweep())
        results = expand_sweep(manifest)
        h10_results = [r for r in results if "H10" in r.experiment.name]
        for r in h10_results:
            assert r.stages.training.horizon_value == 10
            assert "horizon_value" not in r.stages.training.overrides

    def test_no_mutation_of_original(self):
        """expand_sweep should not mutate the original manifest."""
        sweep = _make_sweep()
        manifest = _make_manifest(sweep=sweep)
        original_name = manifest.experiment.name
        expand_sweep(manifest)
        assert manifest.experiment.name == original_name
        assert manifest.sweep is not None  # Original still has sweep

    def test_single_axis(self):
        """Single axis = no Cartesian product, just N experiments."""
        axes = [
            SweepAxis(name="lr", values=[
                SweepAxisValue(label="low", overrides={"train.lr": 0.001}),
                SweepAxisValue(label="high", overrides={"train.lr": 0.01}),
            ]),
        ]
        manifest = _make_manifest(sweep=_make_sweep(axes=axes))
        results = expand_sweep(manifest)
        assert len(results) == 2

    def test_three_axes(self):
        """3 axes: 2x2x3 = 12 experiments."""
        axes = [
            SweepAxis(name="a", values=[
                SweepAxisValue(label="a1", overrides={"x": 1}),
                SweepAxisValue(label="a2", overrides={"x": 2}),
            ]),
            SweepAxis(name="b", values=[
                SweepAxisValue(label="b1", overrides={"y": 1}),
                SweepAxisValue(label="b2", overrides={"y": 2}),
            ]),
            SweepAxis(name="c", values=[
                SweepAxisValue(label="c1", overrides={"z": 1}),
                SweepAxisValue(label="c2", overrides={"z": 2}),
                SweepAxisValue(label="c3", overrides={"z": 3}),
            ]),
        ]
        manifest = _make_manifest(sweep=_make_sweep(axes=axes))
        results = expand_sweep(manifest)
        assert len(results) == 12

    def test_raises_without_sweep(self):
        """expand_sweep should raise if manifest has no sweep."""
        manifest = _make_manifest(sweep=None)
        with pytest.raises(ValueError, match="no sweep"):
            expand_sweep(manifest)

    def test_raises_on_invalid_sweep(self):
        """expand_sweep should raise if sweep validation fails."""
        bad_sweep = SweepConfig(name="", axes=[])
        manifest = _make_manifest(sweep=bad_sweep)
        with pytest.raises(ValueError, match="Invalid sweep"):
            expand_sweep(manifest)
