"""Phase 5 Preview Block 3: sweep expansion × fingerprint integration.

Locks that expanding a sweep manifest produces a clean N-record ledger entry:
the Cartesian product of axes yields N distinct fingerprints (no phantom
dedup misses), and identical overrides produce identical fingerprints
(determinism).

This sits on top of:
- Phase 4 Batch 4c.3 — feature_set ↔ feature_indices fingerprint equivalence
  (test_fingerprint_feature_set_mutation.py)
- Phase 5 Preview Block 1 — axis_values piped through from
  expand_sweep_with_axis_values (test_sweep_axis_values_preserved.py)

The concrete template exercised is ``experiments/sweeps/e5_phase2_sweep.yaml``
(2×2 CVML × Huber-delta).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from hft_ops.ledger.dedup import (
    _cached_resolve_feature_set_indices,
    compute_fingerprint,
)
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.sweep import expand_sweep, expand_sweep_with_axis_values
from hft_ops.paths import PipelinePaths


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SWEEP_PATH = (
    _REPO_ROOT / "hft-ops" / "experiments" / "sweeps" / "e5_phase2_sweep.yaml"
)

# Phase 5 FULL-A Block 5: MVP templates shipped in this phase.
_PHASE5_FULL_A_TEMPLATES = [
    # (template_name, expected_grid_count)
    ("loss_ablation.yaml", 6),
    ("horizon_sensitivity.yaml", 9),
    ("backtest_cost_sensitivity.yaml", 9),
]


@pytest.fixture(autouse=True)
def _clear_fp_caches():
    """Reset LRU cache + trainer-presets module cache between tests."""
    import hft_ops.ledger.dedup as _dedup

    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None
    yield
    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None


@pytest.fixture
def paths() -> PipelinePaths:
    return PipelinePaths(pipeline_root=_REPO_ROOT)


@pytest.fixture
def sweep_manifest():
    """Load the e5_phase2 sweep template (ships in the repo)."""
    assert _SWEEP_PATH.exists(), f"Sweep template missing: {_SWEEP_PATH}"
    return load_manifest(_SWEEP_PATH)


class TestSweepExpansionCardinality:
    """The 2×2 grid must expand to exactly 4 points, all uniquely named."""

    def test_grid_has_four_points(self, sweep_manifest):
        expanded = expand_sweep(sweep_manifest)
        assert len(expanded) == 4, (
            f"Expected 4 grid points (2 CVML × 2 loss_delta), got {len(expanded)}"
        )

    def test_grid_point_names_are_unique(self, sweep_manifest):
        expanded = expand_sweep(sweep_manifest)
        names = [m.experiment.name for m in expanded]
        assert len(set(names)) == len(names), (
            f"Duplicate grid-point names: {names}"
        )

    def test_grid_point_names_match_labels(self, sweep_manifest):
        """Name format: <sweep>__<axis1_label>_<axis2_label>."""
        expanded_with_axes = expand_sweep_with_axis_values(sweep_manifest)
        for manifest, axis_values in expanded_with_axes:
            name = manifest.experiment.name
            # Every axis label must appear in the name
            for label in axis_values.values():
                assert label in name, (
                    f"Grid point '{name}' does not contain axis label '{label}'. "
                    f"axis_values={axis_values}"
                )


class TestSweepAxisValuesPropagate:
    """axis_values dict must be complete for every grid point (B1 regression)."""

    def test_every_point_has_both_axis_keys(self, sweep_manifest):
        expanded_with_axes = expand_sweep_with_axis_values(sweep_manifest)
        for _, axis_values in expanded_with_axes:
            assert set(axis_values.keys()) == {"cvml", "loss_delta"}, (
                f"Grid point missing axis keys. Got: {axis_values}"
            )

    def test_combos_are_full_cartesian(self, sweep_manifest):
        expanded_with_axes = expand_sweep_with_axis_values(sweep_manifest)
        combos = {(av["cvml"], av["loss_delta"]) for _, av in expanded_with_axes}
        assert combos == {
            ("off", "d12_6"),
            ("off", "d15_1"),
            ("on", "d12_6"),
            ("on", "d15_1"),
        }, f"Combo set wrong: {combos}"


class TestSweepFingerprintUniqueness:
    """Four DIFFERENT grid points → four DIFFERENT fingerprints.

    If any two collide, dedup would silently conflate distinct experiments in
    the ledger (Phase 3 §3.3b / 4c.3 class of bug at the sweep level).
    """

    def test_four_distinct_fingerprints(self, sweep_manifest, paths):
        expanded = expand_sweep(sweep_manifest)
        fingerprints: Dict[str, str] = {}
        for manifest in expanded:
            fp = compute_fingerprint(manifest, paths)
            fingerprints[manifest.experiment.name] = fp

        unique_fps = set(fingerprints.values())
        assert len(unique_fps) == 4, (
            "Sweep expansion collapsed to < 4 fingerprints — dedup would "
            "silently conflate grid points. "
            f"fingerprints={fingerprints}"
        )

    def test_fingerprint_is_deterministic(self, sweep_manifest, paths):
        """Re-expanding the same sweep twice yields the same fingerprints."""
        run_a = expand_sweep(sweep_manifest)
        run_b = expand_sweep(sweep_manifest)

        fps_a = {m.experiment.name: compute_fingerprint(m, paths) for m in run_a}
        fps_b = {m.experiment.name: compute_fingerprint(m, paths) for m in run_b}

        assert fps_a == fps_b, (
            f"Fingerprint non-determinism across expansions. A={fps_a}, B={fps_b}"
        )


class TestSweepTrainerConfigOverrides:
    """Each grid point must carry the correct trainer-YAML overrides."""

    def test_cvml_override_reaches_training_stage(self, sweep_manifest):
        expanded_with_axes = expand_sweep_with_axis_values(sweep_manifest)
        for manifest, axis_values in expanded_with_axes:
            cvml_label = axis_values["cvml"]
            training_overrides = manifest.stages.training.overrides
            assert "model.tlob_use_cvml" in training_overrides, (
                f"Missing cvml override on grid point '{manifest.experiment.name}'"
            )
            expected = True if cvml_label == "on" else False
            assert training_overrides["model.tlob_use_cvml"] is expected, (
                f"cvml override wrong for {cvml_label}: "
                f"got {training_overrides['model.tlob_use_cvml']}"
            )

    def test_loss_delta_override_reaches_training_stage(self, sweep_manifest):
        expanded_with_axes = expand_sweep_with_axis_values(sweep_manifest)
        label_to_value = {"d12_6": 12.6, "d15_1": 15.1}
        for manifest, axis_values in expanded_with_axes:
            delta_label = axis_values["loss_delta"]
            training_overrides = manifest.stages.training.overrides
            assert "model.regression_loss_delta" in training_overrides, (
                f"Missing loss_delta override on '{manifest.experiment.name}'"
            )
            expected = label_to_value[delta_label]
            assert training_overrides["model.regression_loss_delta"] == expected, (
                f"loss_delta override wrong for {delta_label}: "
                f"got {training_overrides['model.regression_loss_delta']}"
            )


class TestPhase5FullATemplates:
    """Phase 5 FULL-A Block 5: each MVP template expands, all grid points
    fingerprint distinctly. Parametrized across the 3 shipped templates."""

    @pytest.mark.parametrize("template_name,expected_count", _PHASE5_FULL_A_TEMPLATES)
    def test_template_expansion_produces_expected_grid_count(
        self, template_name, expected_count, paths
    ):
        """Each shipped template's expand_sweep produces the expected
        Cartesian grid cardinality."""
        template_path = _REPO_ROOT / "hft-ops" / "experiments" / "sweeps" / template_name
        assert template_path.exists(), f"template missing: {template_path}"
        manifest = load_manifest(template_path)
        expanded = expand_sweep(manifest)
        assert len(expanded) == expected_count, (
            f"{template_name}: expected {expected_count} grid points, got {len(expanded)}"
        )

    @pytest.mark.parametrize("template_name,expected_count", _PHASE5_FULL_A_TEMPLATES)
    def test_template_fingerprints_all_distinct(
        self, template_name, expected_count, paths
    ):
        """Every grid point in every shipped template produces a distinct
        fingerprint (no silent ledger dedup across the sweep)."""
        template_path = _REPO_ROOT / "hft-ops" / "experiments" / "sweeps" / template_name
        manifest = load_manifest(template_path)
        expanded = expand_sweep(manifest)
        fps = [compute_fingerprint(m, paths) for m in expanded]
        assert len(set(fps)) == expected_count, (
            f"{template_name}: {len(set(fps))} unique fps for {expected_count} grid points — "
            f"possible dedup conflation."
        )

    @pytest.mark.parametrize("template_name,expected_count", _PHASE5_FULL_A_TEMPLATES)
    def test_template_axis_values_complete(self, template_name, expected_count, paths):
        """axis_values dict is populated for every grid point (B1 piping)."""
        template_path = _REPO_ROOT / "hft-ops" / "experiments" / "sweeps" / template_name
        manifest = load_manifest(template_path)
        expanded_with_axes = expand_sweep_with_axis_values(manifest)
        expected_axis_keys = {axis.name for axis in manifest.sweep.axes}
        for m, av in expanded_with_axes:
            assert set(av.keys()) == expected_axis_keys, (
                f"{template_name}: grid point '{m.experiment.name}' axis_values "
                f"{av} != expected axes {expected_axis_keys}"
            )
