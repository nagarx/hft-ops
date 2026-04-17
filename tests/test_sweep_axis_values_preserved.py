"""Phase 5 Preview B1 fix: axis_values MUST be piped through from
`expand_sweep_with_axis_values`, not lossily re-derived from applied overrides.

Regression guard for the prior `cli.py::sweep_run` heuristic (L1107-1118
pre-2026-04-16) that walked axis.values[*].overrides and matched against
exp.stages.training.overrides. Under multi-axis overlap (two axes setting
different keys whose labels mapped to the same axis name, or any case where
`break`-on-first-match discarded later matches), the recorded `axis_values`
was wrong.

The new contract: axis_values comes DIRECTLY from the axis name + selected
label in the Cartesian product, computed BEFORE any override merging.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    Stages,
    TrainingStage,
    SweepAxis,
    SweepAxisValue,
    SweepConfig,
)
from hft_ops.manifest.sweep import (
    expand_sweep,
    expand_sweep_with_axis_values,
    get_axis_values_for_point,
)


def _build_manifest(axes: List[SweepAxis], sweep_name: str = "t") -> ExperimentManifest:
    """Minimal manifest builder for sweep tests."""
    return ExperimentManifest(
        experiment=ExperimentHeader(name="t", contract_version="2.2"),
        stages=Stages(
            training=TrainingStage(
                enabled=True,
                config="configs/base.yaml",
                output_dir="out/${sweep.point_name}",
            ),
        ),
        sweep=SweepConfig(
            name=sweep_name,
            strategy="grid",
            axes=axes,
        ),
    )


class TestAxisValuesPiped:
    def test_single_axis_labels_match_selected_values(self):
        manifest = _build_manifest([
            SweepAxis(
                name="loss",
                values=[
                    SweepAxisValue(label="huber", overrides={"train.loss_type": "huber"}),
                    SweepAxisValue(label="mse", overrides={"train.loss_type": "mse"}),
                ],
            ),
        ])
        result = expand_sweep_with_axis_values(manifest)
        assert len(result) == 2

        _, av0 = result[0]
        _, av1 = result[1]
        assert av0 == {"loss": "huber"}
        assert av1 == {"loss": "mse"}

    def test_two_axis_cartesian_labels_complete(self):
        manifest = _build_manifest([
            SweepAxis(
                name="cvml",
                values=[
                    SweepAxisValue(label="off", overrides={"model.tlob_use_cvml": False}),
                    SweepAxisValue(label="on", overrides={"model.tlob_use_cvml": True}),
                ],
            ),
            SweepAxis(
                name="loss_delta",
                values=[
                    SweepAxisValue(label="d12_6", overrides={"model.regression_loss_delta": 12.6}),
                    SweepAxisValue(label="d15_1", overrides={"model.regression_loss_delta": 15.1}),
                ],
            ),
        ])
        result = expand_sweep_with_axis_values(manifest)
        assert len(result) == 4

        # Every grid point MUST have BOTH axis labels — the heuristic bug
        # would previously drop one of them under certain override patterns.
        for manifest_i, axis_values in result:
            assert set(axis_values.keys()) == {"cvml", "loss_delta"}, (
                f"Grid point '{manifest_i.experiment.name}' missing axis labels. "
                f"Got: {axis_values}"
            )
            assert axis_values["cvml"] in {"off", "on"}
            assert axis_values["loss_delta"] in {"d12_6", "d15_1"}

        # And the set of combinations MUST be the full Cartesian product
        combos = {(av["cvml"], av["loss_delta"]) for _, av in result}
        assert combos == {
            ("off", "d12_6"), ("off", "d15_1"), ("on", "d12_6"), ("on", "d15_1"),
        }

    def test_overlapping_override_keys_labels_still_correct(self):
        """Regression: the prior heuristic broke when two axes both
        overrode `model.*` keys. Ground-truth axis_values are independent
        of override-key overlap."""
        manifest = _build_manifest([
            SweepAxis(
                name="axis_a",
                values=[
                    SweepAxisValue(label="a1", overrides={"model.hidden_dim": 32}),
                    SweepAxisValue(label="a2", overrides={"model.hidden_dim": 64}),
                ],
            ),
            SweepAxis(
                name="axis_b",
                values=[
                    # Different override key, same axis path prefix
                    SweepAxisValue(label="b1", overrides={"model.dropout": 0.1}),
                    SweepAxisValue(label="b2", overrides={"model.dropout": 0.2}),
                ],
            ),
        ])
        result = expand_sweep_with_axis_values(manifest)
        assert len(result) == 4

        # Under the lossy heuristic, axis_b label could have been
        # overwritten by axis_a's first match. Ground-truth pipeline means
        # ALL 4 combos are present with correct labels.
        combos = {(av["axis_a"], av["axis_b"]) for _, av in result}
        assert combos == {("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")}


class TestBackwardCompat:
    def test_expand_sweep_returns_manifests_only(self):
        """Legacy `expand_sweep()` still works — returns List[ExperimentManifest]."""
        manifest = _build_manifest([
            SweepAxis(
                name="loss",
                values=[
                    SweepAxisValue(label="huber", overrides={"train.loss_type": "huber"}),
                    SweepAxisValue(label="mse", overrides={"train.loss_type": "mse"}),
                ],
            ),
        ])
        experiments = expand_sweep(manifest)
        assert len(experiments) == 2
        # Each element is a bare ExperimentManifest (not a tuple)
        for exp in experiments:
            assert isinstance(exp, ExperimentManifest)
            assert exp.sweep is None  # sweep cleared on concrete manifests

    def test_get_axis_values_for_point_helper_consistent(self):
        """The `get_axis_values_for_point` helper (used internally by
        `expand_sweep_with_axis_values`) must produce the same dict shape."""
        axes = [
            SweepAxis(name="x", values=[SweepAxisValue(label="x1", overrides={}), SweepAxisValue(label="x2", overrides={})]),
            SweepAxis(name="y", values=[SweepAxisValue(label="y1", overrides={})]),
        ]
        point = (axes[0].values[0], axes[1].values[0])
        av = get_axis_values_for_point(axes, point)
        assert av == {"x": "x1", "y": "y1"}
