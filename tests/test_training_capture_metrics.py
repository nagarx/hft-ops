"""Tests for TrainingRunner._capture_training_metrics.

Phase 7 Stage 7.4 Round 4 item #1 (C1-complete). Round 1 extended this
function to read ``test_metrics.json`` + a 3-key subset of
``training_history.json`` per-epoch metrics. Round 4 closes the
regression-key gap: the iteration now covers the FULL max-better and
min-better regression val_* taxonomy (5 + 3 from post_training_gate)
plus ``val_signal_rate`` (TLOB / opportunity classification strategies).

Why this matters: Round 1's Phase-7.4 ``_find_prior_best_experiment``
falls back to ``best_val_ic`` when ``test_ic`` is absent. Before
Round 4, ``_capture_training_metrics`` never emitted ``best_val_ic``
(or the other regression val_* keys), so the fallback returned None
and prior-best comparison was silently skipped for every PyTorch
regression run.

These tests lock the three code paths:

1. List-of-dicts ``training_history.json`` (current MetricLogger
   callback format per ``lobtrainer.training.callbacks:552``).
2. Dict-of-lists ``training_history.json`` (legacy retroactive
   record format preserved for backward compat).
3. ``test_metrics.json`` flat-dict format (Round 4 item #6 convention).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hft_ops.stages.base import StageResult, StageStatus
from hft_ops.stages.training import (
    TrainingRunner,
    _CLASSIFICATION_VAL_MAX_KEYS,
    _REGRESSION_VAL_MAX_KEYS,
    _REGRESSION_VAL_MIN_KEYS,
)


@pytest.fixture
def runner():
    return TrainingRunner()


def _make_stage_result(output_dir: Path) -> StageResult:
    return StageResult(
        stage_name="training",
        status=StageStatus.COMPLETED,
        output_dir=output_dir,
    )


class TestListOfDictsHistory:
    """List-of-dicts per-epoch format (current MetricLogger output)."""

    def test_extracts_all_regression_max_keys(self, tmp_path, runner):
        # Three epochs of a regression run — best values at different epochs.
        history = [
            {"epoch": 0, "val_ic": 0.30, "val_directional_accuracy": 0.58,
             "val_r2": 0.08, "val_pearson": 0.33, "val_profitable_accuracy": 0.51},
            {"epoch": 1, "val_ic": 0.38, "val_directional_accuracy": 0.64,
             "val_r2": 0.12, "val_pearson": 0.40, "val_profitable_accuracy": 0.54},
            {"epoch": 2, "val_ic": 0.35, "val_directional_accuracy": 0.62,
             "val_r2": 0.10, "val_pearson": 0.37, "val_profitable_accuracy": 0.52},
        ]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["best_val_ic"] == pytest.approx(0.38)
        assert cm["best_val_directional_accuracy"] == pytest.approx(0.64)
        assert cm["best_val_r2"] == pytest.approx(0.12)
        assert cm["best_val_pearson"] == pytest.approx(0.40)
        assert cm["best_val_profitable_accuracy"] == pytest.approx(0.54)

    def test_extracts_all_regression_min_keys(self, tmp_path, runner):
        history = [
            {"epoch": 0, "val_loss": 1.2, "val_mae": 4.5, "val_rmse": 6.1},
            {"epoch": 1, "val_loss": 0.85, "val_mae": 2.3, "val_rmse": 3.2},
            {"epoch": 2, "val_loss": 0.90, "val_mae": 2.8, "val_rmse": 3.5},
        ]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["best_val_loss"] == pytest.approx(0.85)
        assert cm["best_val_mae"] == pytest.approx(2.3)
        assert cm["best_val_rmse"] == pytest.approx(3.2)

    def test_extracts_classification_max_keys(self, tmp_path, runner):
        history = [
            {"epoch": 0, "val_accuracy": 0.50, "val_macro_f1": 0.45,
             "val_signal_rate": 0.60},
            {"epoch": 1, "val_accuracy": 0.55, "val_macro_f1": 0.52,
             "val_signal_rate": 0.58},
        ]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["best_val_accuracy"] == pytest.approx(0.55)
        assert cm["best_val_macro_f1"] == pytest.approx(0.52)
        assert cm["best_val_signal_rate"] == pytest.approx(0.60)

    def test_filters_nan_and_inf(self, tmp_path, runner):
        # NaN-poisoned early epochs (degenerate IC on constant predictions)
        # must not appear in the max reduction.
        history = [
            {"epoch": 0, "val_ic": float("nan")},
            {"epoch": 1, "val_ic": float("inf")},
            {"epoch": 2, "val_ic": 0.38},
            {"epoch": 3, "val_ic": float("-inf")},
        ]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        assert result.captured_metrics["best_val_ic"] == pytest.approx(0.38)

    def test_ignores_missing_keys(self, tmp_path, runner):
        # Regression-only run: val_accuracy never populated.
        history = [{"epoch": 0, "val_ic": 0.38}]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert "best_val_ic" in cm
        assert "best_val_accuracy" not in cm
        assert "best_val_macro_f1" not in cm

    def test_skips_non_dict_epoch_entries(self, tmp_path, runner):
        # Defensive: a malformed history with non-dict elements should not
        # raise — just extract what's valid.
        history: list[Any] = [
            "not-a-dict",
            {"epoch": 0, "val_ic": 0.30},
            None,
        ]
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        assert result.captured_metrics["best_val_ic"] == pytest.approx(0.30)


class TestDictOfListsHistory:
    """Legacy dict-of-lists format (retroactive records)."""

    def test_extracts_regression_keys(self, tmp_path, runner):
        history = {
            "val_ic": [0.30, 0.38, 0.35],
            "val_loss": [1.2, 0.85, 0.90],
            "val_r2": [0.08, 0.12, 0.10],
        }
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["best_val_ic"] == pytest.approx(0.38)
        assert cm["best_val_r2"] == pytest.approx(0.12)
        assert cm["best_val_loss"] == pytest.approx(0.85)

    def test_filters_nan_in_legacy_format(self, tmp_path, runner):
        history = {"val_ic": [float("nan"), 0.38, float("inf")]}
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        assert result.captured_metrics["best_val_ic"] == pytest.approx(0.38)


class TestTestMetricsJson:
    """Flat ``test_metrics.json`` written by Round 4 item #6."""

    def test_merges_finite_scalars(self, tmp_path, runner):
        payload = {
            "test_ic": 0.38,
            "test_directional_accuracy": 0.64,
            "test_r2": 0.12,
            "test_mae": 5.67,
            "test_rmse": 6.89,
            "test_pearson": 0.40,
            "test_profitable_accuracy": 0.54,
        }
        (tmp_path / "test_metrics.json").write_text(json.dumps(payload))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        for key, value in payload.items():
            assert result.captured_metrics[key] == pytest.approx(value)

    def test_filters_nan_and_inf(self, tmp_path, runner):
        payload = {
            "test_ic": 0.38,
            "test_r2": float("nan"),
            "test_mae": float("inf"),
        }
        (tmp_path / "test_metrics.json").write_text(json.dumps(payload))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["test_ic"] == pytest.approx(0.38)
        assert "test_r2" not in cm
        assert "test_mae" not in cm


class TestHistoryAndTestMerge:
    """Both files present — test-split wins on key collision."""

    def test_both_files_merged(self, tmp_path, runner):
        history = [{"epoch": 0, "val_ic": 0.35}]
        test_metrics = {"test_ic": 0.38}
        (tmp_path / "training_history.json").write_text(json.dumps(history))
        (tmp_path / "test_metrics.json").write_text(json.dumps(test_metrics))
        result = _make_stage_result(tmp_path)

        runner._capture_training_metrics(result)

        cm = result.captured_metrics
        assert cm["best_val_ic"] == pytest.approx(0.35)
        assert cm["test_ic"] == pytest.approx(0.38)

    def test_malformed_json_does_not_raise(self, tmp_path, runner):
        (tmp_path / "training_history.json").write_text("{not valid json")
        (tmp_path / "test_metrics.json").write_text("also not json")
        result = _make_stage_result(tmp_path)

        # Should gracefully no-op, not raise.
        runner._capture_training_metrics(result)

        assert result.captured_metrics == {}

    def test_no_output_dir_is_no_op(self, runner):
        result = StageResult(
            stage_name="training",
            status=StageStatus.COMPLETED,
            output_dir=None,
        )
        runner._capture_training_metrics(result)
        assert result.captured_metrics == {}


class TestConstantTaxonomy:
    """Lock the key taxonomy to prevent silent drift."""

    def test_regression_max_keys_match_gate_ssot(self):
        # If the gate's fallback order changes, this test forces a
        # coordinated update here (documented via the import).
        assert _REGRESSION_VAL_MAX_KEYS == (
            "val_ic",
            "val_directional_accuracy",
            "val_r2",
            "val_pearson",
            "val_profitable_accuracy",
        )

    def test_regression_min_keys_match_gate_ssot(self):
        assert _REGRESSION_VAL_MIN_KEYS == (
            "val_loss",
            "val_mae",
            "val_rmse",
        )

    def test_classification_max_keys(self):
        assert _CLASSIFICATION_VAL_MAX_KEYS == (
            "val_accuracy",
            "val_macro_f1",
            "val_signal_rate",
        )

    def test_no_overlap_between_regression_max_and_min(self):
        # Ensures max() vs min() reduction never ambiguously applied.
        assert set(_REGRESSION_VAL_MAX_KEYS).isdisjoint(
            set(_REGRESSION_VAL_MIN_KEYS)
        )
