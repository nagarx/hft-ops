"""Tests for Phase 7 Stage 7.4 post-training regression-detection gate.

Organized into three layers:

1. Individual check functions (_check_floor, _check_prior_best_ratio,
   _check_cost_breakeven) — pure-function tests, no filesystem.
2. Helpers (_read_training_metrics, _select_primary_metric,
   _find_prior_best_experiment) — filesystem + ledger interactions.
3. PostTrainingGateRunner end-to-end — manifest + config + disk.

Design convention: every test is explicit about the scenario it's
covering (pass / warn / fail / skipped) to keep regression signals
interpretable in CI output.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hft_ops.config import OpsConfig
from hft_ops.paths import PipelinePaths
from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    PostTrainingGateStage,
    Stages,
    TrainingStage,
)


def _make_manifest(
    *,
    name: str = "test_exp",
    training_output_dir: str = "training_out",
    trainer_config: dict = None,
    training_horizon: int = 10,
    post_training_gate: PostTrainingGateStage = None,
) -> ExperimentManifest:
    """Build a minimal ExperimentManifest for testing."""
    return ExperimentManifest(
        experiment=ExperimentHeader(name=name),
        stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=training_output_dir,
                trainer_config=trainer_config or {
                    "model": {"model_type": "tlob"},
                    "data": {"labeling_strategy": "regression"},
                },
                horizon_value=training_horizon,
            ),
            post_training_gate=(
                post_training_gate or PostTrainingGateStage(enabled=True)
            ),
        ),
    )
from hft_ops.stages.base import StageStatus
from hft_ops.stages.post_training_gate import (
    CheckResult,
    GateReport,
    PostTrainingGateRunner,
    _PRIMARY_METRIC_FALLBACK_ORDER,
    _build_match_signature,
    _check_cost_breakeven,
    _check_floor,
    _check_prior_best_ratio,
    _find_prior_best_experiment,
    _read_training_metrics,
    _select_primary_metric,
)


# =============================================================================
# Layer 1: individual check functions
# =============================================================================


class TestCheckFloor:
    def test_pass_when_above_floor(self):
        r = _check_floor("test_ic", 0.38, floor=0.05)
        assert r.status == "pass"
        assert r.metric_value == 0.38

    def test_fail_when_below_floor(self):
        r = _check_floor("test_ic", 0.02, floor=0.05)
        assert r.status == "fail"
        assert "below floor" in r.message

    def test_pass_when_equal_to_floor(self):
        r = _check_floor("test_ic", 0.05, floor=0.05)
        assert r.status == "pass"

    def test_skipped_when_metric_absent(self):
        r = _check_floor("test_ic", None, floor=0.05)
        assert r.status == "skipped"


class TestCheckPriorBestRatio:
    def test_pass_when_above_ratio(self):
        r = _check_prior_best_ratio(
            "test_ic", current_value=0.36, prior_best_value=0.38,
            min_ratio=0.9, n_matching=3,
        )
        assert r.status == "pass"
        # 0.36 / 0.38 = 0.947 >= 0.9 → pass

    def test_fail_regression(self):
        r = _check_prior_best_ratio(
            "test_ic", current_value=0.30, prior_best_value=0.38,
            min_ratio=0.9, n_matching=3,
        )
        assert r.status == "fail"
        assert "REGRESSION" in r.message
        # 0.30 / 0.38 = 0.789 < 0.9 → fail

    def test_skipped_when_no_prior(self):
        r = _check_prior_best_ratio(
            "test_ic", current_value=0.36, prior_best_value=None,
            min_ratio=0.9, n_matching=0,
        )
        assert r.status == "skipped"
        assert "no prior experiments" in r.message

    def test_skipped_when_ratio_zero(self):
        r = _check_prior_best_ratio(
            "test_ic", current_value=0.36, prior_best_value=0.38,
            min_ratio=0.0, n_matching=3,
        )
        assert r.status == "skipped"
        assert "disabled" in r.message

    def test_pass_when_prior_zero_or_negative(self):
        """Ratio-vs-zero is ill-defined; must not crash/fail."""
        r = _check_prior_best_ratio(
            "test_ic", current_value=0.36, prior_best_value=0.0,
            min_ratio=0.9, n_matching=3,
        )
        assert r.status == "pass"
        assert "vacuously passes" in r.message


class TestCheckCostBreakeven:
    def test_skipped_when_disabled(self):
        r = _check_cost_breakeven(metrics={"test_rmse": 5.0}, cost_breakeven_bps=0)
        assert r.status == "skipped"

    def test_pass_when_magnitude_above_floor(self):
        r = _check_cost_breakeven(
            metrics={"test_rmse": 3.5}, cost_breakeven_bps=1.4
        )
        assert r.status == "pass"

    def test_warn_when_below_floor(self):
        r = _check_cost_breakeven(
            metrics={"test_rmse": 0.5}, cost_breakeven_bps=1.4
        )
        # Informational only — must WARN (never FAIL) so no abort.
        assert r.status == "warn"
        assert "INFORMATIONAL" in r.message

    def test_uses_mae_when_rmse_missing(self):
        r = _check_cost_breakeven(
            metrics={"test_mae": 2.0}, cost_breakeven_bps=1.4
        )
        assert r.status == "pass"

    def test_skipped_when_no_magnitude(self):
        """Classification runs have no test_rmse/test_mae."""
        r = _check_cost_breakeven(
            metrics={"best_val_accuracy": 0.55}, cost_breakeven_bps=1.4
        )
        assert r.status == "skipped"


# =============================================================================
# Layer 2: helpers
# =============================================================================


class TestReadTrainingMetrics:
    def test_reads_test_metrics_json(self, tmp_path: Path):
        (tmp_path / "test_metrics.json").write_text(json.dumps({
            "test_ic": 0.38,
            "test_r2": 0.12,
            "test_directional_accuracy": 0.64,
        }))
        metrics = _read_training_metrics(tmp_path)
        assert metrics["test_ic"] == 0.38
        assert metrics["test_r2"] == 0.12
        assert metrics["test_directional_accuracy"] == 0.64

    def test_drops_non_finite(self, tmp_path: Path):
        (tmp_path / "test_metrics.json").write_text(
            '{"test_ic": 0.38, "test_nan": NaN, "test_inf": Infinity}'
        )
        metrics = _read_training_metrics(tmp_path)
        assert metrics["test_ic"] == 0.38
        assert "test_nan" not in metrics
        assert "test_inf" not in metrics

    def test_reads_list_of_dicts_history(self, tmp_path: Path):
        """Per-epoch history: extract best (max) val_* values."""
        (tmp_path / "training_history.json").write_text(json.dumps([
            {"epoch": 0, "val_accuracy": 0.50, "val_macro_f1": 0.48},
            {"epoch": 1, "val_accuracy": 0.55, "val_macro_f1": 0.52},
            {"epoch": 2, "val_accuracy": 0.53, "val_macro_f1": 0.50},
        ]))
        metrics = _read_training_metrics(tmp_path)
        assert metrics["best_val_accuracy"] == 0.55
        assert metrics["best_val_macro_f1"] == 0.52

    def test_merges_history_and_test_metrics(self, tmp_path: Path):
        (tmp_path / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.38})
        )
        (tmp_path / "training_history.json").write_text(json.dumps([
            {"epoch": 0, "val_accuracy": 0.55},
        ]))
        metrics = _read_training_metrics(tmp_path)
        assert metrics["test_ic"] == 0.38
        assert metrics["best_val_accuracy"] == 0.55

    def test_missing_files_no_op(self, tmp_path: Path):
        metrics = _read_training_metrics(tmp_path)
        assert metrics == {}

    def test_malformed_json_skipped(self, tmp_path: Path):
        (tmp_path / "test_metrics.json").write_text("not valid json")
        metrics = _read_training_metrics(tmp_path)
        assert metrics == {}


class TestSelectPrimaryMetric:
    def test_explicit_config_name(self):
        metrics = {"test_ic": 0.38, "best_val_macro_f1": 0.52}
        name, value = _select_primary_metric(metrics, "test_ic")
        assert name == "test_ic"
        assert value == 0.38

    def test_explicit_config_missing_returns_none(self):
        """Configured but not captured → name preserved, value=None."""
        metrics = {"other": 0.50}
        name, value = _select_primary_metric(metrics, "test_ic")
        assert name == "test_ic"
        assert value is None

    def test_fallback_order(self):
        # Regression metrics present → test_ic wins
        metrics = {
            "test_ic": 0.38,
            "test_directional_accuracy": 0.64,
            "best_val_macro_f1": 0.52,
        }
        name, value = _select_primary_metric(metrics, "")
        assert name == "test_ic"
        assert value == 0.38

    def test_fallback_classification(self):
        metrics = {"best_val_macro_f1": 0.52, "best_val_accuracy": 0.55}
        name, value = _select_primary_metric(metrics, "")
        assert name == "best_val_macro_f1"  # higher priority than accuracy

    def test_fallback_empty_returns_none(self):
        name, value = _select_primary_metric({}, "")
        assert name == ""
        assert value is None


class TestBuildMatchSignature:
    def test_inline_trainer_config(self):
        manifest = MagicMock()
        manifest.stages.training.trainer_config = {
            "model": {"model_type": "tlob"},
            "data": {"labeling_strategy": "regression"},
        }
        manifest.stages.training.horizon_value = 10
        sig = _build_match_signature(
            manifest,
            match_fields=("model_type", "labeling_strategy", "horizon_value"),
        )
        assert sig == {
            "model_type": "tlob",
            "labeling_strategy": "regression",
            "horizon_value": 10,
        }

    def test_missing_trainer_config_empty_strings(self):
        manifest = MagicMock()
        manifest.stages.training.trainer_config = None
        manifest.stages.training.horizon_value = 0
        sig = _build_match_signature(
            manifest,
            match_fields=("model_type", "labeling_strategy", "horizon_value"),
        )
        assert sig["model_type"] == ""
        assert sig["labeling_strategy"] == ""
        assert sig["horizon_value"] == 0


class TestFindPriorBestExperiment:
    def test_empty_ledger_returns_empty(self, tmp_path: Path):
        result_id, result_val, n = _find_prior_best_experiment(
            ledger_dir=tmp_path,
            match_signature={"model_type": "tlob"},
            metric_name="test_ic",
            exclude_experiment_name="current_exp",
        )
        assert result_id == ""
        assert result_val is None
        assert n == 0

    def test_finds_best_among_matching(self, tmp_path: Path):
        """With 3 matching records, returns the one with highest test_ic."""
        records_dir = tmp_path / "records"
        records_dir.mkdir()

        # Write 3 mock records via the ExperimentRecord API
        from hft_contracts.experiment_record import ExperimentRecord

        for name, ic in [("exp_a", 0.30), ("exp_b", 0.40), ("exp_c", 0.35)]:
            rec = ExperimentRecord(
                experiment_id=f"{name}_20260101_abc12345",
                name=name,
                fingerprint="a" * 64,
                training_metrics={"test_ic": ic},
                training_config={
                    "model": {"model_type": "tlob"},
                    "data": {"labeling_strategy": "regression"},
                },
                record_type="training",
                contract_version="2.2",
            )
            rec.save(records_dir / f"{rec.experiment_id}.json")

        # Force ledger rebuild
        from hft_ops.ledger import ExperimentLedger
        ledger = ExperimentLedger(tmp_path)
        ledger._rebuild_index()

        result_id, result_val, n = _find_prior_best_experiment(
            ledger_dir=tmp_path,
            match_signature={
                "model_type": "tlob",
                "labeling_strategy": "regression",
            },
            metric_name="test_ic",
            exclude_experiment_name="different_current",
        )
        assert n == 3
        assert result_val == 0.40
        assert "exp_b" in result_id

    def test_excludes_current_experiment(self, tmp_path: Path):
        records_dir = tmp_path / "records"
        records_dir.mkdir()
        from hft_contracts.experiment_record import ExperimentRecord

        rec = ExperimentRecord(
            experiment_id="cur_20260101_abc12345",
            name="current_exp",
            fingerprint="a" * 64,
            training_metrics={"test_ic": 0.50},
            training_config={"model": {"model_type": "tlob"}},
            record_type="training",
            contract_version="2.2",
        )
        rec.save(records_dir / f"{rec.experiment_id}.json")

        from hft_ops.ledger import ExperimentLedger
        ExperimentLedger(tmp_path)._rebuild_index()

        result_id, result_val, n = _find_prior_best_experiment(
            ledger_dir=tmp_path,
            match_signature={"model_type": "tlob"},
            metric_name="test_ic",
            exclude_experiment_name="current_exp",
        )
        # current_exp is excluded by name → no matches
        assert n == 0
        assert result_val is None

    def test_degenerate_signature_returns_empty(self, tmp_path: Path):
        """Phase 7 post-validation regression guard: when match_signature has
        no meaningful fields (all empty/zero), the query must SKIP the ledger
        scan and return empty — not silently match every historical experiment.

        Pre-fix (Phase 7 Stage 7.4 initial): degenerate signature would pass
        every entry through the field-matching loop (via the
        ``expected_value == ""`` continue), producing a false-positive prior-
        best that could flag regressions against wildly different experiment
        types (e.g., a classification run vs. the best-ever regression IC).
        """
        # Populate the ledger with a non-matching-type record to prove the
        # guard isn't accidentally letting it through.
        records_dir = tmp_path / "records"
        records_dir.mkdir()
        from hft_contracts.experiment_record import ExperimentRecord

        rec = ExperimentRecord(
            experiment_id="legitimate_20260101_abc12345",
            name="legitimate_run",
            fingerprint="a" * 64,
            training_metrics={"test_ic": 0.99},  # unrealistically high
            training_config={"model": {"model_type": "tlob"}},
            record_type="training",
            contract_version="2.2",
        )
        rec.save(records_dir / f"{rec.experiment_id}.json")

        from hft_ops.ledger import ExperimentLedger
        ExperimentLedger(tmp_path)._rebuild_index()

        # Degenerate signature — every field empty/zero
        result_id, result_val, n = _find_prior_best_experiment(
            ledger_dir=tmp_path,
            match_signature={
                "model_type": "",
                "labeling_strategy": "",
                "horizon_value": 0,
            },
            metric_name="test_ic",
            exclude_experiment_name="current_exp",
        )
        # Must return empty — NOT the 0.99 IC record that would pass
        # the vacuous-match.
        assert result_id == ""
        assert result_val is None
        assert n == 0

    def test_signature_mismatch_excludes(self, tmp_path: Path):
        records_dir = tmp_path / "records"
        records_dir.mkdir()
        from hft_contracts.experiment_record import ExperimentRecord

        # One matching, one non-matching
        rec1 = ExperimentRecord(
            experiment_id="match_a_20260101_abc12345",
            name="match_a",
            fingerprint="a" * 64,
            training_metrics={"test_ic": 0.30},
            training_config={"model": {"model_type": "tlob"}},
            record_type="training",
            contract_version="2.2",
        )
        rec1.save(records_dir / f"{rec1.experiment_id}.json")
        rec2 = ExperimentRecord(
            experiment_id="nomatch_b_20260101_def67890",
            name="nomatch_b",
            fingerprint="b" * 64,
            training_metrics={"test_ic": 0.80},  # better, but doesn't match signature
            training_config={"model": {"model_type": "deeplob"}},
            record_type="training",
            contract_version="2.2",
        )
        rec2.save(records_dir / f"{rec2.experiment_id}.json")

        from hft_ops.ledger import ExperimentLedger
        ExperimentLedger(tmp_path)._rebuild_index()

        result_id, result_val, n = _find_prior_best_experiment(
            ledger_dir=tmp_path,
            match_signature={"model_type": "tlob"},
            metric_name="test_ic",
            exclude_experiment_name="nothing",
        )
        assert n == 1
        assert result_val == 0.30  # matched tlob, not the higher deeplob


# =============================================================================
# Layer 3: PostTrainingGateRunner end-to-end
# =============================================================================


@pytest.fixture
def runner():
    return PostTrainingGateRunner()


@pytest.fixture
def make_manifest():
    def _make(
        enabled: bool = True,
        on_regression: str = "warn",
        min_metric_floor: float = 0.05,
        min_ratio_vs_prior_best: float = 0.9,
        training_output_dir: str = "training_out",
    ) -> ExperimentManifest:
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=training_output_dir,
                trainer_config={
                    "model": {"model_type": "tlob"},
                    "data": {"labeling_strategy": "regression"},
                },
                horizon_value=10,
            ),
            post_training_gate=PostTrainingGateStage(
                enabled=enabled,
                on_regression=on_regression,
                min_metric_floor=min_metric_floor,
                min_ratio_vs_prior_best=min_ratio_vs_prior_best,
            ),
            ),
        )
        return manifest
    return _make


@pytest.fixture
def make_ops_config(tmp_path: Path):
    def _make(training_output_dir: Path) -> OpsConfig:
        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        # Ensure the training output_dir resolves under tmp_path
        (tmp_path / "trainer").mkdir(exist_ok=True)
        return ops
    return _make


class TestValidateInputs:
    def test_valid_passes(self, runner, make_manifest):
        manifest = make_manifest()
        ops = MagicMock()
        errors = runner.validate_inputs(manifest, ops)
        assert errors == []

    def test_invalid_on_regression(self, runner, make_manifest):
        manifest = make_manifest(on_regression="invalid")
        errors = runner.validate_inputs(manifest, MagicMock())
        assert any("on_regression" in e for e in errors)

    def test_negative_floor(self, runner, make_manifest):
        manifest = make_manifest(min_metric_floor=-0.1)
        errors = runner.validate_inputs(manifest, MagicMock())
        assert any("min_metric_floor" in e for e in errors)

    def test_ratio_out_of_bounds(self, runner, make_manifest):
        manifest = make_manifest(min_ratio_vs_prior_best=1.5)
        errors = runner.validate_inputs(manifest, MagicMock())
        assert any("min_ratio_vs_prior_best" in e for e in errors)

    def test_training_disabled_rejected(self, runner, make_manifest):
        manifest = make_manifest()
        manifest.stages.training.enabled = False
        errors = runner.validate_inputs(manifest, MagicMock())
        assert any("training.enabled" in e for e in errors)


class TestRunPassPath:
    def test_pass_when_good_ic_no_prior(self, runner, tmp_path: Path):
        """First-of-kind experiment: floor pass, no prior to compare → pass."""
        train_out = tmp_path / "train_out"
        train_out.mkdir()
        (train_out / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.38, "test_rmse": 5.0})
        )

        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=str(train_out),
                trainer_config={
                    "model": {"model_type": "tlob"},
                    "data": {"labeling_strategy": "regression"},
                },
            ),
            post_training_gate=PostTrainingGateStage(enabled=True),
            ),
        )

        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        (tmp_path / "trainer").mkdir()

        result = runner.run(manifest, ops)
        assert result.status == StageStatus.COMPLETED
        gate = result.captured_metrics["post_training_gate"]
        assert gate["status"] == "pass"
        assert gate["primary_metric_name"] == "test_ic"
        assert gate["primary_metric_value"] == 0.38

    def test_fail_below_floor_warn_mode(self, runner, tmp_path: Path):
        """Below-floor IC with on_regression=warn → COMPLETED + warn status."""
        train_out = tmp_path / "train_out"
        train_out.mkdir()
        (train_out / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.02})  # below 0.05 floor
        )

        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=str(train_out),
                trainer_config={"model": {"model_type": "tlob"}},
            ),
            post_training_gate=PostTrainingGateStage(
                enabled=True, on_regression="warn",
            ),
            ),
        )
        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        (tmp_path / "trainer").mkdir()

        result = runner.run(manifest, ops)
        # warn mode: status stays COMPLETED; gate.status == "warn"
        assert result.status == StageStatus.COMPLETED
        gate = result.captured_metrics["post_training_gate"]
        assert gate["status"] == "warn"
        # Floor check must have failed
        floor_check = next(c for c in gate["checks"] if c["name"] == "floor")
        assert floor_check["status"] == "fail"

    def test_fail_below_floor_abort_mode(self, runner, tmp_path: Path):
        """Below-floor IC with on_regression=abort → FAILED."""
        train_out = tmp_path / "train_out"
        train_out.mkdir()
        (train_out / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.02})
        )
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=str(train_out),
                trainer_config={"model": {"model_type": "tlob"}},
            ),
            post_training_gate=PostTrainingGateStage(
                enabled=True, on_regression="abort",
            ),
            ),
        )
        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        (tmp_path / "trainer").mkdir()

        result = runner.run(manifest, ops)
        assert result.status == StageStatus.FAILED
        assert "regression gate failed" in result.error_message.lower()

    def test_record_only_never_fails(self, runner, tmp_path: Path):
        """Below-floor IC with on_regression=record_only → pass, silent."""
        train_out = tmp_path / "train_out"
        train_out.mkdir()
        (train_out / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.02})
        )
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=str(train_out),
                trainer_config={"model": {"model_type": "tlob"}},
            ),
            post_training_gate=PostTrainingGateStage(
                enabled=True, on_regression="record_only",
            ),
            ),
        )
        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        (tmp_path / "trainer").mkdir()

        result = runner.run(manifest, ops)
        assert result.status == StageStatus.COMPLETED
        gate = result.captured_metrics["post_training_gate"]
        assert gate["status"] == "pass"  # record_only forces pass

    def test_gate_report_persisted(self, runner, tmp_path: Path):
        """gate_report.json must be written to output_dir."""
        train_out = tmp_path / "train_out"
        train_out.mkdir()
        (train_out / "test_metrics.json").write_text(
            json.dumps({"test_ic": 0.38, "test_rmse": 3.0})
        )
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(
                enabled=True,
                output_dir=str(train_out),
                trainer_config={"model": {"model_type": "tlob"}},
            ),
            post_training_gate=PostTrainingGateStage(enabled=True),
            ),
        )
        ops = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path))
        (tmp_path / "trainer").mkdir()

        result = runner.run(manifest, ops)
        assert result.status == StageStatus.COMPLETED

        # gate_report.json in the runs_dir (PipelinePaths.runs_dir
        # resolves to pipeline_root/hft-ops/ledger/runs).
        report_path = (
            tmp_path / "hft-ops" / "ledger" / "runs" / "test_exp"
            / "post_training_gate" / "gate_report.json"
        )
        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
        assert report["status"] == "pass"
        assert "checks" in report


class TestDryRun:
    def test_dry_run_skips(self, runner, tmp_path: Path):
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_exp"),
            stages=Stages(
            training=TrainingStage(enabled=True, output_dir="irrelevant"),
            post_training_gate=PostTrainingGateStage(enabled=True),
            ),
        )
        ops = OpsConfig(
            paths=PipelinePaths(pipeline_root=tmp_path),
            dry_run=True,
        )

        result = runner.run(manifest, ops)
        assert result.status == StageStatus.SKIPPED
