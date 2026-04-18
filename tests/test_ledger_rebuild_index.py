"""Phase 7 Stage 7.4 post-validation C2 fix: `hft-ops ledger rebuild-index`.

Locks:
1. Invoking the CLI rebuilds index.json from records/*.json.
2. After whitelist expansion (Phase 7.4 added 7 regression metric keys),
   historical records that were originally projected with the OLD
   whitelist get re-projected with the NEW whitelist on rebuild, making
   them visible to the PostTrainingGate's prior-best queries.
3. `--dry-run` does not modify index.json.
4. The subcommand handles an empty records dir gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from hft_contracts.experiment_record import ExperimentRecord
from hft_ops.cli import main
from hft_ops.ledger import ExperimentLedger


def _write_ledger_record(
    records_dir: Path,
    experiment_id: str,
    *,
    name: str = "test_exp",
    training_metrics: dict = None,
    record_type: str = "training",
) -> ExperimentRecord:
    """Build + write a record file directly (bypasses register path)."""
    records_dir.mkdir(parents=True, exist_ok=True)
    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=name,
        fingerprint="a" * 64,
        training_metrics=training_metrics or {},
        training_config={"model": {"model_type": "tlob"}},
        record_type=record_type,
        contract_version="2.2",
    )
    record.save(records_dir / f"{experiment_id}.json")
    return record


class TestRebuildIndexCLI:
    def test_basic_rebuild_reports_count(self, tmp_path: Path):
        """Rebuild with 3 records produces an index with 3 entries."""
        records = tmp_path / "hft-ops" / "ledger" / "records"  # PipelinePaths layout
        for i, name in enumerate(["exp_a", "exp_b", "exp_c"]):
            _write_ledger_record(
                records,
                f"{name}_20260101_abc{i:05d}",
                name=name,
                training_metrics={"test_ic": 0.1 * (i + 1)},
            )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--pipeline-root", str(tmp_path), "ledger", "rebuild-index"],
        )
        assert result.exit_code == 0, result.output
        assert "Rebuilding index from 3 record files" in result.output

        # Verify index.json landed at the PipelinePaths-derived location
        index_path = tmp_path / "hft-ops" / "ledger" / "index.json"
        assert index_path.exists()
        with open(index_path) as f:
            idx = json.load(f)
        assert len(idx) == 3

    def test_rebuild_repopulates_new_whitelist_keys(self, tmp_path: Path):
        """Simulate a whitelist expansion scenario: records on disk have
        regression metrics, but the current cached index.json was written
        with an OLD whitelist that dropped them. Rebuild re-projects with
        the CURRENT whitelist and surfaces the metrics.
        """
        # Use PipelinePaths layout: tmp_path/hft-ops/ledger/records/
        ledger_dir = tmp_path / "hft-ops" / "ledger"
        records = ledger_dir / "records"
        records.mkdir(parents=True)

        # Record with regression metric
        _write_ledger_record(
            records,
            "regression_exp_20260101_abc12345",
            name="regression_exp",
            training_metrics={"test_ic": 0.38, "test_r2": 0.12},
        )

        # Simulate stale index.json: write ONLY the experiment_id + name
        # (missing regression fields as if produced by the old whitelist)
        index_path = ledger_dir / "index.json"
        stale_entry = {
            "experiment_id": "regression_exp_20260101_abc12345",
            "name": "regression_exp",
            "training_metrics": {},  # pre-Phase-7.4 whitelist: empty
        }
        with open(index_path, "w") as f:
            json.dump([stale_entry], f)

        # Now rebuild
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--pipeline-root", str(tmp_path), "ledger", "rebuild-index"],
        )
        assert result.exit_code == 0, result.output

        # Re-read index: regression metrics should now be present
        with open(index_path) as f:
            refreshed = json.load(f)
        assert len(refreshed) == 1
        entry = refreshed[0]
        # Post-Phase-7.4 whitelist includes test_ic + test_r2
        assert entry["training_metrics"]["test_ic"] == 0.38
        assert entry["training_metrics"]["test_r2"] == 0.12

    def test_dry_run_does_not_write(self, tmp_path: Path):
        """--dry-run reports counts but leaves index.json unchanged."""
        ledger_dir = tmp_path / "hft-ops" / "ledger"
        records = ledger_dir / "records"
        records.mkdir(parents=True)
        _write_ledger_record(
            records,
            "exp_20260101_abc12345",
            training_metrics={"test_ic": 0.5},
        )

        # Write a stale index with obviously-wrong content
        index_path = ledger_dir / "index.json"
        wrong_payload = [{"experiment_id": "WRONG", "name": "stale"}]
        with open(index_path, "w") as f:
            json.dump(wrong_payload, f)
        original_mtime = index_path.stat().st_mtime

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--pipeline-root", str(tmp_path),
                "ledger", "rebuild-index", "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output

        # Index file must be UNCHANGED
        with open(index_path) as f:
            unchanged = json.load(f)
        assert unchanged == wrong_payload, (
            "--dry-run must not modify index.json"
        )

    def test_empty_records_dir_graceful(self, tmp_path: Path):
        """No records/ dir → graceful message, no crash, exit 0."""
        # Ensure ledger_dir exists but records/ doesn't
        ledger_dir = tmp_path / "hft-ops" / "ledger"
        ledger_dir.mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--pipeline-root", str(tmp_path), "ledger", "rebuild-index"],
        )
        assert result.exit_code == 0, result.output
        assert "nothing to rebuild" in result.output.lower()


class TestGateReportPersistedInRecord:
    """Phase 7 Stage 7.4 post-validation B-H1 fix: gate report is persisted
    into ExperimentRecord.training_metrics via cli.py::_record_experiment.

    These tests use the _record_experiment helper directly rather than
    round-tripping the full CLI — keeps the test focused on the
    integration point.
    """

    def test_gate_report_lands_in_training_metrics(self, tmp_path: Path):
        """When post_training_gate ran, its captured_metrics dict
        containing the full GateReport is nested under
        training_metrics["post_training_gate"] in the ExperimentRecord.
        """
        from hft_ops.cli import _record_experiment
        from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageResult, StageStatus

        paths = PipelinePaths(pipeline_root=tmp_path)

        # Build fake StageResults for training + post_training_gate
        training_result = StageResult(stage_name="training")
        training_result.status = StageStatus.COMPLETED
        training_result.captured_metrics = {"test_ic": 0.38, "test_r2": 0.12}

        gate_report = {
            "status": "pass",
            "primary_metric_name": "test_ic",
            "primary_metric_value": 0.38,
            "prior_best_experiment_id": "",
            "prior_best_metric_value": None,
            "n_matching_prior_experiments": 0,
            "checks": [],
            "match_signature": {"model_type": "tlob"},
        }
        gate_result = StageResult(stage_name="post_training_gate")
        gate_result.status = StageStatus.COMPLETED
        gate_result.captured_metrics = {
            "post_training_gate": gate_report,
            "post_training_gate_summary": "post_training_gate: PASS | test_ic=0.3800",
        }

        results = {
            "training": training_result,
            "post_training_gate": gate_result,
        }

        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_gate_persistence"),
        )

        experiment_id = _record_experiment(
            manifest, paths, fingerprint="b" * 64,
            results=results, total_duration=1.0,
        )

        # Load the record and verify gate report is present
        ledger = ExperimentLedger(paths.ledger_dir)
        record = ledger.get(experiment_id)
        assert record is not None

        # Gate report lives under training_metrics["post_training_gate"]
        assert "post_training_gate" in record.training_metrics
        gate_data = record.training_metrics["post_training_gate"]
        assert gate_data["status"] == "pass"
        assert gate_data["primary_metric_name"] == "test_ic"
        assert gate_data["primary_metric_value"] == 0.38

        # One-line summary also persisted for grep-friendly display
        assert "post_training_gate_summary" in record.training_metrics
        assert "PASS" in record.training_metrics["post_training_gate_summary"]

    def test_gate_report_absent_when_gate_not_run(self, tmp_path: Path):
        """When post_training_gate was not in results (disabled / skipped),
        training_metrics does NOT contain the gate keys."""
        from hft_ops.cli import _record_experiment
        from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageResult, StageStatus

        paths = PipelinePaths(pipeline_root=tmp_path)
        training_result = StageResult(stage_name="training")
        training_result.status = StageStatus.COMPLETED
        training_result.captured_metrics = {"test_ic": 0.38}

        results = {"training": training_result}  # gate key absent

        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="test_no_gate"),
        )
        experiment_id = _record_experiment(
            manifest, paths, fingerprint="c" * 64,
            results=results, total_duration=1.0,
        )

        record = ExperimentLedger(paths.ledger_dir).get(experiment_id)
        assert record is not None
        assert "post_training_gate" not in record.training_metrics
        assert "post_training_gate_summary" not in record.training_metrics
        # Training metrics preserved
        assert record.training_metrics["test_ic"] == 0.38
