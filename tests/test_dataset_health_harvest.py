"""Step 6 (2026-05-31): dataset_analysis is no longer a DARK stage.

`record.dataset_health` was NEVER populated (register.py:212 "not populated by");
the analyzer subprocess wrote per-analyzer JSONs the ledger never saw. This
harvests a SCHEMA-LIGHT, FAIL-SOFT summary (resolved report dir + the requested
analyzers/profile/split, deliberately NO count) into `record.dataset_health` — NOT
per-analyzer scalars (run_analysis.py writes N idx-prefixed per-analyzer JSONs
with no stable aggregate; harvesting specific health scalars is a deliberate
follow-on needing a pinned analyzer-output contract).
"""

from __future__ import annotations

from hft_ops.stages.dataset_analysis import _summarize_dataset_health


def _stage(**kw):
    from hft_ops.manifest.schema import DatasetAnalysisStage
    return DatasetAnalysisStage(**kw)


def _config(tmp_path):
    from hft_ops.config import OpsConfig, PipelinePaths
    return OpsConfig(paths=PipelinePaths(pipeline_root=tmp_path), dry_run=False, verbose=False)


# --------------------------------------------------------------------------
# _summarize_dataset_health
# --------------------------------------------------------------------------
def test_summary_with_explicit_output_dir(tmp_path):
    out = tmp_path / "myreports"
    out.mkdir()
    stage = _stage(enabled=True, output_dir="myreports", profile="full")
    dh = _summarize_dataset_health(stage, _config(tmp_path))
    assert dh["report_dir"] == str(out)
    assert dh["profile"] == "full"
    assert dh["split"] == "train"  # schema default
    assert "n_reports" not in dh  # deliberately no count (stale-overcount hazard)


def test_summary_analyzers_override_profile(tmp_path):
    stage = _stage(enabled=True, output_dir="r",
                   analyzers=["data_quality", "return_analysis"], split="test")
    dh = _summarize_dataset_health(stage, _config(tmp_path))
    assert dh["analyzers"] == ["data_quality", "return_analysis"]
    assert "profile" not in dh  # analyzers override profile (mirrors run() dispatch)
    assert dh["split"] == "test"


def test_summary_default_output_dir_when_unset(tmp_path):
    """stage.output_dir unset -> run_analysis.py default outputs/analysis relative
    to the analyzer cwd (dataset_analyzer_dir)."""
    stage = _stage(enabled=True, profile="quick")
    cfg = _config(tmp_path)
    dh = _summarize_dataset_health(stage, cfg)
    expected = str(cfg.paths.dataset_analyzer_dir / "outputs" / "analysis")
    assert dh["report_dir"] == expected


def test_summary_fail_soft_returns_dict(tmp_path):
    stage = _stage(enabled=True)
    dh = _summarize_dataset_health(stage, _config(tmp_path))
    assert isinstance(dh, dict)
    assert "report_dir" in dh


# --------------------------------------------------------------------------
# _record_experiment wire: captured_metrics['dataset_health'] -> record
# --------------------------------------------------------------------------
class TestRecordExperimentDatasetHealthWire:
    @staticmethod
    def _record(tmp_path, *, dataset_health=None, include_da=True):
        from hft_ops.cli import _record_experiment
        from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageResult, StageStatus
        from hft_ops.ledger.ledger import ExperimentLedger

        paths = PipelinePaths(pipeline_root=tmp_path)
        training = StageResult(stage_name="training")
        training.status = StageStatus.COMPLETED
        training.captured_metrics = {"test_ic": 0.3}
        results = {"training": training}
        if include_da:
            da = StageResult(stage_name="dataset_analysis")
            da.status = StageStatus.COMPLETED
            da.captured_metrics = (
                {} if dataset_health is None else {"dataset_health": dataset_health}
            )
            results["dataset_analysis"] = da
        manifest = ExperimentManifest(experiment=ExperimentHeader(name="dh_exp"))
        eid = _record_experiment(
            manifest, paths, fingerprint="f" * 64,
            results=results, total_duration=1.0,
        )
        return ExperimentLedger(paths.ledger_dir).get(eid)

    def test_dataset_health_persisted_on_record(self, tmp_path):
        dh = {"report_dir": "/x/reports", "profile": "full", "split": "train"}
        record = self._record(tmp_path, dataset_health=dh)
        assert record.dataset_health == dh

    def test_empty_dataset_health_leaves_default(self, tmp_path):
        record = self._record(tmp_path, dataset_health={})
        assert record.dataset_health == {}

    def test_no_dataset_analysis_stage_leaves_default(self, tmp_path):
        record = self._record(tmp_path, include_da=False)
        assert record.dataset_health == {}
