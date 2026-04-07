"""Tests for manifest schema and loader."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from hft_ops.manifest.loader import load_manifest, _resolve_variables
from hft_ops.manifest.schema import (
    BacktestingStage,
    BacktestParams,
    DatasetAnalysisStage,
    ExperimentHeader,
    ExperimentManifest,
    ExtractionStage,
    Stages,
    TrainingStage,
)


class TestExperimentHeader:
    def test_defaults(self):
        h = ExperimentHeader(name="test")
        assert h.name == "test"
        assert h.description == ""
        assert h.hypothesis == ""
        assert h.contract_version == ""
        assert h.tags == []

    def test_full_construction(self):
        h = ExperimentHeader(
            name="exp1",
            description="desc",
            hypothesis="hyp",
            contract_version="2.2",
            tags=["a", "b"],
        )
        assert h.tags == ["a", "b"]
        assert h.contract_version == "2.2"


class TestBacktestParams:
    def test_defaults(self):
        p = BacktestParams()
        assert p.initial_capital == 100_000.0
        assert p.position_size == 0.1
        assert p.spread_bps == 1.0
        assert p.slippage_bps == 0.5
        assert p.threshold == 0.0
        assert p.no_short is False
        assert p.device == "cpu"


class TestExperimentManifest:
    def test_defaults(self):
        m = ExperimentManifest(
            experiment=ExperimentHeader(name="test")
        )
        assert m.pipeline_root == ".."
        assert m.stages.extraction.enabled is True
        assert m.stages.raw_analysis.enabled is False
        assert m.stages.dataset_analysis.enabled is True
        assert m.stages.training.enabled is True
        assert m.stages.backtesting.enabled is True


class TestVariableResolution:
    def test_simple_reference(self):
        raw = {
            "experiment": {"name": "foo"},
            "stages": {"training": {"output": "${experiment.name}_out"}},
        }
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["stages"]["training"]["output"] == "foo_out"

    def test_timestamp_variable(self):
        raw = {"dir": "runs/${timestamp}"}
        now = datetime(2026, 3, 5, 12, 30, 45, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["dir"] == "runs/20260305T123045"

    def test_date_variable(self):
        raw = {"dir": "runs/${date}"}
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["dir"] == "runs/2026-03-05"

    def test_deferred_variable_preserved(self):
        raw = {"idx": "${resolved.horizon_idx}"}
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["idx"] == "${resolved.horizon_idx}"

    def test_transitive_resolution(self):
        raw = {
            "a": {"val": "hello"},
            "b": {"ref": "${a.val}"},
            "c": {"ref": "${b.ref}_world"},
        }
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["c"]["ref"] == "hello_world"

    def test_unresolvable_preserved(self):
        raw = {"x": "${nonexistent.key}"}
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["x"] == "${nonexistent.key}"

    def test_list_resolution(self):
        raw = {
            "name": "foo",
            "items": ["${name}_a", "${name}_b"],
        }
        now = datetime(2026, 3, 5, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)
        assert resolved["items"] == ["foo_a", "foo_b"]


class TestLoadManifest:
    def test_load_basic(self, sample_manifest_yaml: Path):
        now = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
        manifest = load_manifest(sample_manifest_yaml, now=now)

        assert manifest.experiment.name == "test_experiment"
        assert manifest.experiment.contract_version == "2.2"
        assert "test" in manifest.experiment.tags
        assert manifest.stages.extraction.enabled is True
        assert manifest.stages.raw_analysis.enabled is False
        assert manifest.stages.training.horizon_value == 100

    def test_variable_resolution_in_load(self, sample_manifest_yaml: Path):
        now = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
        manifest = load_manifest(sample_manifest_yaml, now=now)
        assert manifest.stages.training.overrides.get("data.data_dir") == (
            "data/exports/nvda_test"
        )

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.yaml")

    def test_missing_name_raises(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("experiment:\n  description: no name\n")
        with pytest.raises(ValueError, match="experiment.name"):
            load_manifest(bad_yaml)

    def test_empty_file_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError):
            load_manifest(empty)

    def test_backtest_params_parsed(self, sample_manifest_yaml: Path):
        manifest = load_manifest(sample_manifest_yaml)
        params = manifest.stages.backtesting.params
        assert params.initial_capital == 100_000
        assert params.spread_bps == 1.0

    def test_deferred_horizon_idx(self, sample_manifest_yaml: Path):
        manifest = load_manifest(sample_manifest_yaml)
        assert manifest.stages.backtesting.horizon_idx is None
