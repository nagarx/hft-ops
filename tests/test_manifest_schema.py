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


class TestBacktestingStageScript:
    """BacktestingStage.script field (Phase 0.1) enables multi-script backtests."""

    def test_default_script(self):
        from hft_ops.manifest.schema import BacktestingStage
        stage = BacktestingStage()
        assert stage.script == "scripts/backtest_deeplob.py"

    def test_custom_script(self):
        from hft_ops.manifest.schema import BacktestingStage
        stage = BacktestingStage(script="scripts/run_regression_backtest.py")
        assert stage.script == "scripts/run_regression_backtest.py"

    def test_script_loaded_from_yaml(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_bt_script
pipeline_root: ".."
stages:
  backtesting:
    enabled: true
    script: "scripts/run_readability_backtest.py"
    signals_dir: "outputs/test/signals"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.backtesting.script == "scripts/run_readability_backtest.py"
        assert m.stages.backtesting.signals_dir == "outputs/test/signals"

    def test_params_file_field(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_pf
pipeline_root: ".."
stages:
  backtesting:
    params_file: "configs/bt_params.yaml"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.backtesting.params_file == "configs/bt_params.yaml"


class TestSignalExportStage:
    """SignalExportStage (Phase 0.2) is a first-class stage between training and backtesting."""

    def test_defaults(self):
        from hft_ops.manifest.schema import SignalExportStage
        stage = SignalExportStage()
        assert stage.enabled is False
        assert stage.script == "scripts/export_signals.py"
        assert stage.checkpoint == ""
        assert stage.split == "test"
        assert stage.output_dir == ""
        assert stage.extra_args == []

    def test_full_construction(self):
        from hft_ops.manifest.schema import SignalExportStage
        stage = SignalExportStage(
            enabled=True,
            script="scripts/export_hmhp_signals.py",
            checkpoint="outputs/exp1/checkpoints/best.pt",
            split="val",
            output_dir="outputs/exp1/signals/val",
            extra_args=["--verbose"],
        )
        assert stage.enabled is True
        assert stage.split == "val"
        assert stage.extra_args == ["--verbose"]

    def test_loaded_from_yaml(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_se
pipeline_root: ".."
stages:
  signal_export:
    enabled: true
    script: "scripts/export_hmhp_signals.py"
    checkpoint: "${stages.training.output_dir}/checkpoints/best.pt"
    split: test
    output_dir: "${stages.training.output_dir}/signals/test"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.signal_export.enabled is True
        assert m.stages.signal_export.script == "scripts/export_hmhp_signals.py"
        assert m.stages.signal_export.split == "test"
        # ${...} variables remain as strings until resolution
        assert "${" in m.stages.signal_export.checkpoint

    def test_signal_export_present_in_stages(self):
        from hft_ops.manifest.schema import Stages, SignalExportStage
        s = Stages()
        assert isinstance(s.signal_export, SignalExportStage)
        assert s.signal_export.enabled is False


class TestTrainingStageInlineConfig:
    """Phase 1.1: TrainingStage.trainer_config enables wrapper-less manifests."""

    def test_default_trainer_config_is_none(self):
        from hft_ops.manifest.schema import TrainingStage
        stage = TrainingStage()
        assert stage.trainer_config is None

    def test_legacy_path_config_still_works(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_legacy
pipeline_root: ".."
stages:
  training:
    enabled: true
    config: "lob-model-trainer/configs/experiments/test.yaml"
    overrides:
      data.data_dir: "/path/to/data"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.training.config == "lob-model-trainer/configs/experiments/test.yaml"
        assert m.stages.training.trainer_config is None

    def test_inline_trainer_config_loaded(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_inline
pipeline_root: ".."
stages:
  training:
    enabled: true
    trainer_config:
      name: InlineExperiment
      data:
        data_dir: "/path/to/data"
        feature_count: 98
      model:
        model_type: tlob
        input_size: 98
      train:
        epochs: 10
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.training.config == ""
        assert m.stages.training.trainer_config is not None
        assert m.stages.training.trainer_config["name"] == "InlineExperiment"
        assert m.stages.training.trainer_config["data"]["feature_count"] == 98
        assert m.stages.training.trainer_config["model"]["model_type"] == "tlob"

    def test_both_path_and_inline_raises(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_both
pipeline_root: ".."
stages:
  training:
    enabled: true
    config: "some/path.yaml"
    trainer_config:
      name: InlineExperiment
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        with pytest.raises(ValueError, match="mutually exclusive|EITHER"):
            load_manifest(manifest_path)

    def test_trainer_config_must_be_dict(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_wrong_type
pipeline_root: ".."
stages:
  training:
    enabled: true
    trainer_config: "this is not a dict"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        with pytest.raises(ValueError, match="must be a dict"):
            load_manifest(manifest_path)

    def test_empty_dict_normalized_to_none(self, tmp_path: Path):
        """Empty dict {} is treated as unset (None), so validator can catch 'neither set'."""
        yaml_text = """
experiment:
  name: test_empty
pipeline_root: ".."
stages:
  training:
    enabled: true
    trainer_config: {}
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.training.trainer_config is None


class TestValidationStage:
    """Phase 2a: ValidationStage dataclass + loader parsing."""

    def test_defaults(self):
        from hft_ops.manifest.schema import ValidationStage
        v = ValidationStage()
        assert v.enabled is True
        assert v.on_fail == "warn"   # CRITICAL: default is warn, not abort
        assert v.target_horizon == ""
        assert v.min_ic == 0.05
        assert v.min_ic_count == 2
        assert v.min_return_std_bps == 5.0
        assert v.min_stability == 2.0
        assert v.sample_size == 200_000
        assert v.n_folds == 20
        assert v.allow_zero_ic_names == []
        assert v.profile_ref == ""
        assert v.output_dir == ""

    def test_validation_in_stages_container(self):
        from hft_ops.manifest.schema import Stages, ValidationStage
        s = Stages()
        assert isinstance(s.validation, ValidationStage)

    def test_loaded_from_yaml(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_val
pipeline_root: ".."
stages:
  validation:
    enabled: true
    on_fail: abort
    target_horizon: H10
    min_ic: 0.04
    min_ic_count: 3
    sample_size: 100000
    allow_zero_ic_names:
      - time_regime
      - dark_share
    profile_ref: outputs/eval/profiles.json
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        v = m.stages.validation
        assert v.enabled is True
        assert v.on_fail == "abort"
        assert v.target_horizon == "H10"
        assert v.min_ic == 0.04
        assert v.min_ic_count == 3
        assert v.sample_size == 100_000
        assert v.allow_zero_ic_names == ["time_regime", "dark_share"]
        assert v.profile_ref == "outputs/eval/profiles.json"

    def test_invalid_on_fail_raises(self, tmp_path: Path):
        """Fail-fast on typos — `on_fail: fail` is a common misspelling."""
        yaml_text = """
experiment:
  name: test_typo
pipeline_root: ".."
stages:
  validation:
    on_fail: fail
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        with pytest.raises(ValueError, match="on_fail must be one of"):
            load_manifest(manifest_path)

    def test_allow_zero_ic_names_must_be_list(self, tmp_path: Path):
        yaml_text = """
experiment:
  name: test_azn
pipeline_root: ".."
stages:
  validation:
    allow_zero_ic_names: "time_regime"
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        with pytest.raises(ValueError, match="must be a list"):
            load_manifest(manifest_path)

    def test_default_warn_is_intentional(self, tmp_path: Path):
        """Document the design decision: default is ``warn``, not ``abort``.

        Evaluator CLAUDE.md §Known Limitations explicitly warns against using
        DISCARD as a hard gate — individual-feature IC misses interaction,
        temporal, and context-feature value. ``warn`` default lets the gate
        surface failures while not blocking legitimate experiments.
        """
        yaml_text = """
experiment:
  name: test_default
pipeline_root: ".."
stages:
  validation:
    enabled: true
"""
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml_text)
        m = load_manifest(manifest_path)
        assert m.stages.validation.on_fail == "warn"


class TestFingerprintStability:
    """Phase 2a: Fingerprint MUST NOT change when only validation / metadata differ.

    C1 bug from pre-Phase-2 validation: adding a `validation` block or
    changing gate thresholds would have changed the fingerprint, breaking
    dedup for the SAME underlying experiment. These tests pin the contract.
    """

    def _make_manifest_with_validation(self, tmp_path: Path, name: str, on_fail: str, min_ic: float) -> str:
        """Write a minimal manifest with the specified validation config.
        Both variants reference the same trainer YAML so the training-side
        fingerprint content is identical.
        """
        trainer_yaml = tmp_path / "trainer.yaml"
        trainer_yaml.write_text("data:\n  feature_count: 98\nmodel:\n  model_type: tlob\n")
        manifest_path = tmp_path / f"{name}.yaml"
        manifest_path.write_text(f"""
experiment:
  name: {name}
  contract_version: "2.2"
pipeline_root: "."
stages:
  validation:
    enabled: true
    on_fail: {on_fail}
    min_ic: {min_ic}
  training:
    enabled: true
    config: "trainer.yaml"
""")
        return str(manifest_path)

    def test_validation_changes_do_not_affect_fingerprint(self, tmp_path: Path):
        """Two manifests differing only in gate config → SAME fingerprint."""
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.ledger.dedup import compute_fingerprint
        from hft_ops.paths import PipelinePaths

        paths = PipelinePaths(pipeline_root=tmp_path)

        m1_path = self._make_manifest_with_validation(tmp_path, "m1", "warn", 0.05)
        m2_path = self._make_manifest_with_validation(tmp_path, "m2", "abort", 0.03)

        m1 = load_manifest(m1_path)
        m2 = load_manifest(m2_path)

        fp1 = compute_fingerprint(m1, paths)
        fp2 = compute_fingerprint(m2, paths)

        assert fp1 == fp2, (
            "Validation config changes should NOT affect fingerprint. "
            "The gate is an observation, not a treatment; identical trainer "
            "configs with different gate thresholds should dedupe as the "
            "same underlying experiment."
        )

    def test_inline_vs_path_trainer_configs_with_identical_content_same_fingerprint(self, tmp_path: Path):
        """Legacy `training.config` path and inline `training.trainer_config` → SAME fingerprint
        when the trainer content is identical.
        """
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.ledger.dedup import compute_fingerprint
        from hft_ops.paths import PipelinePaths

        paths = PipelinePaths(pipeline_root=tmp_path)

        # Legacy-path manifest
        trainer_yaml = tmp_path / "trainer.yaml"
        trainer_yaml.write_text("data:\n  feature_count: 98\nmodel:\n  model_type: tlob\n")
        legacy = tmp_path / "legacy.yaml"
        legacy.write_text("""
experiment:
  name: legacy
  contract_version: "2.2"
pipeline_root: "."
stages:
  training:
    enabled: true
    config: "trainer.yaml"
""")

        # Inline manifest with identical trainer content
        inline = tmp_path / "inline.yaml"
        inline.write_text("""
experiment:
  name: inline
  contract_version: "2.2"
pipeline_root: "."
stages:
  training:
    enabled: true
    trainer_config:
      data:
        feature_count: 98
      model:
        model_type: tlob
""")

        fp_legacy = compute_fingerprint(load_manifest(legacy), paths)
        fp_inline = compute_fingerprint(load_manifest(inline), paths)

        assert fp_legacy == fp_inline, (
            "Legacy path and inline trainer_config should fingerprint identically "
            "when they resolve to the same trainer YAML content."
        )
