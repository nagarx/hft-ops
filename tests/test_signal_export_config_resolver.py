"""Phase 7.5-A (2026-04-23): SignalExportRunner config resolution tests.

Locks the 3-tier fallback cascade for resolving the trainer YAML path
passed to `export_signals.py --config <path>`:

  Priority 1 — `stages.signal_export.config` (explicit escape hatch)
  Priority 2 — `<training.output_dir>/config.yaml` (auto-persisted by
               train.py after Phase 7.5-A+ extension)
  Priority 3 — `stages.training.config` (legacy wrapper manifest path)

Closes Bug #2 from the Frame 5 Task 1 audit: SignalExportRunner previously
passed `--experiment <name>` which `export_signals.py` does NOT accept
(requires `--config <path>`). Prior bug was silent because signal_export
stage had never been exercised live.

Plus 2 validator tests:
  - `split="train"` is rejected (export_signals.py accepts val|test only)
  - Validator cross-checks that at least one config-resolution tier can
    succeed at manifest-load time
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from hft_ops.stages.signal_export import (
    _resolve_signal_export_config,
    SignalExportRunner,
)


# =============================================================================
# Stubs (minimal shapes for the resolver function signature)
# =============================================================================


@dataclass
class _StubStage:
    config: Optional[str] = None


@dataclass
class _StubTraining:
    output_dir: str = ""
    config: str = ""
    enabled: bool = False


@dataclass
class _StubStages:
    signal_export: _StubStage
    training: _StubTraining


@dataclass
class _StubExperiment:
    name: str = "unit_test_exp"


@dataclass
class _StubManifest:
    stages: _StubStages
    experiment: _StubExperiment


@dataclass
class _StubPaths:
    pipeline_root: Path

    def resolve(self, p: str) -> Path:
        pth = Path(p)
        if pth.is_absolute():
            return pth
        return self.pipeline_root / pth


@dataclass
class _StubConfig:
    paths: _StubPaths


# =============================================================================
# Resolver cascade tests (6)
# =============================================================================


class TestResolverCascade:
    """Lock the 3-tier priority order + miss-cases."""

    def test_priority_1_explicit_escape_hatch_wins(self, tmp_path: Path):
        """When stage.config is set AND file exists, use it (ignore P2/P3)."""
        p1 = tmp_path / "escape_hatch_config.yaml"
        p1.write_text("name: test\n")
        p2 = tmp_path / "outputs" / "config.yaml"
        p2.parent.mkdir()
        p2.write_text("name: unused_p2\n")
        p3 = tmp_path / "legacy_wrapper_config.yaml"
        p3.write_text("name: unused_p3\n")

        stage = _StubStage(config="escape_hatch_config.yaml")
        training = _StubTraining(
            output_dir="outputs",
            config="legacy_wrapper_config.yaml",
        )
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))

        resolved = _resolve_signal_export_config(stage, manifest, config)
        assert resolved == p1, (
            f"Priority 1 must win; got {resolved!r}; expected {p1!r}"
        )

    def test_priority_2_auto_persisted_config_yaml(self, tmp_path: Path):
        """When Priority 1 unset and <training.output_dir>/config.yaml exists,
        use it (ignore P3)."""
        p2 = tmp_path / "outputs" / "config.yaml"
        p2.parent.mkdir()
        p2.write_text("name: p2_wins\n")
        p3 = tmp_path / "legacy_wrapper.yaml"
        p3.write_text("name: unused\n")

        stage = _StubStage(config=None)
        training = _StubTraining(
            output_dir="outputs",
            config="legacy_wrapper.yaml",
        )
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))

        resolved = _resolve_signal_export_config(stage, manifest, config)
        assert resolved == p2, (
            f"Priority 2 must win when P1 absent + file exists; got {resolved!r}"
        )

    def test_priority_3_legacy_wrapper_fallback(self, tmp_path: Path):
        """When Priorities 1 and 2 both miss, fall through to Priority 3."""
        # Priority 2 file deliberately NOT created (mimics training not yet run)
        p3 = tmp_path / "legacy_wrapper.yaml"
        p3.write_text("name: p3_wins\n")

        stage = _StubStage(config=None)
        training = _StubTraining(
            output_dir="outputs",  # dir doesn't exist yet
            config="legacy_wrapper.yaml",
        )
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))

        resolved = _resolve_signal_export_config(stage, manifest, config)
        assert resolved == p3

    def test_priority_1_explicitly_missing_does_not_fall_through(
        self, tmp_path: Path,
    ):
        """When stage.config is set but file doesn't exist, return None even
        if Priorities 2/3 could resolve. Operator explicit intent > fallback."""
        # P2 file exists at <output_dir>/config.yaml
        p2 = tmp_path / "outputs" / "config.yaml"
        p2.parent.mkdir()
        p2.write_text("name: p2_candidate\n")
        # P3 file exists
        p3 = tmp_path / "legacy.yaml"
        p3.write_text("name: p3_candidate\n")

        stage = _StubStage(config="operator_set_this_but_missing.yaml")
        training = _StubTraining(output_dir="outputs", config="legacy.yaml")
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))

        resolved = _resolve_signal_export_config(stage, manifest, config)
        assert resolved is None, (
            "Operator explicitly set stage.config to a missing path — "
            "must NOT silently fall back to Priorities 2/3 (hides typo)"
        )

    def test_all_three_priorities_missing_returns_none(self, tmp_path: Path):
        """No config anywhere → None (caller fails with StageResult(FAILED))."""
        stage = _StubStage(config=None)
        training = _StubTraining(output_dir="", config="")
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))
        assert _resolve_signal_export_config(stage, manifest, config) is None

    def test_priority_2_missing_config_yaml_falls_through(self, tmp_path: Path):
        """<training.output_dir> exists but config.yaml doesn't — fall to P3.
        Mimics post-training-failure state where output_dir exists but
        config.yaml wasn't persisted."""
        (tmp_path / "outputs").mkdir()  # dir exists, no config.yaml
        p3 = tmp_path / "wrapper.yaml"
        p3.write_text("name: p3\n")

        stage = _StubStage(config=None)
        training = _StubTraining(output_dir="outputs", config="wrapper.yaml")
        manifest = _StubManifest(
            stages=_StubStages(signal_export=stage, training=training),
            experiment=_StubExperiment(),
        )
        config = _StubConfig(paths=_StubPaths(pipeline_root=tmp_path))
        assert _resolve_signal_export_config(stage, manifest, config) == p3


# =============================================================================
# Validator tests (integrated with the real SignalExportRunner)
# =============================================================================


class TestValidatorSplitChoices:
    """Locks export_signals.py's `--split` argparse choices (val | test only)."""

    def _make_manifest(self, split: str, enabled: bool = False):
        """Build a minimal manifest for validator invocation."""
        # Import here to avoid module-level coupling
        from hft_ops.manifest.schema import (
            ExperimentHeader,
            ExperimentManifest,
            SignalExportStage,
            Stages,
            TrainingStage,
        )
        sig = SignalExportStage(enabled=enabled, split=split, config=None)
        training = TrainingStage(enabled=True, config="cfg.yaml")
        # enabled=False on other stages; default construction sufficient
        stages = Stages(training=training, signal_export=sig)
        return ExperimentManifest(
            experiment=ExperimentHeader(
                name="test_exp",
                description="test",
                hypothesis="test",
            ),
            stages=stages,
        )

    def test_split_train_rejected(self, tmp_path: Path):
        """"train" is not accepted by export_signals.py argparse."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        manifest = self._make_manifest(split="train", enabled=True)
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert any("split" in e and "train" in e for e in errors), (
            f"Expected error about split='train' being rejected; got: {errors}"
        )

    def test_split_val_accepted(self, tmp_path: Path):
        """"val" is a valid split per export_signals.py argparse."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        manifest = self._make_manifest(split="val", enabled=True)
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        # Validator may produce OTHER errors (trainer_dir not found, script
        # not found, etc.) but MUST NOT produce a "Invalid signal_export.split"
        # error for split="val".
        assert not any("Invalid signal_export.split" in e for e in errors)

    def test_split_test_accepted(self, tmp_path: Path):
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        manifest = self._make_manifest(split="test", enabled=True)
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert not any("Invalid signal_export.split" in e for e in errors)


class TestValidatorConfigResolutionCrossCheck:
    """Locks the manifest-load-time cross-check that at least one of the 3
    config-resolution tiers can succeed when signal_export is enabled."""

    def _make_manifest(
        self,
        *,
        sig_enabled: bool,
        sig_config: Optional[str],
        training_enabled: bool,
        training_config: str,
    ):
        from hft_ops.manifest.schema import (
            ExperimentHeader,
            ExperimentManifest,
            SignalExportStage,
            Stages,
            TrainingStage,
        )
        sig = SignalExportStage(
            enabled=sig_enabled,
            split="test",
            config=sig_config,
        )
        training = TrainingStage(
            enabled=training_enabled,
            config=training_config,
        )
        stages = Stages(training=training, signal_export=sig)
        return ExperimentManifest(
            experiment=ExperimentHeader(
                name="test_exp",
                description="test",
                hypothesis="test",
            ),
            stages=stages,
        )

    def test_sig_disabled_no_config_error(self, tmp_path: Path):
        """When signal_export.enabled=False, config-resolution is not checked."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        # Create trainer_dir so validator progresses past the early return
        (tmp_path / "lob-model-trainer").mkdir()
        manifest = self._make_manifest(
            sig_enabled=False,
            sig_config=None,
            training_enabled=False,
            training_config="",
        )
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert not any("resolve trainer config" in e for e in errors)

    def test_sig_enabled_training_enabled_passes(self, tmp_path: Path):
        """Priority 2 will be auto-populated at runtime — validator passes
        even with sig_config + training_config both unset."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        (tmp_path / "lob-model-trainer").mkdir()
        manifest = self._make_manifest(
            sig_enabled=True,
            sig_config=None,
            training_enabled=True,
            training_config="",  # empty; training stage runs inline
        )
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert not any("resolve trainer config" in e for e in errors)

    def test_sig_enabled_training_disabled_no_configs_fails(
        self, tmp_path: Path,
    ):
        """All 3 tiers will miss at runtime → fail-loud at validate time."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        (tmp_path / "lob-model-trainer").mkdir()
        manifest = self._make_manifest(
            sig_enabled=True,
            sig_config=None,
            training_enabled=False,
            training_config="",
        )
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert any("resolve trainer config" in e for e in errors), (
            f"Expected config-resolution failure error; got: {errors}"
        )

    def test_sig_enabled_with_explicit_escape_hatch_passes(
        self, tmp_path: Path,
    ):
        """stage.config set → validator passes even if training disabled."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        (tmp_path / "lob-model-trainer").mkdir()
        manifest = self._make_manifest(
            sig_enabled=True,
            sig_config="some_config.yaml",
            training_enabled=False,
            training_config="",
        )
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert not any("resolve trainer config" in e for e in errors)

    def test_sig_enabled_with_legacy_training_config_passes(
        self, tmp_path: Path,
    ):
        """training.config set (legacy wrapper) → validator passes."""
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        (tmp_path / "lob-model-trainer").mkdir()
        manifest = self._make_manifest(
            sig_enabled=True,
            sig_config=None,
            training_enabled=False,
            training_config="legacy_wrapper.yaml",
        )
        paths = PipelinePaths(pipeline_root=tmp_path)
        config = OpsConfig(paths=paths)

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        assert not any("resolve trainer config" in e for e in errors)
