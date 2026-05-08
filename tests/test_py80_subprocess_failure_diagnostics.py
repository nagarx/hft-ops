"""Tests for #PY-80 orchestrator subprocess-failure diagnostics (Phase α-2 / 2026-05-10).

Locks two behaviors:

1. **Fix-A** — `SignalExportRunner.validate_inputs` fail-loud on empty
   ``stage.checkpoint`` when stage enabled. Catches the original #PY-80
   bug (manifest omits checkpoint field → runner skips emitting
   ``--checkpoint`` to subprocess → argparse exit 2 in <100ms).

2. **Fix-B** — `_format_subprocess_failure` shared helper at
   ``stages/base.py``. Tails the last 20 stderr lines into the
   ``error_message`` displayed by ``cli.py`` so argparse errors,
   tracebacks, and missing-dependency messages stop being buried in
   ``result.stderr``. Per hft-rules §8 (never silently drop diagnostics).

Cited stress test (the original failure):
``$ python lob-model-trainer/scripts/export_signals.py --config /tmp/dummy.yaml --split test``
``export_signals.py: error: the following arguments are required: --checkpoint``
``$ echo $?``
``2``

Pre-Fix-B, ``cli.py`` showed only ``"export_signals.py exited with code 2"`` —
operators wasted hours debugging. Post-Fix-B, the actionable argparse error
is in the message tail.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hft_ops.stages.base import _format_subprocess_failure


# ---------------------------------------------------------------------------
# Fix-B helper unit tests (5)
# ---------------------------------------------------------------------------


def _mock_proc(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    """Build a CompletedProcess fixture for testing _format_subprocess_failure."""
    return subprocess.CompletedProcess(
        args=["python", "fake.py"],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


class TestFormatSubprocessFailureHelper:
    """Unit tests for the shared _format_subprocess_failure helper."""

    def test_empty_stderr_produces_generic_message(self):
        proc = _mock_proc(returncode=1, stderr="")
        msg = _format_subprocess_failure(proc, "train.py")
        assert msg == "train.py exited with code 1"

    def test_whitespace_only_stderr_treated_as_empty(self):
        proc = _mock_proc(returncode=2, stderr="\n  \n\t\n")
        msg = _format_subprocess_failure(proc, "x.py")
        assert msg == "x.py exited with code 2"

    def test_short_stderr_appended_in_full(self):
        proc = _mock_proc(returncode=2, stderr="line1\nline2\nerror line")
        msg = _format_subprocess_failure(proc, "export_signals.py")
        assert "exited with code 2" in msg
        assert "--- last 3 stderr lines ---" in msg
        assert "error line" in msg
        assert "line1" in msg

    def test_long_stderr_truncated_to_tail(self):
        long = "\n".join(f"line{i}" for i in range(100))
        proc = _mock_proc(returncode=2, stderr=long)
        msg = _format_subprocess_failure(proc, "x.py", stderr_tail_lines=5)
        # Last 5 lines (line95..line99) included
        assert "line99" in msg
        assert "line95" in msg
        # Earlier lines (line50, line0) should NOT be in tail
        assert "line50" not in msg
        assert "line0\n" not in msg

    def test_argparse_failure_signature_surfaces(self):
        # The actual signature of #PY-80's failure: argparse exits with
        # code 2 and a usage line + "the following arguments are required"
        argparse_stderr = (
            "usage: export_signals.py [-h] --config CONFIG --checkpoint CHECKPOINT\n"
            "                         [--split {val,test}] [--output-dir OUTPUT_DIR]\n"
            "                         [--calibrate {none,variance_match}]\n"
            "                         [--batch-size BATCH_SIZE]\n"
            "export_signals.py: error: the following arguments are required: --checkpoint"
        )
        proc = _mock_proc(returncode=2, stderr=argparse_stderr)
        msg = _format_subprocess_failure(proc, "export_signals.py")
        # Operator-actionable diagnostic now in error_message
        assert "exited with code 2" in msg
        assert "the following arguments are required: --checkpoint" in msg


# ---------------------------------------------------------------------------
# Fix-A regression tests (3) — validate_inputs catches empty checkpoint
# ---------------------------------------------------------------------------


class TestSignalExportValidateInputsEmptyCheckpoint:
    """Validate-time fail-loud on empty signal_export.checkpoint."""

    def _build_manifest_with_signal_export(self, tmp_pipeline: Path, *, checkpoint: str):
        """Helper: build a minimal manifest with given signal_export.checkpoint."""
        from hft_ops.manifest.schema import (
            ExperimentHeader,
            ExperimentManifest,
            ExtractionStage,
            SignalExportStage,
            Stages,
            TrainingStage,
        )
        return ExperimentManifest(
            experiment=ExperimentHeader(name="test_py80", contract_version="3.0"),
            pipeline_root="..",
            stages=Stages(
                extraction=ExtractionStage(enabled=False),
                training=TrainingStage(enabled=True),
                signal_export=SignalExportStage(
                    enabled=True,
                    script="lob-model-trainer/scripts/export_signals.py",
                    checkpoint=checkpoint,
                    split="test",
                    output_dir="outputs/experiments/test_py80/signals/test",
                ),
            ),
        )

    def test_empty_checkpoint_when_enabled_fails_validation(self, tmp_pipeline: Path):
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.signal_export import SignalExportRunner

        # Need to ensure trainer config resolution can succeed (other validators)
        # — we depend on training.enabled=True so Priority 2 (auto-persisted)
        # check passes regardless. The empty-checkpoint check should fire.
        manifest = self._build_manifest_with_signal_export(tmp_pipeline, checkpoint="")
        config = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_pipeline))

        # Need to write a fake script so the script-existence check passes
        script_dir = tmp_pipeline / "lob-model-trainer" / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "export_signals.py").write_text("# stub\n")

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        # The empty-checkpoint error MUST appear
        empty_checkpoint_errors = [e for e in errors if "checkpoint is empty" in e]
        assert len(empty_checkpoint_errors) == 1, (
            f"Expected exactly 1 'checkpoint is empty' error, got {len(empty_checkpoint_errors)}: {errors}"
        )
        # Error message must cite the convention to set
        assert "stages.signal_export.checkpoint" in empty_checkpoint_errors[0]
        assert "${stages.training.output_dir}" in empty_checkpoint_errors[0]

    def test_template_checkpoint_does_not_trigger_empty_check(self, tmp_pipeline: Path):
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.signal_export import SignalExportRunner

        manifest = self._build_manifest_with_signal_export(
            tmp_pipeline,
            checkpoint="${stages.training.output_dir}/checkpoints/best.pt",
        )
        config = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_pipeline))
        script_dir = tmp_pipeline / "lob-model-trainer" / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "export_signals.py").write_text("# stub\n")

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        # Template checkpoint is treated as valid (resolved at runtime).
        empty_checkpoint_errors = [e for e in errors if "checkpoint is empty" in e]
        assert len(empty_checkpoint_errors) == 0, (
            f"Template checkpoint should not trigger empty-check, got: {errors}"
        )

    def test_disabled_signal_export_does_not_require_checkpoint(self, tmp_pipeline: Path):
        from hft_ops.config import OpsConfig
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.signal_export import SignalExportRunner

        # Disabled stage skips the empty-checkpoint check (per `if stage.enabled`).
        manifest = self._build_manifest_with_signal_export(tmp_pipeline, checkpoint="")
        manifest.stages.signal_export.enabled = False
        config = OpsConfig(paths=PipelinePaths(pipeline_root=tmp_pipeline))

        runner = SignalExportRunner()
        errors = runner.validate_inputs(manifest, config)
        empty_checkpoint_errors = [e for e in errors if "checkpoint is empty" in e]
        assert len(empty_checkpoint_errors) == 0, (
            f"Disabled signal_export should not require checkpoint, got: {errors}"
        )
