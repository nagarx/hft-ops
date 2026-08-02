"""Output-contract enforcement — ``validate_outputs`` is CALLED, and failing
it stops a stage from reaching COMPLETED or being cached.

WHAT THIS LOCKS
---------------
Until 2026-08-03 ``validate_outputs`` was declared EIGHT times and called
ZERO times::

    grep -rn --no-ignore-files "validate_outputs" src/hft_ops/
      stages/base.py:88              (StageRunner protocol)
      stages/extraction.py:243
      stages/raw_analysis.py:119
      stages/dataset_analysis.py:166
      stages/validation.py:279
      stages/training.py:674
      stages/signal_export.py:695
      stages/backtesting.py:464
      -> 8 lines, every one a `def`. No call site existed.

Every stage declared the artifacts it promised to produce, and the
orchestrator never checked a single one. A subprocess that exited 0 without
writing anything produced a green COMPLETED stage, a *cached* empty export,
and a downstream stage failing far from the cause.

The tests below fail if that wiring is reverted:

- ``TestDriverEnforcesOutputContract`` drives the REAL ``hft-ops run`` CLI
  through ``CliRunner``. A fake runner reports COMPLETED while its
  ``validate_outputs`` returns a sentinel violation. If the driver stops
  calling ``enforce_output_contract``, ``validate_outputs`` is never invoked,
  the stage stays COMPLETED, and the assertions fail. This is the direct
  regression test for "declared 8 times, called 0 times".
- ``TestBothDriverLoopsWired`` parses ``cli.py`` and asserts BOTH stage-driver
  loops (``run`` and ``sweep_run``) call the helper — the sweep loop is the
  same defect on the grid-point path and needs a far heavier fixture to drive
  end to end.
- ``TestExtractionOutputContractGatesCache`` proves a violating extraction is
  never published into the content-addressed cache.
- ``TestExtractionValidateOutputsGlobs`` locks the recursive globs; the
  original non-recursive form returned three spurious violations for every
  valid export on disk.
"""

from __future__ import annotations

import ast
import textwrap
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

import hft_ops.cli as cli_mod
from hft_ops.cli import main
from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    ExtractionStage,
    Stages,
)
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    enforce_output_contract,
)
from hft_ops.stages.extraction import ExtractionRunner

SENTINEL = "SENTINEL_VIOLATION: the promised artifact was never written"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pipeline_root(tmp_path: Path) -> Path:
    """Minimal on-disk pipeline root that passes ``validate_manifest``."""
    for rel in (
        "feature-extractor-MBO-LOB/configs",
        "MBO-LOB-reconstructor",
        "data/exports",
        "hft-ops/ledger",
        "contracts",
        "lob-model-trainer",
        "lob-backtester",
    ):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    (tmp_path / "feature-extractor-MBO-LOB/configs/x.toml").write_text("")
    (tmp_path / "contracts/pipeline_contract.toml").write_text("")
    return tmp_path


def _write_manifest(root: Path, name: str) -> Path:
    manifest = root / f"{name}.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            experiment:
              name: "{name}"
              contract_version: "3.0"
            pipeline_root: "."
            stages:
              extraction:
                enabled: true
                config: "feature-extractor-MBO-LOB/configs/x.toml"
                output_dir: "data/exports/{name}"
              raw_analysis: {{enabled: false}}
              dataset_analysis: {{enabled: false}}
              validation: {{enabled: false}}
              training: {{enabled: false}}
              post_training_gate: {{enabled: false}}
              signal_export: {{enabled: false}}
              backtesting: {{enabled: false}}
            """
        ).strip()
    )
    return manifest


class _SpyRunner:
    """Reports COMPLETED; its output postcondition is caller-supplied."""

    stage_name = "extraction"

    def __init__(self, violations):
        self._violations = violations
        self.validate_outputs_calls = 0

    def validate_inputs(self, manifest, config):
        return []

    def run(self, manifest, config):
        return StageResult(
            stage_name=self.stage_name, status=StageStatus.COMPLETED
        )

    def validate_outputs(self, manifest, config):
        self.validate_outputs_calls += 1
        return list(self._violations)


def _invoke_run(monkeypatch, root: Path, manifest: Path, runner: _SpyRunner):
    monkeypatch.setattr(
        cli_mod,
        "_build_stage_runners",
        lambda manifest_obj: [("extraction", True, runner)],
    )
    return CliRunner().invoke(
        main, ["--pipeline-root", str(root), "run", str(manifest)]
    )


# ---------------------------------------------------------------------------
# THE lock: the real CLI driver calls validate_outputs and fails closed
# ---------------------------------------------------------------------------


class TestDriverEnforcesOutputContract:
    def test_violating_stage_never_reaches_completed(self, tmp_path, monkeypatch):
        """A stage that reports COMPLETED but violates its output contract is
        flipped to FAILED and aborts the pipeline.

        REVERT BEHAVIOUR: remove the ``enforce_output_contract`` call from the
        ``cli.run`` stage loop and ``validate_outputs`` is never invoked
        (``validate_outputs_calls == 0``), the stage prints ``completed``, and
        every assertion below fails.
        """
        root = _make_pipeline_root(tmp_path)
        manifest = _write_manifest(root, "ocp_violating")
        runner = _SpyRunner([SENTINEL])

        result = _invoke_run(monkeypatch, root, manifest, runner)

        assert runner.validate_outputs_calls == 1, (
            "The stage driver never called validate_outputs — the declared "
            "output postcondition is unenforced (the original defect: 8 "
            "definitions, 0 call sites)."
        )
        assert SENTINEL in result.output, (
            "The violation must be surfaced to the operator, not swallowed."
        )
        assert "Status: failed" in result.output, (
            "A violated output contract must flip the stage to FAILED; got:\n"
            f"{result.output}"
        )
        assert "Pipeline aborted at stage: extraction" in result.output, (
            "A violated output contract must abort the pipeline at the true "
            "point of failure, not let downstream stages consume the bad "
            "artifacts."
        )

    def test_satisfied_contract_still_completes(self, tmp_path, monkeypatch):
        """Control arm: an empty violation list leaves COMPLETED intact.

        Without this, the test above would also pass if enforcement simply
        failed every stage unconditionally.
        """
        root = _make_pipeline_root(tmp_path)
        manifest = _write_manifest(root, "ocp_satisfied")
        runner = _SpyRunner([])

        result = _invoke_run(monkeypatch, root, manifest, runner)

        assert runner.validate_outputs_calls == 1
        assert "Status: completed" in result.output, result.output
        assert "Pipeline aborted" not in result.output


class TestBothDriverLoopsWired:
    """``run`` and ``sweep run`` are two independent stage loops (see
    ``_build_stage_runners``'s docstring: "Keeping it in one place prevents
    the two call-sites from diverging"). The CLI test above covers ``run``;
    driving a sweep grid point end-to-end needs a much heavier fixture, so
    the sweep loop is locked structurally.
    """

    @staticmethod
    def _calls_in(func_name: str) -> set:
        source = Path(cli_mod.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
        raise AssertionError(f"cli.py has no function named {func_name!r}")

    @pytest.mark.parametrize("func_name", ["run", "sweep_run"])
    def test_stage_loop_enforces_output_contract(self, func_name):
        assert "enforce_output_contract" in self._calls_in(func_name), (
            f"cli.{func_name} runs stages but never calls "
            f"enforce_output_contract — stages in that loop can reach "
            f"COMPLETED without producing their declared artifacts."
        )


# ---------------------------------------------------------------------------
# enforce_output_contract semantics
# ---------------------------------------------------------------------------


class TestEnforceOutputContractSemantics:
    class _Runner:
        def __init__(self, outcome):
            self._outcome = outcome

        def validate_outputs(self, manifest, config):
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return self._outcome

    @staticmethod
    def _result(status, message=""):
        return StageResult(
            stage_name="s", status=status, error_message=message
        )

    def test_violations_flip_completed_to_failed(self):
        res = self._result(StageStatus.COMPLETED)
        enforce_output_contract(self._Runner(["a", "b"]), None, None, res)
        assert res.status is StageStatus.FAILED
        assert "a" in res.error_message and "b" in res.error_message
        assert res.captured_metrics["output_validation_errors"] == ["a", "b"]

    def test_no_violations_preserves_completed(self):
        res = self._result(StageStatus.COMPLETED)
        enforce_output_contract(self._Runner([]), None, None, res)
        assert res.status is StageStatus.COMPLETED
        assert res.error_message == ""
        assert "output_validation_errors" not in res.captured_metrics

    @pytest.mark.parametrize(
        "status", [StageStatus.SKIPPED, StageStatus.PENDING]
    )
    def test_non_completed_stage_is_not_validated(self, status):
        """A stage that did not run has no postcondition to meet — a cache
        hit / skip_if_exists / dry run must not be failed by it."""
        res = self._result(status)
        enforce_output_contract(self._Runner([SENTINEL]), None, None, res)
        assert res.status is status
        assert res.error_message == ""

    def test_already_failed_stage_keeps_its_original_error(self):
        res = self._result(StageStatus.FAILED, "cargo exited with code 101")
        enforce_output_contract(self._Runner([SENTINEL]), None, None, res)
        assert res.status is StageStatus.FAILED
        assert res.error_message == "cargo exited with code 101"

    def test_runner_without_validate_outputs_is_tolerated(self):
        """``PostTrainingGateRunner`` declares no ``validate_outputs``; the
        helper must skip it rather than raise AttributeError and take down
        the whole orchestrator run."""
        res = self._result(StageStatus.COMPLETED)
        enforce_output_contract(types.SimpleNamespace(), None, None, res)
        assert res.status is StageStatus.COMPLETED

    def test_raising_validate_outputs_fails_closed(self):
        """A broken postcondition is a hard error (hft-rules §8), never a
        silent pass."""
        res = self._result(StageStatus.COMPLETED)
        enforce_output_contract(
            self._Runner(RuntimeError("boom")), None, None, res
        )
        assert res.status is StageStatus.FAILED
        assert "RuntimeError" in res.error_message
        assert "boom" in res.error_message


# ---------------------------------------------------------------------------
# Cache publication gate
# ---------------------------------------------------------------------------


class TestExtractionOutputContractGatesCache:
    """Extraction enforces its own contract INSIDE ``run()``, before the
    cache-populate block — the driver-level check happens after ``run()``
    returns, which is too late to stop a bad export being published into the
    content-addressed cache and silently re-linked into every future run
    sharing that cache key.
    """

    @staticmethod
    def _manifest(name: str) -> ExperimentManifest:
        return ExperimentManifest(
            experiment=ExperimentHeader(name=name),
            stages=Stages(
                extraction=ExtractionStage(
                    config="feature-extractor-MBO-LOB/configs/x.toml",
                    output_dir=f"data/exports/{name}",
                ),
            ),
        )

    def _stub_cache(self, monkeypatch, populated: list):
        monkeypatch.setattr(
            "hft_ops.stages.extraction.prepare_cache_key_inputs",
            lambda **kw: types.SimpleNamespace(),
        )
        monkeypatch.setattr(
            "hft_ops.stages.extraction.compute_cache_key", lambda inputs: "k" * 64
        )
        monkeypatch.setattr(
            "hft_ops.stages.extraction.resolve_or_link",
            lambda key, out, root: types.SimpleNamespace(
                status="miss", seconds_saved=0.0, linked_files=0, link_type=""
            ),
        )
        monkeypatch.setattr(
            "hft_ops.stages.extraction.populate",
            lambda *a, **kw: populated.append(a),
        )

    def test_violating_extraction_is_not_cached(self, tmp_path, monkeypatch):
        """Extractor exits 0 but writes nothing → FAILED, and ``populate`` is
        never reached.

        REVERT BEHAVIOUR: drop the ``enforce_output_contract`` call from
        ``ExtractionRunner.run`` and the stage stays COMPLETED, so the
        populate block's ``status == COMPLETED`` guard opens and an empty
        export is published to the cache.
        """
        populated: list = []
        self._stub_cache(monkeypatch, populated)
        # Exit 0, write NOTHING — the exact silent-failure shape.
        monkeypatch.setattr(
            "hft_ops.stages.extraction.run_subprocess",
            lambda cmd, cwd=None, verbose=False, env=None: types.SimpleNamespace(
                returncode=0, stdout="ok", stderr=""
            ),
        )
        name = "ocp_cache_violating"
        (tmp_path / "data" / "exports" / name).mkdir(parents=True)

        config = OpsConfig.from_pipeline_root(pipeline_root=tmp_path)
        result = ExtractionRunner().run(self._manifest(name), config)

        assert result.status is StageStatus.FAILED, (
            "An extractor that exits 0 without writing artifacts must not "
            "report COMPLETED."
        )
        assert populated == [], (
            "A violating extraction was published into the content-addressed "
            "cache — it would be silently re-linked into every future run "
            "with the same cache key."
        )
        assert result.captured_metrics["output_validation_errors"]

    def test_valid_extraction_is_still_cached(self, tmp_path, monkeypatch):
        """Control arm: a well-formed export still populates the cache, so the
        test above cannot pass by disabling caching outright."""
        populated: list = []
        self._stub_cache(monkeypatch, populated)
        name = "ocp_cache_valid"
        export_dir = tmp_path / "data" / "exports" / name

        def _writing_extractor(cmd, cwd=None, verbose=False, env=None):
            split = export_dir / "train"
            split.mkdir(parents=True, exist_ok=True)
            (split / "20250203_metadata.json").write_text("{}")
            (split / "20250203_sequences.npy").write_bytes(b"")
            (split / "20250203_regression_labels.npy").write_bytes(b"")
            return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(
            "hft_ops.stages.extraction.run_subprocess", _writing_extractor
        )
        export_dir.mkdir(parents=True)

        config = OpsConfig.from_pipeline_root(pipeline_root=tmp_path)
        result = ExtractionRunner().run(self._manifest(name), config)

        assert result.status is StageStatus.COMPLETED, result.error_message
        assert len(populated) == 1, "A valid export must still be cached."


# ---------------------------------------------------------------------------
# The recursive-glob fix
# ---------------------------------------------------------------------------


class TestExtractionValidateOutputsGlobs:
    """An export is ``<dir>/{train,val,test}/<day>_*.{npy,json}`` — the per-day
    artifacts are one level DOWN. The original non-recursive globs measured
    0 / 0 / 0 against ``data/exports/e5_timebased_60s_v3p0`` while the
    recursive form measured 230 / 230 / 230, so failing closed on the old form
    would have broken every extraction run.
    """

    @staticmethod
    def _config_and_manifest(tmp_path: Path, name: str):
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name=name),
            stages=Stages(
                extraction=ExtractionStage(
                    config="feature-extractor-MBO-LOB/configs/x.toml",
                    output_dir=f"data/exports/{name}",
                ),
            ),
        )
        return OpsConfig.from_pipeline_root(pipeline_root=tmp_path), manifest

    def test_split_subdir_regression_export_passes(self, tmp_path):
        """The real on-disk shape: artifacts under ``train/``, and a REGRESSION
        export that emits ``_regression_labels.npy`` with no ``_labels.npy``.

        REVERT BEHAVIOUR: restore the non-recursive globs and this returns
        three violations for a perfectly valid export.
        """
        name = "ocp_glob_ok"
        split = tmp_path / "data" / "exports" / name / "train"
        split.mkdir(parents=True)
        (split / "20250203_metadata.json").write_text("{}")
        (split / "20250203_sequences.npy").write_bytes(b"")
        (split / "20250203_regression_labels.npy").write_bytes(b"")

        config, manifest = self._config_and_manifest(tmp_path, name)
        assert ExtractionRunner().validate_outputs(manifest, config) == []

    def test_empty_export_dir_is_rejected(self, tmp_path):
        """Control arm: the check must still catch a genuinely empty export,
        so the recursive fix cannot pass by never failing.
        """
        name = "ocp_glob_empty"
        (tmp_path / "data" / "exports" / name).mkdir(parents=True)

        config, manifest = self._config_and_manifest(tmp_path, name)
        errors = ExtractionRunner().validate_outputs(manifest, config)
        assert len(errors) == 3, errors

    def test_missing_export_dir_is_rejected(self, tmp_path):
        name = "ocp_glob_absent"
        config, manifest = self._config_and_manifest(tmp_path, name)
        errors = ExtractionRunner().validate_outputs(manifest, config)
        assert len(errors) == 1
        assert "not found" in errors[0]
