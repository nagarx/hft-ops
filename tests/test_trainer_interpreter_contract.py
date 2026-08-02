"""The trainer interpreter is an EXPLICIT, CONFIGURABLE, PREFLIGHTED contract.

WHAT THIS LOCKS
---------------
``TrainingRunner.run`` built its command as ``[sys.executable, train.py, ...]``
— the interpreter running hft-ops itself. Measured 2026-08-03::

    hft-ops/.venv/bin/python  lob-model-trainer/scripts/train.py --help
        -> EXIT 1: ModuleNotFoundError: No module named 'pydantic'
    lob-model-trainer/.venv/bin/python  <same script>
        -> EXIT 0

hft-ops is deliberately torch-free (locked by ``test_monitor_torch_free.py``
and the contract-preflight AST test), so its interpreter cannot import
``lobtrainer``. ``hft-ops run`` therefore could not train AT ALL — while root
``CLAUDE.md`` and hft-rules §13 both mandate the orchestrator for any published
R², because it is the path that emits and persists the zero-skill floor. The
failure was quiet: a subprocess died on an import after the whole pipeline's
setup had been paid for.

The fix does NOT install trainer dependencies into the hft-ops venv (that
couples two deliberately-separate dependency sets and rots again silently).
It makes the interpreter an explicit contract with a preflight, converting a
silent 189-record-era rot into a loud startup error.

Each test states the revert behaviour it detects.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

import hft_ops.stages.contract_preflight as contract_preflight_mod
import hft_ops.stages.signal_export as signal_export_mod
import hft_ops.stages.training as training_mod
from hft_ops.cli import main
from hft_ops.config import OpsConfig
from hft_ops.interpreters import (
    TRAINER_PROBE_MODULE,
    TRAINER_PYTHON_ENV_VAR,
    InterpreterPreflightError,
    preflight_interpreter,
    resolve_trainer_python,
)
from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    SignalExportStage,
    Stages,
    TrainingStage,
)
from hft_ops.paths import PipelinePaths
from hft_ops.stages.base import StageStatus
from hft_ops.stages.signal_export import SignalExportRunner
from hft_ops.stages.training import TrainingRunner

SENTINEL_PYTHON = Path("/sentinel/interpreter/python")

#: An import that no interpreter on earth satisfies — keeps the negative
#: preflight tests deterministic instead of depending on machine venv state.
IMPOSSIBLE_MODULE = "hft_ops_no_such_module_9d3f1a"


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


class TestResolutionPrecedence:
    @staticmethod
    def _paths(tmp_path: Path, *, with_venv: bool) -> PipelinePaths:
        paths = PipelinePaths(pipeline_root=tmp_path)
        if with_venv:
            venv_python = paths.trainer_dir / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("#!/bin/sh\n")
        return paths

    def test_configured_beats_everything(self, tmp_path):
        paths = self._paths(tmp_path, with_venv=True)
        got = resolve_trainer_python(
            paths,
            configured="/cfg/python",
            env={TRAINER_PYTHON_ENV_VAR: "/env/python"},
        )
        assert got == Path("/cfg/python")

    def test_env_var_beats_repo_venv(self, tmp_path):
        paths = self._paths(tmp_path, with_venv=True)
        got = resolve_trainer_python(
            paths, env={TRAINER_PYTHON_ENV_VAR: "/env/python"}
        )
        assert got == Path("/env/python")

    def test_repo_venv_used_when_present(self, tmp_path):
        paths = self._paths(tmp_path, with_venv=True)
        got = resolve_trainer_python(paths, env={})
        assert got == paths.trainer_dir / ".venv" / "bin" / "python"

    def test_falls_back_to_sys_executable(self, tmp_path):
        """Back-compat: a single-venv install (CI / monorepo-wide venv) keeps
        the pre-contract behaviour."""
        paths = self._paths(tmp_path, with_venv=False)
        assert resolve_trainer_python(paths, env={}) == Path(sys.executable)


# ---------------------------------------------------------------------------
# The preflight itself
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_passes_when_interpreter_can_import(self):
        """Control arm: a satisfiable import must NOT raise, so the negative
        tests cannot pass by failing unconditionally."""
        preflight_interpreter(
            Path(sys.executable), "json", stage_name="training"
        )

    def test_raises_when_interpreter_cannot_import(self):
        with pytest.raises(InterpreterPreflightError) as exc:
            preflight_interpreter(
                Path(sys.executable), IMPOSSIBLE_MODULE, stage_name="training"
            )
        message = str(exc.value)
        assert sys.executable in message, "must name the interpreter it tried"
        assert IMPOSSIBLE_MODULE in message, "must name the failing import"
        assert "ModuleNotFoundError" in message, "must surface the real error"
        assert TRAINER_PYTHON_ENV_VAR in message, "must name the env override"
        assert "--trainer-python" in message, "must name the CLI override"

    def test_raises_when_interpreter_does_not_exist(self, tmp_path):
        with pytest.raises(InterpreterPreflightError) as exc:
            preflight_interpreter(
                tmp_path / "not-a-python", "json", stage_name="training"
            )
        assert "does not exist or is not executable" in str(exc.value)


# ---------------------------------------------------------------------------
# Stage wiring
# ---------------------------------------------------------------------------


def _trainer_root(tmp_path: Path) -> Path:
    (tmp_path / "lob-model-trainer" / "configs").mkdir(parents=True)
    (tmp_path / "lob-model-trainer" / "scripts").mkdir(parents=True)
    (tmp_path / "lob-model-trainer" / "configs" / "c.yaml").write_text("name: t\n")
    (tmp_path / "lob-model-trainer" / "scripts" / "train.py").write_text("")
    (tmp_path / "lob-model-trainer" / "scripts" / "export_signals.py").write_text("")
    (tmp_path / "hft-ops" / "ledger" / "runs").mkdir(parents=True)
    return tmp_path


class TestTrainingStageUsesTheContract:
    @staticmethod
    def _manifest() -> ExperimentManifest:
        return ExperimentManifest(
            experiment=ExperimentHeader(name="interp_probe"),
            stages=Stages(
                training=TrainingStage(config="lob-model-trainer/configs/c.yaml")
            ),
        )

    def test_launches_with_resolved_trainer_python_not_sys_executable(
        self, tmp_path, monkeypatch
    ):
        """``cmd[0]`` must be the RESOLVED trainer interpreter.

        REVERT BEHAVIOUR: restore ``sys.executable`` in the cmd list and
        ``cmd[0]`` becomes the hft-ops interpreter — which cannot import
        lobtrainer — so both assertions fail.
        """
        root = _trainer_root(tmp_path)
        captured = {}

        monkeypatch.setattr(
            contract_preflight_mod, "preflight_trainer_config",
            lambda path, paths=None: None,
        )
        monkeypatch.setattr(
            training_mod, "resolve_and_preflight_trainer_python",
            lambda paths, configured=None, stage_name="": SENTINEL_PYTHON,
        )

        def _capture(cmd, cwd=None, verbose=False, env=None):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(training_mod, "run_subprocess", _capture)

        result = TrainingRunner().run(
            self._manifest(), OpsConfig.from_pipeline_root(pipeline_root=root)
        )

        assert result.status is StageStatus.COMPLETED, result.error_message
        assert captured["cmd"][0] == str(SENTINEL_PYTHON), (
            "The training stage must launch train.py with the resolved trainer "
            "interpreter, not with sys.executable."
        )
        assert captured["cmd"][0] != sys.executable
        assert result.captured_metrics["trainer_python"] == str(SENTINEL_PYTHON)

    def test_preflight_failure_aborts_before_any_subprocess(
        self, tmp_path, monkeypatch
    ):
        """A bad interpreter fails the stage LOUDLY and never shells out.

        REVERT BEHAVIOUR: delete the preflight and the stage happily builds a
        command around an interpreter that cannot import the trainer, so
        ``run_subprocess`` IS called and the stage reports whatever the doomed
        subprocess returned instead of an actionable message.
        """
        root = _trainer_root(tmp_path)
        shelled_out = []

        monkeypatch.setattr(
            contract_preflight_mod, "preflight_trainer_config",
            lambda path, paths=None: None,
        )
        monkeypatch.setattr(
            training_mod, "run_subprocess",
            lambda *a, **kw: shelled_out.append(a)
            or types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        )
        # Real resolution + real preflight, pointed at an interpreter that
        # provably cannot import the trainer.
        monkeypatch.setenv(TRAINER_PYTHON_ENV_VAR, str(tmp_path / "not-a-python"))

        result = TrainingRunner().run(
            self._manifest(), OpsConfig.from_pipeline_root(pipeline_root=root)
        )

        assert result.status is StageStatus.FAILED
        assert shelled_out == [], (
            "The interpreter preflight must run BEFORE the stage starts — no "
            "subprocess should have been launched."
        )
        assert TRAINER_PROBE_MODULE in result.error_message
        assert TRAINER_PYTHON_ENV_VAR in result.error_message
        assert (
            result.captured_metrics["gate_report"]["reason"]
            == "trainer_interpreter_preflight"
        )


class TestSignalExportStageUsesTheContract:
    def test_launches_with_resolved_trainer_python(self, tmp_path, monkeypatch):
        """``export_signals.py`` is a lob-model-trainer script and had the
        identical ``sys.executable`` defect.

        REVERT BEHAVIOUR: restore ``sys.executable`` and ``cmd[0]`` stops
        matching the resolved interpreter.
        """
        root = _trainer_root(tmp_path)
        captured = {}

        monkeypatch.setattr(
            signal_export_mod, "resolve_and_preflight_trainer_python",
            lambda paths, configured=None, stage_name="": SENTINEL_PYTHON,
        )

        def _capture(cmd, cwd=None, verbose=False, env=None):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(signal_export_mod, "run_subprocess", _capture)

        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="interp_probe_se"),
            stages=Stages(
                signal_export=SignalExportStage(
                    config="lob-model-trainer/configs/c.yaml",
                    checkpoint="lob-model-trainer/configs/c.yaml",
                ),
            ),
        )
        SignalExportRunner().run(
            manifest, OpsConfig.from_pipeline_root(pipeline_root=root)
        )

        assert captured.get("cmd"), "signal_export never reached the subprocess"
        assert captured["cmd"][0] == str(SENTINEL_PYTHON)
        assert captured["cmd"][0] != sys.executable


# ---------------------------------------------------------------------------
# The knob must be REACHABLE (hft-rules §5)
# ---------------------------------------------------------------------------


class TestTrainerPythonKnobIsReachable:
    """A constructor default that no config path threads is an unreachable
    knob, not a default. ``EarlyStopping.min_delta`` (FINDING-136) was exactly
    that for every experiment in the ledger — this locks the same trap shut
    for ``trainer_python``.
    """

    def test_cli_exposes_trainer_python(self):
        opts = {
            opt
            for param in main.params
            for opt in getattr(param, "opts", [])
        }
        assert "--trainer-python" in opts, (
            "OpsConfig.trainer_python is unreachable from the CLI — an "
            "unreachable knob, not a default."
        )

    @pytest.mark.parametrize("subcommand", ["run", "sweep"])
    def test_cli_value_reaches_ops_config(self, tmp_path, monkeypatch, subcommand):
        """The flag must actually land in ``OpsConfig.trainer_python`` at BOTH
        driver construction sites (``run`` and ``sweep run``), not merely exist
        on the parser."""
        import hft_ops.cli as cli_mod
        from click.testing import CliRunner

        seen = {}
        real_ops_config = cli_mod.OpsConfig

        def _spy(*args, **kwargs):
            seen["trainer_python"] = kwargs.get("trainer_python")
            return real_ops_config(*args, **kwargs)

        monkeypatch.setattr(cli_mod, "OpsConfig", _spy)
        # The manifest must EXIST (click.Path(exists=True)) but need not be
        # valid — the command aborts at load/validate, which is AFTER the
        # OpsConfig construction under test.
        manifest = tmp_path / "not_a_real_manifest.yaml"
        manifest.write_text("experiment: {name: x}\n")
        args = ["--trainer-python", "/spy/python"]
        args += ["run", str(manifest)] if subcommand == "run" else [
            "sweep", "run", str(manifest)
        ]

        CliRunner().invoke(main, args)

        assert seen.get("trainer_python") == "/spy/python", (
            f"--trainer-python did not reach OpsConfig in `{subcommand}`; the "
            f"CLI flag and the config field are disconnected."
        )


# ---------------------------------------------------------------------------
# Real-environment integration — the original measurement
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINER_VENV = _REPO_ROOT / "lob-model-trainer" / ".venv" / "bin" / "python"
_TRAIN_SCRIPT = _REPO_ROOT / "lob-model-trainer" / "scripts" / "train.py"


@pytest.mark.skipif(
    not (_TRAINER_VENV.exists() and _TRAIN_SCRIPT.exists()),
    reason="lob-model-trainer venv or train.py not present in this checkout",
)
def test_resolved_interpreter_can_actually_run_train_py():
    """End-to-end against the real checkout: the interpreter this contract
    resolves must be able to run ``train.py``.

    This is the original defect measurement, inverted into an assertion —
    ``hft-ops/.venv/bin/python train.py --help`` exited 1 on
    ``ModuleNotFoundError: pydantic`` while the trainer venv exited 0.
    """
    resolved = resolve_trainer_python(
        PipelinePaths(pipeline_root=_REPO_ROOT), env={}
    )
    assert resolved == _TRAINER_VENV

    proc = subprocess.run(
        [str(resolved), str(_TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"The resolved trainer interpreter cannot run train.py:\n"
        f"{proc.stderr[-2000:]}"
    )
