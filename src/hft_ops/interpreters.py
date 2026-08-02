"""Explicit, configurable, validated interpreter contracts for cross-repo
subprocess stages.

THE DEFECT THIS CLOSES
----------------------
Every stage runner launched its module's script with ``sys.executable`` — the
interpreter running *hft-ops itself*. That silently assumes one interpreter can
import every sibling module's dependency set. It cannot, and on this machine it
did not (measured 2026-08-03)::

    hft-ops/.venv/bin/python           -> Python 3.14.2, no pydantic, no torch
    hft-ops/.venv/bin/python lob-model-trainer/scripts/train.py --help
        -> EXIT 1: ModuleNotFoundError: No module named 'pydantic'
    lob-model-trainer/.venv/bin/python lob-model-trainer/scripts/train.py --help
        -> EXIT 0

So ``hft-ops run`` could not train at all — while root ``CLAUDE.md`` and
``.claude/rules/hft-rules.md`` §13 both mandate the orchestrator for any
published R² (it is the path that emits and persists the zero-skill floor).
The failure mode was quiet: the training stage shelled out, the subprocess died
on an import, and the operator saw a generic non-zero exit code from a stage
that had already burned the whole pipeline's setup.

WHY NOT JUST ``pip install`` INTO THE hft-ops VENV
--------------------------------------------------
Because it couples two dependency sets that are deliberately separate — the
orchestrator is intentionally torch-free (locked by
``tests/test_monitor_torch_free.py`` and the contract-preflight AST test) — and
because it rots silently again the moment either side adds a dependency. The
durable fix is to make the interpreter an EXPLICIT contract with a PREFLIGHT,
so a mismatch is a loud startup error naming the interpreter and the missing
module instead of an opaque subprocess exit 189 records later.

RESOLUTION ORDER (highest first)
--------------------------------
1. ``OpsConfig.trainer_python`` — set by ``hft-ops --trainer-python <path>``.
2. ``$HFT_OPS_TRAINER_PYTHON`` — persists across sessions and into CI.
3. ``<trainer_dir>/.venv/bin/python`` — the conventional per-repo venv.
4. ``sys.executable`` — last-resort back-compat, so a single-venv install (CI,
   a monorepo-wide venv) keeps working exactly as before.

Whatever wins, the preflight then PROVES it can import the module before the
stage starts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from hft_ops.paths import PipelinePaths

#: Operator-facing env override for the trainer interpreter. Named on every
#: preflight failure message so the fix is discoverable from the error alone.
TRAINER_PYTHON_ENV_VAR = "HFT_OPS_TRAINER_PYTHON"

#: Import probe for the trainer scripts (``train.py`` / ``export_signals.py``).
#: ``lobtrainer`` transitively pulls pydantic and torch, so a successful import
#: is a sufficient proxy for "this interpreter can run the trainer".
TRAINER_PROBE_MODULE = "lobtrainer"

#: Seconds allowed for the import probe. Generous — importing ``lobtrainer``
#: pulls torch, which is slow on a cold filesystem cache — but bounded so a
#: wedged interpreter cannot hang the orchestrator forever.
_PREFLIGHT_TIMEOUT_SECONDS = 120


class InterpreterPreflightError(RuntimeError):
    """An interpreter cannot import the module its stage needs.

    Raised BEFORE the stage does any work, so the operator sees the real cause
    (wrong interpreter) rather than a downstream subprocess exit code.
    """


def resolve_trainer_python(
    paths: PipelinePaths,
    *,
    configured: Optional[str] = None,
    env: Optional[dict] = None,
) -> Path:
    """Resolve the interpreter that runs ``lob-model-trainer`` scripts.

    Args:
        paths: Resolved pipeline paths (supplies ``trainer_dir``).
        configured: ``OpsConfig.trainer_python`` — the CLI-supplied override.
        env: Environment mapping to read ``HFT_OPS_TRAINER_PYTHON`` from.
            Defaults to ``os.environ``. Injectable for testing.

    Returns:
        Absolute path to the interpreter. Existence is NOT asserted here —
        ``preflight_interpreter`` reports a missing or unusable interpreter
        with a far more actionable message than a bare ``FileNotFoundError``.
    """
    environ = os.environ if env is None else env

    if configured:
        return Path(configured).expanduser()

    from_env = environ.get(TRAINER_PYTHON_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()

    venv_python = paths.trainer_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python

    # Back-compat: a single-venv install (CI, monorepo-wide venv) behaves
    # exactly as it did before this contract existed.
    return Path(sys.executable)


@lru_cache(maxsize=32)
def _probe(python: str, module: str) -> tuple:
    """Run the import probe once per (interpreter, module) per process.

    A sweep runs the training stage once per grid point; without the cache
    that is one subprocess launch per point for an answer that cannot change
    mid-run.

    Returns:
        ``(ok: bool, detail: str)`` — ``detail`` is the stderr tail on failure.
    """
    try:
        proc = subprocess.run(
            [python, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return (False, "interpreter does not exist or is not executable")
    except PermissionError as exc:
        return (False, f"interpreter is not executable: {exc}")
    except subprocess.TimeoutExpired:
        return (
            False,
            f"import probe timed out after {_PREFLIGHT_TIMEOUT_SECONDS}s",
        )
    if proc.returncode == 0:
        return (True, "")
    stderr = (proc.stderr or "").strip().splitlines()
    return (False, stderr[-1] if stderr else f"exit code {proc.returncode}")


def preflight_interpreter(
    python: Path,
    module: str,
    *,
    stage_name: str,
    env_var: str = TRAINER_PYTHON_ENV_VAR,
) -> None:
    """Prove ``python`` can ``import module``, or fail loudly.

    Args:
        python: Interpreter resolved by :func:`resolve_trainer_python`.
        module: Import probe (e.g. ``"lobtrainer"``).
        stage_name: Stage being guarded — named in the error.
        env_var: The env override to name in the remediation hint.

    Raises:
        InterpreterPreflightError: With the interpreter path, the module that
            failed to import, the underlying error, and every way to fix it.
    """
    ok, detail = _probe(str(python), module)
    if ok:
        return
    raise InterpreterPreflightError(
        f"Stage '{stage_name}' cannot run: the configured interpreter\n"
        f"    {python}\n"
        f"cannot 'import {module}' — {detail}\n"
        f"\n"
        f"hft-ops intentionally does NOT share a dependency set with "
        f"lob-model-trainer (the orchestrator is kept torch-free), so it must "
        f"launch trainer scripts with the trainer's own interpreter.\n"
        f"Point it at one, highest precedence first:\n"
        f"  1. hft-ops --trainer-python /path/to/python run <manifest>\n"
        f"  2. export {env_var}=/path/to/python\n"
        f"  3. create the conventional venv at "
        f"<lob-model-trainer>/.venv/bin/python\n"
        f"Verify your choice with:  <python> -c 'import {module}'"
    )


def resolve_and_preflight_trainer_python(
    paths: PipelinePaths,
    *,
    configured: Optional[str] = None,
    stage_name: str,
) -> Path:
    """Resolve the trainer interpreter and prove it works. Convenience wrapper
    for the two stages that launch ``lob-model-trainer`` scripts.

    Raises:
        InterpreterPreflightError: If the resolved interpreter cannot import
            :data:`TRAINER_PROBE_MODULE`.
    """
    python = resolve_trainer_python(paths, configured=configured)
    preflight_interpreter(
        python, TRAINER_PROBE_MODULE, stage_name=stage_name
    )
    return python
