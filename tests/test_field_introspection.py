"""Tests for ``manifest/_field_introspection.py`` — the stage-introspection SSoT.

Locks the contract that pipeline-stage names + per-stage known keys are
DERIVED from the ``Stages`` schema dataclass (never hand-maintained), resolved
via ``typing.get_type_hints`` (never ``dataclasses.Field.type`` — the
vacuous-test trap), and that the module stays torch-free (hft-ops invariant).

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 1.
"""

import ast
import dataclasses
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from hft_ops.manifest import _field_introspection as fi
from hft_ops.manifest.schema import (
    BacktestingStage,
    PostTrainingGateStage,
    Stages,
    TrainingStage,
)

# The 8 canonical pipeline-stage names. Hand-written ONCE here as the
# independent oracle the derivation is checked against (if someone hand-edits
# Stages, this oracle forces a conscious update — that is the test's job).
EXPECTED_STAGE_NAMES = frozenset(
    {
        "extraction",
        "raw_analysis",
        "dataset_analysis",
        "validation",
        "training",
        "post_training_gate",
        "signal_export",
        "backtesting",
    }
)


class TestStageNames:
    def test_stage_names_are_the_eight_canonical_stages(self):
        assert fi.stage_names() == EXPECTED_STAGE_NAMES

    def test_stage_names_derived_from_stages_fields(self):
        # The whole point of the SSoT: the name set is DERIVED from Stages, so
        # adding/removing a Stages field is reflected automatically (no drift).
        assert fi.stage_names() == frozenset(f.name for f in fields(Stages))

    def test_post_training_gate_is_present(self):
        # Regression guard for the 3 PTG hand-mirror omissions this SSoT fixes.
        assert "post_training_gate" in fi.stage_names()

    def test_stage_names_is_frozenset(self):
        assert isinstance(fi.stage_names(), frozenset)


class TestStageDataclass:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("training", TrainingStage),
            ("backtesting", BacktestingStage),
            ("post_training_gate", PostTrainingGateStage),
        ],
    )
    def test_resolves_to_the_real_dataclass(self, name, cls):
        assert fi.stage_dataclass(name) is cls

    def test_every_stage_resolves_to_a_dataclass(self):
        # Guards the keystone's own invariant: a future non-dataclass field on
        # Stages must fail loud, not silently corrupt known-key derivation.
        for name in fi.stage_names():
            assert dataclasses.is_dataclass(fi.stage_dataclass(name)), name

    def test_unknown_stage_raises(self):
        with pytest.raises(KeyError):
            fi.stage_dataclass("trainning")  # the classic typo


class TestKnownKeysForStage:
    def test_backtesting_known_keys_equal_dataclass_fields(self):
        # This SSoT retires loader._KNOWN_BACKTESTING_KEYS — they MUST be the
        # same 9-key set or the retirement (Step 5) changes behavior.
        assert fi.known_keys_for_stage("backtesting") == frozenset(
            f.name for f in fields(BacktestingStage)
        )

    def test_training_known_keys_include_specials(self):
        # H2 (Step 5) uses the FULL field set incl. specials like `overrides`
        # and `trainer_config`; only sweep-routing subtracts `overrides`.
        keys = fi.known_keys_for_stage("training")
        assert "overrides" in keys
        assert "trainer_config" in keys

    def test_known_keys_is_frozenset(self):
        assert isinstance(fi.known_keys_for_stage("training"), frozenset)

    def test_unknown_stage_raises(self):
        with pytest.raises(KeyError):
            fi.known_keys_for_stage("nope")


class TestTorchFreeInvariant:
    """hft-ops is torch-free by design (root CLAUDE.md §Module Technical Map);
    the SSoT must not pull torch/lobmodels. Mirrors
    test_contract_preflight.py:549 (AST) + :589 (sys.modules sentinel)."""

    def test_module_imports_are_torch_free_ast(self):
        src_path = Path(fi.__file__)
        assert src_path.exists(), f"Module source not found: {src_path}"
        tree = ast.parse(src_path.read_text())

        module_scope_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_scope_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_scope_imports.append(node.module or "")

        forbidden = ("torch", "lobmodels", "lob_models", "lobtrainer")
        offenders = [
            imp
            for imp in module_scope_imports
            if imp and any(imp.startswith(p) for p in forbidden)
        ]
        assert not offenders, (
            f"_field_introspection must stay torch-free (hft-ops invariant). "
            f"Offending module-scope imports: {offenders}"
        )

    def test_runtime_sys_modules_sentinel(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "import hft_ops.manifest._field_introspection; "
                "bad = [m for m in sys.modules if m.startswith"
                "(('torch', 'lobmodels', 'lobtrainer'))]; "
                "assert not bad, f'torch-free violation: {bad}'; "
                "print('TORCH_FREE_OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Runtime torch-free sentinel FAILED.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "TORCH_FREE_OK" in result.stdout
