"""Phase V.A.8 MVP: InputContract pre-flight unit tests.

Covers the pure-function `validate_input_contract` + the YAML-loading
`preflight_trainer_config` convenience wrapper.

Test matrix (8 cases):
  T1: pass path — tlob with valid (F=98, T=100) → no raise.
  T2: non-positive feature_count → ValueError.
  T3: non-positive window_size → ValueError.
  T4: deeplob with feature_count=128 (above max_features=98) → ValueError.
  T5: deeplob with feature_count=20 (below min_features=40) → ValueError.
  T6: deeplob with window_size=10 (below min_sequence_length=20) → ValueError.
  T7: unknown model_type → WARN but no raise (observation tier).
  T8: preflight_trainer_config happy path — valid YAML loads + passes.
  T9: preflight_trainer_config missing model.model_type → ValueError.
  T10: preflight_trainer_config missing data.feature_count → ValueError.
  T11: preflight_trainer_config window_size defaults to 100 when absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from hft_ops.stages.contract_preflight import (
    _INPUT_CONTRACTS,
    preflight_trainer_config,
    validate_input_contract,
)


# =============================================================================
# validate_input_contract — pure function tests
# =============================================================================


class TestValidateInputContractPassPath:
    """T1: canonical-passing manifests."""

    def test_tlob_canonical_passes(self):
        # E5 canonical: TLOB + 98 features + T=20 (above min_sequence_length=4)
        validate_input_contract("tlob", feature_count=98, window_size=20)

    def test_deeplob_canonical_passes(self):
        # DeepLOB paper-canonical: F=40, T=100 — at the constraint boundary.
        validate_input_contract("deeplob", feature_count=40, window_size=100)

    def test_deeplob_at_upper_bound_passes(self):
        validate_input_contract("deeplob", feature_count=98, window_size=20)

    def test_permissive_model_passes_minimal(self):
        # hmhp / temporal_ridge / lstm all have min_features=1, min_seq=1.
        validate_input_contract("hmhp", feature_count=1, window_size=1)
        validate_input_contract("temporal_ridge", feature_count=34, window_size=1)


class TestValidateInputContractFailures:
    """T2-T6: misconfigurations that must fail-loud."""

    def test_zero_features_raises(self):
        with pytest.raises(ValueError, match=r"feature_count=0 must be positive"):
            validate_input_contract("tlob", feature_count=0, window_size=20)

    def test_negative_features_raises(self):
        with pytest.raises(ValueError, match=r"feature_count=-5 must be positive"):
            validate_input_contract("tlob", feature_count=-5, window_size=20)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match=r"window_size=0 must be positive"):
            validate_input_contract("tlob", feature_count=98, window_size=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match=r"window_size=-1 must be positive"):
            validate_input_contract("tlob", feature_count=98, window_size=-1)

    def test_deeplob_too_many_features_raises(self):
        """DeepLOB max_features=98; 128 should fail."""
        with pytest.raises(ValueError, match=r"max_features=98.*feature_count=128"):
            validate_input_contract("deeplob", feature_count=128, window_size=100)

    def test_deeplob_too_few_features_raises(self):
        """DeepLOB min_features=40; 20 should fail."""
        with pytest.raises(ValueError, match=r"min_features=40.*feature_count=20"):
            validate_input_contract("deeplob", feature_count=20, window_size=100)

    def test_deeplob_too_short_sequence_raises(self):
        """DeepLOB min_sequence_length=20; 10 should fail."""
        with pytest.raises(
            ValueError, match=r"min_sequence_length=20.*window_size=10"
        ):
            validate_input_contract("deeplob", feature_count=40, window_size=10)

    def test_tlob_too_short_sequence_raises(self):
        """TLOB min_sequence_length=4; 1 should fail."""
        with pytest.raises(
            ValueError, match=r"min_sequence_length=4.*window_size=1"
        ):
            validate_input_contract("tlob", feature_count=98, window_size=1)

    def test_mlplob_too_short_sequence_raises(self):
        """MLPLOB min_sequence_length=4; 1 should fail."""
        with pytest.raises(
            ValueError, match=r"min_sequence_length=4.*window_size=2"
        ):
            validate_input_contract("mlplob", feature_count=98, window_size=2)


class TestValidateInputContractUnknownModel:
    """T7: unknown model_type → WARN but don't raise (observation tier)."""

    def test_unknown_model_does_not_raise(self, caplog):
        # Returns normally — trainer will surface its own error.
        with caplog.at_level(logging.WARNING):
            validate_input_contract("some_future_model_xyz", feature_count=128, window_size=50)
        # Expect WARNING level message about unknown model_type
        assert any(
            "unknown model_type" in rec.getMessage() and "some_future_model_xyz" in rec.getMessage()
            for rec in caplog.records
        ), f"Expected WARNING log about unknown model_type. Got records: {[r.getMessage() for r in caplog.records]}"

    def test_unknown_model_still_checks_positivity(self):
        """Structural checks (non-positive) run BEFORE per-model lookup, so
        unknown model_type + zero features still raises."""
        with pytest.raises(ValueError, match=r"feature_count=0"):
            validate_input_contract("some_unknown_model", feature_count=0, window_size=20)


# =============================================================================
# preflight_trainer_config — YAML-loading wrapper
# =============================================================================


def _write_trainer_yaml(tmp_path: Path, cfg: dict) -> Path:
    """Helper — write a trainer YAML to tmp_path and return the path."""
    path = tmp_path / "trainer_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


class TestPreflightTrainerConfigHappyPath:
    """T8: happy-path YAML loads + passes through to validate_input_contract."""

    def test_full_valid_config(self, tmp_path):
        cfg = {
            "model": {"model_type": "tlob"},
            "data": {
                "feature_count": 98,
                "sequence": {"window_size": 20},
                "data_source": "mbo_lob",
            },
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        # Should not raise
        preflight_trainer_config(config_path)

    def test_minimal_valid_config_defaults_to_window_100(self, tmp_path):
        """T11: window_size absent → defaults to 100 (DataConfig default)."""
        cfg = {
            "model": {"model_type": "tlob"},
            "data": {"feature_count": 98},
            # No sequence block at all
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        # window_size falls back to 100; tlob min_seq=4 → passes
        preflight_trainer_config(config_path)

    def test_case_normalization(self, tmp_path):
        """Model name case is normalized to lowercase before registry lookup."""
        cfg = {
            "model": {"model_type": "TLOB"},   # uppercase
            "data": {"feature_count": 98, "sequence": {"window_size": 20}},
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        # Lowercase conversion in preflight_trainer_config → tlob in _INPUT_CONTRACTS
        preflight_trainer_config(config_path)


class TestPreflightTrainerConfigStructuralErrors:
    """T9-T10: missing required YAML keys."""

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ValueError, match=r"trainer config not found"):
            preflight_trainer_config(missing)

    def test_missing_model_type_raises(self, tmp_path):
        cfg = {
            "model": {},   # no model_type
            "data": {"feature_count": 98},
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match=r"missing model.model_type"):
            preflight_trainer_config(config_path)

    def test_missing_data_feature_count_raises(self, tmp_path):
        cfg = {
            "model": {"model_type": "tlob"},
            "data": {},   # no feature_count
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match=r"missing or.*non-integer data.feature_count"):
            preflight_trainer_config(config_path)

    def test_non_integer_feature_count_raises(self, tmp_path):
        cfg = {
            "model": {"model_type": "tlob"},
            "data": {"feature_count": "ninety-eight"},   # string not int
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match=r"non-integer data.feature_count"):
            preflight_trainer_config(config_path)

    def test_non_dict_yaml_raises(self, tmp_path):
        """YAML that parses to a list instead of a dict."""
        path = tmp_path / "trainer_config.yaml"
        path.write_text("- 1\n- 2\n- 3\n")  # YAML list
        with pytest.raises(ValueError, match=r"is not a mapping"):
            preflight_trainer_config(path)


class TestPreflightTrainerConfigPropagatesValidation:
    """preflight_trainer_config raises on SAME failures that
    validate_input_contract catches."""

    def test_deeplob_with_too_many_features_raises(self, tmp_path):
        cfg = {
            "model": {"model_type": "deeplob"},
            "data": {"feature_count": 128, "sequence": {"window_size": 100}},
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match=r"max_features=98"):
            preflight_trainer_config(config_path)

    def test_tlob_with_window_1_raises(self, tmp_path):
        cfg = {
            "model": {"model_type": "tlob"},
            "data": {"feature_count": 98, "sequence": {"window_size": 1}},
        }
        config_path = _write_trainer_yaml(tmp_path, cfg)
        with pytest.raises(ValueError, match=r"min_sequence_length=4"):
            preflight_trainer_config(config_path)


# =============================================================================
# _INPUT_CONTRACTS table sanity
# =============================================================================


class TestConstraintTableSanity:
    """The hardcoded table must list every model name that the trainer
    currently dispatches to. Drift-detection: if a researcher adds a model
    to lob-models without updating this table, the unknown-model WARN path
    activates silently — this sanity check catches the forgotten update
    by asserting the live registry is a subset of our table."""

    def test_table_covers_live_registry(self):
        """Forward drift: lob-models adds a model, hft-ops table missing it.
        Skip when lob-models not importable (standalone hft-ops checkout)."""
        pytest.importorskip("lobmodels")
        import lobmodels.models  # trigger @register side-effects
        from lobmodels.registry.core import ModelRegistry

        registry_names = set(ModelRegistry.list_models())
        table_names = set(_INPUT_CONTRACTS.keys())

        missing_from_table = registry_names - table_names
        assert not missing_from_table, (
            f"Models registered in lob-models but missing from hft-ops "
            f"_INPUT_CONTRACTS: {missing_from_table}. Update the table in "
            f"hft_ops.stages.contract_preflight (MVP drift-maintenance path)."
        )

    def test_table_does_not_contain_deleted_models(self):
        """Reverse drift (Phase V.1 L2.5 2026-04-21): lob-models DELETES a
        model but hft-ops table still has it. Non-critical (the dead entry
        applies constraints that can never trigger) but symptomatic of a
        forgotten sync step. Surfaces the drift at test time so future
        Phase VI snapshot migration catches all cases — forward AND
        reverse. Skip when lob-models not importable."""
        pytest.importorskip("lobmodels")
        import lobmodels.models  # trigger @register side-effects
        from lobmodels.registry.core import ModelRegistry

        registry_names = set(ModelRegistry.list_models())
        table_names = set(_INPUT_CONTRACTS.keys())

        missing_from_registry = table_names - registry_names
        assert not missing_from_registry, (
            f"Models in hft-ops _INPUT_CONTRACTS but MISSING from live "
            f"lobmodels.ModelRegistry: {missing_from_registry}. Either a "
            f"model was deleted upstream (remove the dead entry here) or "
            f"the name changed (rename here). MVP drift-maintenance path."
        )

    def test_input_contracts_is_frozen_via_mapping_proxy(self):
        """Phase V.1 L2.4 (2026-04-21): _INPUT_CONTRACTS is exposed as a
        read-only MappingProxyType wrapper — runtime mutation
        (monkeypatching, accidental assignment in a test that forgets
        cleanup) must raise TypeError instead of silently poisoning
        subsequent tests."""
        from types import MappingProxyType
        from hft_ops.stages.contract_preflight import _INPUT_CONTRACTS

        assert isinstance(_INPUT_CONTRACTS, MappingProxyType), (
            f"Expected _INPUT_CONTRACTS to be MappingProxyType; got "
            f"{type(_INPUT_CONTRACTS).__name__}. The frozen wrapper "
            f"prevents accidental runtime mutation — removing it opens "
            f"the door to test-pollution / import-order-dependent bugs."
        )
        # Attempting to add a new top-level key must raise TypeError
        with pytest.raises(TypeError):
            _INPUT_CONTRACTS["phantom_model"] = {
                "min_features": 1,
                "max_features": None,
                "min_sequence_length": 1,
                "compatible_sources": ["any"],
            }
        # Attempting to delete an existing key must raise TypeError
        with pytest.raises(TypeError):
            del _INPUT_CONTRACTS["tlob"]

    def test_contract_preflight_module_imports_are_torch_free(self):
        """Phase V.1 L2.6 (2026-04-21): static AST analysis to lock the
        architectural invariant that `hft_ops.stages.contract_preflight`
        NEVER imports torch or lobmodels at module scope. hft-ops is
        torch-free by design (root CLAUDE.md §Module Technical Map);
        Phase VI snapshot-file migration must preserve this invariant.

        This test catches an accidental `import lobmodels` at module
        scope — which would (via lobmodels.__init__.py) pull 726 torch
        modules into sys.modules and break the invariant silently.
        """
        import ast
        import hft_ops.stages.contract_preflight as _module_under_test

        src_path = Path(_module_under_test.__file__)
        assert src_path.exists(), f"Module source not found: {src_path}"
        tree = ast.parse(src_path.read_text())

        module_scope_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_scope_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_scope_imports.append(node.module or "")

        forbidden_prefixes = ("torch", "lobmodels", "lob_models", "lobtrainer")
        offenders = [
            imp for imp in module_scope_imports
            if imp and any(imp.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        assert not offenders, (
            f"hft_ops.stages.contract_preflight must NOT import torch or "
            f"lobmodels at module scope (hft-ops torch-free invariant — "
            f"root CLAUDE.md §Module Technical Map). Offending imports: "
            f"{offenders}. If a lazy import inside a function is needed, "
            f"gate it with `if TYPE_CHECKING:` or an inline `import X` "
            f"inside the function body."
        )

    def test_table_entries_have_required_fields(self):
        required_keys = {
            "min_features", "max_features",
            "min_sequence_length", "compatible_sources",
        }
        for model_name, contract in _INPUT_CONTRACTS.items():
            assert required_keys.issubset(contract.keys()), (
                f"Contract for {model_name!r} missing keys: "
                f"{required_keys - set(contract.keys())}"
            )
            assert isinstance(contract["min_features"], int)
            assert isinstance(contract["min_sequence_length"], int)
            assert contract["max_features"] is None or isinstance(contract["max_features"], int)
            assert isinstance(contract["compatible_sources"], list)
