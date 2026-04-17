"""Phase 4 Batch 4c.3: R11 hard-fail policy for fingerprint normalization.

The §3.3b class of ledger-conflation bug (distinct broken configs fingerprinting
identically because a shared input was skipped) must NEVER recur. These tests
lock that every failure mode raises `FingerprintNormalizationError` — NO silent
fallback, NO opaque pass-through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hft_ops.ledger.dedup import (
    compute_fingerprint,
    _cached_resolve_feature_set_indices,
    FingerprintNormalizationError,
)
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths

_REAL_PIPELINE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_caches():
    import hft_ops.ledger.dedup as _dedup
    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None
    yield
    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch):
    registry = tmp_path / "feature_sets"
    registry.mkdir(parents=True)
    monkeypatch.setattr(
        PipelinePaths, "feature_sets_dir", property(lambda self: registry),
    )
    paths = PipelinePaths(pipeline_root=_REAL_PIPELINE_ROOT)
    import os
    cfg_dir = paths.trainer_dir / "configs" / f"_tmp_hf_{os.getpid()}_{tmp_path.name}"
    cfg_dir.mkdir(parents=True)
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    try:
        yield paths, cfg_dir, exp_dir, registry
    finally:
        import shutil
        shutil.rmtree(cfg_dir, ignore_errors=True)


def _write_trainer_cfg(path: Path, data_block: Dict[str, Any]) -> None:
    content = {
        "data": data_block,
        "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
        "train": {"batch_size": 128, "epochs": 30, "seed": 42},
    }
    content["data"].setdefault("feature_count", 98)
    with open(path, "w") as f:
        yaml.dump(content, f)


def _write_manifest(exp_dir: Path, name: str, trainer_cfg_path: str) -> Path:
    m = {
        "experiment": {"name": name, "contract_version": "2.2"},
        "pipeline_root": "..",
        "stages": {
            "extraction": {"enabled": False, "output_dir": "fake"},
            "dataset_analysis": {"enabled": False},
            "validation": {"enabled": False},
            "training": {"enabled": True, "config": trainer_cfg_path, "output_dir": f"out/{name}"},
            "signal_export": {"enabled": False},
            "backtesting": {"enabled": False},
        },
    }
    p = exp_dir / f"{name}.yaml"
    with open(p, "w") as f:
        yaml.dump(m, f)
    return p


class TestHardFailOnMissingRegistry:
    def test_missing_feature_set_file_raises(self, scratch):
        paths, cfg_dir, exp_dir, registry = scratch
        # Registry exists but <name>.json does NOT
        cfg_path = cfg_dir / "missing_ref.yaml"
        _write_trainer_cfg(cfg_path, {"feature_set": "does_not_exist"})
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "missing", rel)

        with pytest.raises(FingerprintNormalizationError, match="does_not_exist"):
            compute_fingerprint(load_manifest(m), paths)

    def test_missing_registry_dir_raises(self, scratch, tmp_path, monkeypatch):
        paths, cfg_dir, exp_dir, _ = scratch
        # Redirect feature_sets_dir to a path that does NOT exist
        bogus = tmp_path / "bogus_does_not_exist" / "feature_sets"
        monkeypatch.setattr(
            PipelinePaths, "feature_sets_dir", property(lambda self: bogus),
        )
        cfg_path = cfg_dir / "any_ref.yaml"
        _write_trainer_cfg(cfg_path, {"feature_set": "any_name"})
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "bogus_registry", rel)

        with pytest.raises(FingerprintNormalizationError):
            compute_fingerprint(load_manifest(m), paths)


class TestHardFailOnTamperedJson:
    def test_tampered_content_hash_raises(self, scratch):
        paths, cfg_dir, exp_dir, registry = scratch
        # Write a JSON with a deliberately-wrong content_hash (FeatureSet
        # integrity-verify rejects mismatches).
        tampered = {
            "schema_version": "1.0",
            "name": "tampered_v1",
            "content_hash": "0" * 64,  # wrong hash
            "contract_version": "2.2",
            "source_feature_count": 98,
            "applies_to": {"assets": ["NVDA"], "horizons": [10]},
            "feature_indices": [5, 12],
            "feature_names": ["f_5", "f_12"],
            "produced_by": {
                "tool": "test", "tool_version": "0",
                "config_path": "test", "config_hash": "a" * 64,
                "source_profile_hash": "b" * 64,
                "data_export": "test", "data_dir_hash": "c" * 64,
            },
            "criteria": {}, "criteria_schema_version": "1.0",
            "description": "", "notes": "", "created_at": "", "created_by": "",
        }
        (registry / "tampered_v1.json").write_text(json.dumps(tampered))

        cfg_path = cfg_dir / "ref_tampered.yaml"
        _write_trainer_cfg(cfg_path, {"feature_set": "tampered_v1"})
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "tampered_exp", rel)

        with pytest.raises(FingerprintNormalizationError, match="tampered_v1"):
            compute_fingerprint(load_manifest(m), paths)


class TestHardFailOnMultiSet:
    def test_multiple_selection_fields_raises(self, scratch):
        paths, cfg_dir, exp_dir, _ = scratch
        # BOTH feature_set AND feature_indices set — DataConfig mutual
        # exclusion would catch this at trainer load, but the fingerprint
        # hook must ALSO raise (defensive — a manifest edit could bypass
        # trainer validation).
        cfg_path = cfg_dir / "multi.yaml"
        _write_trainer_cfg(cfg_path, {
            "feature_set": "whatever",
            "feature_indices": [1, 2, 3],
        })
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "multi_exp", rel)

        with pytest.raises(FingerprintNormalizationError, match="multiple feature-selection"):
            compute_fingerprint(load_manifest(m), paths)


class TestHardFailOnInvalidShape:
    def test_non_string_feature_set_raises(self, scratch):
        paths, cfg_dir, exp_dir, _ = scratch
        cfg_path = cfg_dir / "bad_fs.yaml"
        _write_trainer_cfg(cfg_path, {"feature_set": 42})  # not a string
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "bad_fs_exp", rel)

        with pytest.raises(FingerprintNormalizationError, match="non-empty string"):
            compute_fingerprint(load_manifest(m), paths)

    def test_non_list_feature_indices_raises(self, scratch):
        paths, cfg_dir, exp_dir, _ = scratch
        cfg_path = cfg_dir / "bad_fi.yaml"
        _write_trainer_cfg(cfg_path, {"feature_indices": "not_a_list"})
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "bad_fi_exp", rel)

        with pytest.raises(FingerprintNormalizationError, match="list/tuple"):
            compute_fingerprint(load_manifest(m), paths)


class TestStubCleanupOnSuccess:
    """Phase 6 6A.4 regression guard — sys-modules stubs installed during
    `_load_trainer_feature_presets_module` must be POPPED on BOTH the success
    AND failure paths (previously rolled back only on failure; success left
    stubs polluting sys.modules interpreter-wide, breaking downstream test
    isolation + notebook workflows that legitimately want the real package).
    """

    def test_stubs_cleaned_after_successful_fingerprint_normalization(self, scratch):
        """After a successful fingerprint compute that exercises the
        feature_preset normalization path (which loads the trainer's
        feature_presets.py via sys-modules stubbing), the stubs MUST be
        popped so that `lobtrainer.constants.feature_index` (the real
        module, if pre-existing in sys.modules, or absent if not) is
        untouched for subsequent imports."""
        import sys
        from hft_ops.ledger.dedup import _load_trainer_feature_presets_module

        paths, _, _, _ = scratch
        # Capture pre-state. We do NOT assume lobtrainer is installed.
        pre_state = {
            name: sys.modules.get(name)
            for name in [
                "lobtrainer",
                "lobtrainer.constants",
                "lobtrainer.constants.feature_index",
            ]
        }

        # Clear the module cache so the loader actually installs stubs again.
        import hft_ops.ledger.dedup as _dedup
        _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None

        # Invoke the loader (success path). This is the same entry point
        # `compute_fingerprint_explain` uses when normalizing feature_preset.
        module = _load_trainer_feature_presets_module(paths)
        assert module is not None
        assert hasattr(module, "FEATURE_PRESETS"), (
            "Loader must produce a module exposing FEATURE_PRESETS"
        )

        # Post-state: the stubs we installed (names not in pre_state as
        # pre-existing) must have been popped by the finally-block.
        for name in [
            "lobtrainer",
            "lobtrainer.constants",
            "lobtrainer.constants.feature_index",
        ]:
            if pre_state[name] is None:
                # Was NOT in sys.modules pre-call → our stub was the only entry
                # → must be popped post-call.
                assert name not in sys.modules, (
                    f"Stub `{name}` leaked into sys.modules after successful "
                    f"fingerprint normalization (Phase 6 6A.4 regression). "
                    f"Pollutes interpreter-wide — subsequent notebook / test "
                    f"that legitimately imports real lobtrainer gets the stub."
                )
            else:
                # Was pre-existing real package → our stub was NEVER installed
                # → same real entry must still be there.
                assert sys.modules.get(name) is pre_state[name], (
                    f"Pre-existing real package at sys.modules[{name!r}] was "
                    f"clobbered by the stub-cleanup. MUST preserve pre-existing "
                    f"entries — idempotent-only cleanup (Phase 6 6A.4)."
                )
