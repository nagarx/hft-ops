"""Phase 4 Batch 4c.3 regression guard: feature_set ↔ feature_indices
fingerprint equivalence + content-mutation lock.

Mirrors ``test_fingerprint_base_mutation.py`` (Phase 3 §3.3b). Locks:
1. A manifest with ``data.feature_set: <name>`` fingerprints IDENTICALLY
   to an equivalent manifest with ``data.feature_indices: [<resolved>]``.
2. Regenerating the same-named FeatureSet with DIFFERENT indices changes
   the fingerprint (content-addressed lock).
3. ``data.feature_preset: <name>`` normalizes to the same fingerprint as
   the equivalent explicit indices.
4. Sort/dedup is order-invariant for the fingerprint — two manifests with
   different index ordering but equal sorted-deduped SET produce equal
   fingerprints.
5. Inline trainer_config dict with feature_set matches path-based form.

If ANY of these fail: the 4c.3 hook regressed into a pre-4c.3 pass-through
or into a path-addressed hash; ledger-conflation is back.
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
    _TRAINER_FEATURE_PRESETS_MODULE_CACHE,
)
from hft_ops.feature_sets.schema import FeatureSet, FeatureSetAppliesTo, FeatureSetProducedBy
from hft_contracts._testing import require_monorepo_root
from hft_ops.feature_sets.writer import write_feature_set
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


# Monorepo root via SSoT helper (Phase V.A.0). Skips module-level on
# standalone-clone environments where the trainer sibling is absent.
_REAL_PIPELINE_ROOT = require_monorepo_root(
    "lob-model-trainer/src/lobtrainer/config/merge.py",
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Reset module-level caches between tests (isolation)."""
    import hft_ops.ledger.dedup as _dedup

    # Clear LRU cache on the feature-set resolver
    _cached_resolve_feature_set_indices.cache_clear()
    # Clear the feature_presets module cache (may be polluted from prior tests)
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None
    yield
    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None


@pytest.fixture
def scratch_paths(tmp_path: Path, monkeypatch):
    """Real pipeline_root (for trainer merge.py) + tmp feature_sets_dir.

    Monkeypatches ``PipelinePaths.feature_sets_dir`` so writes to the registry
    land in a tmp directory instead of polluting the real monorepo
    ``contracts/feature_sets/``.
    """
    tmp_registry = tmp_path / "feature_sets"
    tmp_registry.mkdir(parents=True, exist_ok=True)

    # Monkeypatch the property to return our tmp dir
    monkeypatch.setattr(
        PipelinePaths,
        "feature_sets_dir",
        property(lambda self: tmp_registry),
    )

    paths = PipelinePaths(pipeline_root=_REAL_PIPELINE_ROOT)

    # Per-test trainer scratch dir (shared pattern with test_fingerprint_base_mutation)
    import os
    scratch_dir = (
        paths.trainer_dir
        / "configs"
        / f"_tmp_fs_test_{os.getpid()}_{tmp_path.name}"
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)

    hft_ops_exp_dir = tmp_path / "experiments"
    hft_ops_exp_dir.mkdir()

    try:
        yield paths, scratch_dir, hft_ops_exp_dir, tmp_registry
    finally:
        import shutil
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)


def _build_fs(
    name: str,
    indices: list[int],
    source_feature_count: int = 98,
    contract_version: str = "2.2",
) -> FeatureSet:
    """Helper: build a synthetic FeatureSet for tests."""
    return FeatureSet.build(
        name=name,
        feature_indices=indices,
        feature_names=[f"f_{i}" for i in indices],
        source_feature_count=source_feature_count,
        contract_version=contract_version,
        applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(10,)),
        produced_by=FeatureSetProducedBy(
            tool="hft-feature-evaluator",
            tool_version="test",
            config_path="test.yaml",
            config_hash="a" * 64,
            source_profile_hash="b" * 64,
            data_export="test",
            data_dir_hash="c" * 64,
        ),
        criteria={"name": "test"},
        criteria_schema_version="1.0",
    )


def _write_manifest(
    exp_dir: Path,
    name: str,
    trainer_cfg_path: str | None = None,
    trainer_cfg_inline: Dict[str, Any] | None = None,
) -> Path:
    """Write a minimal manifest pointing at trainer config."""
    training: Dict[str, Any] = {"enabled": True, "output_dir": f"outputs/{name}"}
    if trainer_cfg_path is not None:
        training["config"] = trainer_cfg_path
    if trainer_cfg_inline is not None:
        training["trainer_config"] = trainer_cfg_inline

    manifest = {
        "experiment": {"name": name, "contract_version": "2.2"},
        "pipeline_root": "..",
        "stages": {
            "extraction": {"enabled": False, "output_dir": "data/exports/fake"},
            "dataset_analysis": {"enabled": False},
            "validation": {"enabled": False},
            "training": training,
            "signal_export": {"enabled": False},
            "backtesting": {"enabled": False},
        },
    }
    path = exp_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureSetIndicesEquivalence:
    """Two configs differing only in feature_set vs equivalent feature_indices
    MUST produce byte-equal fingerprints (the core 4c.3 contract)."""

    def test_feature_set_equals_explicit_indices(self, scratch_paths):
        """feature_set: <name> ↔ feature_indices: [<resolved>] → SAME fingerprint."""
        paths, scratch, exp_dir, registry = scratch_paths

        indices = [5, 12, 84, 85]
        fs = _build_fs("momentum_test_v1", indices)
        write_feature_set(registry / "momentum_test_v1.json", fs)

        shared_body = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        # Variant A: feature_set reference
        cfg_a = dict(shared_body)
        cfg_a["data"] = dict(shared_body["data"])
        cfg_a["data"]["feature_set"] = "momentum_test_v1"
        path_a = scratch / "with_feature_set.yaml"
        with open(path_a, "w") as f:
            yaml.dump(cfg_a, f)

        # Variant B: explicit feature_indices (sorted-deduped equivalent)
        cfg_b = dict(shared_body)
        cfg_b["data"] = dict(shared_body["data"])
        cfg_b["data"]["feature_indices"] = sorted(set(indices))
        path_b = scratch / "with_indices.yaml"
        with open(path_b, "w") as f:
            yaml.dump(cfg_b, f)

        rel_a = str(path_a.relative_to(paths.pipeline_root))
        rel_b = str(path_b.relative_to(paths.pipeline_root))

        m_a = _write_manifest(exp_dir, "exp_a", trainer_cfg_path=rel_a)
        m_b = _write_manifest(exp_dir, "exp_b", trainer_cfg_path=rel_b)

        fp_a = compute_fingerprint(load_manifest(m_a), paths)
        fp_b = compute_fingerprint(load_manifest(m_b), paths)

        assert fp_a == fp_b, (
            "CRITICAL 4c.3 regression: feature_set and equivalent feature_indices "
            "produced DIFFERENT fingerprints. The normalization hook regressed. "
            f"fp_feature_set={fp_a}, fp_indices={fp_b}"
        )

    def test_unsorted_indices_normalize_to_same_fingerprint(self, scratch_paths):
        """Different index orderings with same SET → SAME fingerprint."""
        paths, scratch, exp_dir, _ = scratch_paths

        shared_body = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        cfg_unsorted = dict(shared_body)
        cfg_unsorted["data"] = dict(shared_body["data"])
        cfg_unsorted["data"]["feature_indices"] = [84, 5, 12, 85]  # unsorted

        cfg_sorted = dict(shared_body)
        cfg_sorted["data"] = dict(shared_body["data"])
        cfg_sorted["data"]["feature_indices"] = [5, 12, 84, 85]  # sorted

        cfg_dup = dict(shared_body)
        cfg_dup["data"] = dict(shared_body["data"])
        cfg_dup["data"]["feature_indices"] = [5, 5, 12, 84, 12, 85]  # duplicates

        results = []
        for label, cfg in [("unsorted", cfg_unsorted), ("sorted", cfg_sorted), ("dup", cfg_dup)]:
            path = scratch / f"idx_{label}.yaml"
            with open(path, "w") as f:
                yaml.dump(cfg, f)
            rel = str(path.relative_to(paths.pipeline_root))
            m = _write_manifest(exp_dir, f"exp_{label}", trainer_cfg_path=rel)
            results.append(compute_fingerprint(load_manifest(m), paths))

        assert results[0] == results[1] == results[2], (
            f"Normalization sort/dedup failed: unsorted={results[0]}, "
            f"sorted={results[1]}, duplicated={results[2]}. All should be equal."
        )


class TestFeatureSetContentAddressedLock:
    """Regenerating a FeatureSet with the same name but different indices MUST
    change the fingerprint. Content-addressed identity protects against
    invisible registry mutations."""

    def test_regenerating_feature_set_changes_fingerprint(self, scratch_paths):
        paths, scratch, exp_dir, registry = scratch_paths

        # Write v1 with [5, 12]
        fs_v1 = _build_fs("locked_v1", [5, 12])
        write_feature_set(registry / "locked_v1.json", fs_v1)

        cfg = {
            "data": {"feature_count": 98, "feature_set": "locked_v1"},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }
        cfg_path = scratch / "locked.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        rel = str(cfg_path.relative_to(paths.pipeline_root))
        m = _write_manifest(exp_dir, "locked_exp", trainer_cfg_path=rel)

        fp_before = compute_fingerprint(load_manifest(m), paths)

        # Regenerate with same name but DIFFERENT indices. Must clear the LRU
        # cache to simulate a fresh process (production sweep grids run in one
        # process, but a registry regeneration across processes would pick up
        # the new content; we clear to test that path).
        _cached_resolve_feature_set_indices.cache_clear()
        fs_v1_mutated = _build_fs("locked_v1", [5, 12, 84])  # new index added
        write_feature_set(registry / "locked_v1.json", fs_v1_mutated, force=True)

        fp_after = compute_fingerprint(load_manifest(m), paths)

        assert fp_before != fp_after, (
            "CRITICAL 4c.3 regression: regenerating the same-named FeatureSet "
            "with different indices produced the SAME fingerprint. Content-"
            "addressed lock broken. Two distinct ML setups would now dedup "
            "silently in the ledger."
        )


class TestFeaturePresetNormalization:
    """feature_preset: <name> should normalize to the same fingerprint as
    the equivalent explicit indices (preset is a DEPRECATED alias for a
    fixed index list; fingerprint sees only the indices)."""

    def test_feature_preset_matches_equivalent_indices(self, scratch_paths):
        paths, scratch, exp_dir, _ = scratch_paths

        # lob_only preset = range(0, 40)
        shared_body = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 40, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        cfg_preset = dict(shared_body)
        cfg_preset["data"] = dict(shared_body["data"])
        cfg_preset["data"]["feature_preset"] = "lob_only"
        path_preset = scratch / "preset.yaml"
        with open(path_preset, "w") as f:
            yaml.dump(cfg_preset, f)

        cfg_indices = dict(shared_body)
        cfg_indices["data"] = dict(shared_body["data"])
        cfg_indices["data"]["feature_indices"] = list(range(40))
        path_indices = scratch / "indices.yaml"
        with open(path_indices, "w") as f:
            yaml.dump(cfg_indices, f)

        rel_preset = str(path_preset.relative_to(paths.pipeline_root))
        rel_indices = str(path_indices.relative_to(paths.pipeline_root))

        m_preset = _write_manifest(exp_dir, "exp_preset", trainer_cfg_path=rel_preset)
        m_indices = _write_manifest(exp_dir, "exp_indices", trainer_cfg_path=rel_indices)

        fp_preset = compute_fingerprint(load_manifest(m_preset), paths)
        fp_indices = compute_fingerprint(load_manifest(m_indices), paths)

        assert fp_preset == fp_indices, (
            "feature_preset='lob_only' and feature_indices=[0..39] produced "
            f"different fingerprints: preset={fp_preset}, indices={fp_indices}. "
            "4c.3 preset normalization regressed."
        )


class TestInlineVsPathFeatureSetEquivalence:
    """Inline trainer_config dict with feature_set must match path-based form."""

    def test_inline_feature_set_matches_path_form(self, scratch_paths):
        paths, scratch, exp_dir, registry = scratch_paths

        fs = _build_fs("inline_test_v1", [0, 5, 12])
        write_feature_set(registry / "inline_test_v1.json", fs)

        body = {
            "data": {"feature_count": 98, "feature_set": "inline_test_v1"},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        path_form = scratch / "inline_ref.yaml"
        with open(path_form, "w") as f:
            yaml.dump(body, f)
        rel = str(path_form.relative_to(paths.pipeline_root))

        m_path = _write_manifest(exp_dir, "path_form", trainer_cfg_path=rel)
        m_inline = _write_manifest(exp_dir, "inline_form", trainer_cfg_inline=dict(body))

        fp_path = compute_fingerprint(load_manifest(m_path), paths)
        fp_inline = compute_fingerprint(load_manifest(m_inline), paths)

        assert fp_path == fp_inline, (
            f"Inline and path-based feature_set configs produced different "
            f"fingerprints: path={fp_path}, inline={fp_inline}."
        )
