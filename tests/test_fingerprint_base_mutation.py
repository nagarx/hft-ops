"""§3.3b regression guard: base mutation MUST change dependent fingerprints.

The PRE-fix behavior: ``compute_fingerprint`` loaded trainer YAMLs via raw
``yaml.safe_load``, seeing the sparse ``_base: [...]`` + overrides as the
hash input. Mutating a shared base (e.g., ``bases/train/regression_default.yaml:
epochs 30 → 40``) left every dependent experiment's fingerprint unchanged
— silently conflating pre/post-mutation runs in the ledger.

The POST-fix behavior (§3.3b of the Phase 3 plan): both path-based AND
inline trainer configs are resolved through ``resolve_inheritance`` before
fingerprinting. The hash reflects the effective resolved dict, so base
mutations propagate correctly.

These tests lock that behavior in. They do NOT use the fake ``tmp_pipeline``
fixture because that mocks trainer_dir; instead, they point ``pipeline_root``
at the real repo root so the actual ``lob-model-trainer/src/lobtrainer/config/merge.py``
is loadable via ``spec_from_file_location``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hft_ops.ledger.dedup import compute_fingerprint
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


# The REAL pipeline root — same machine, real trainer merge.py available.
_REAL_PIPELINE_ROOT = Path(__file__).resolve().parents[2]

assert (_REAL_PIPELINE_ROOT / "lob-model-trainer" / "src" / "lobtrainer" / "config" / "merge.py").exists(), (
    "This test requires the real trainer merge.py to exist; run from the monorepo root."
)


# -----------------------------------------------------------------------------
# Helper: write a trainer config tree into a test-private subdir of the real
# trainer so `_load_trainer_merge_module` can find merge.py at the normal path.
# -----------------------------------------------------------------------------


def _write_base(
    trainer_configs: Path,
    rel_path: str,
    content: Dict[str, Any],
) -> Path:
    """Write a base YAML under <trainer_configs>/<rel_path>. Returns abs path."""
    p = trainer_configs / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(content, f)
    return p


def _write_manifest(
    hft_ops_experiments: Path,
    name: str,
    trainer_config_path: str | None = None,
    trainer_config_inline: Dict[str, Any] | None = None,
    extraction_output_dir: str = "data/exports/fake",
) -> Path:
    """Write a synthetic manifest YAML pointing at a trainer config."""
    training_block: Dict[str, Any] = {
        "enabled": True,
        "output_dir": f"outputs/{name}",
    }
    if trainer_config_path is not None:
        training_block["config"] = trainer_config_path
    if trainer_config_inline is not None:
        training_block["trainer_config"] = trainer_config_inline

    manifest = {
        "experiment": {"name": name, "contract_version": "2.2"},
        "pipeline_root": "..",
        "stages": {
            "extraction": {
                "enabled": False,
                "output_dir": extraction_output_dir,
            },
            "dataset_analysis": {"enabled": False},
            "validation": {"enabled": False},
            "training": training_block,
            "signal_export": {"enabled": False},
            "backtesting": {"enabled": False},
        },
    }
    path = hft_ops_experiments / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return path


@pytest.fixture
def scratch_trainer_configs(tmp_path: Path):
    """Yield a tuple of (paths, trainer_configs_dir, hft_ops_experiments_dir).

    ``paths`` uses the REAL pipeline root so the trainer merge.py loads. We
    write synthetic YAMLs into a test-private subdirectory of the real
    ``lob-model-trainer/configs/``, clean them up at teardown.

    We DON'T pollute the shared bases tree — we write into a per-test
    subdirectory like ``configs/_tmp_fingerprint_test_<pid>/``.
    """
    paths = PipelinePaths(pipeline_root=_REAL_PIPELINE_ROOT)
    import os

    scratch_dir = (
        paths.trainer_dir
        / "configs"
        / f"_tmp_fingerprint_test_{os.getpid()}_{tmp_path.name}"
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # hft-ops experiments scratch — use tmp_path directly
    hft_ops_exp_dir = tmp_path / "experiments"
    hft_ops_exp_dir.mkdir()

    try:
        yield paths, scratch_dir, hft_ops_exp_dir
    finally:
        # Cleanup
        import shutil
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestBaseMutationChangesFingerprint:
    """The core §3.3b assertion: base mutation → fingerprint change."""

    def test_base_mutation_changes_path_based_fingerprint(
        self,
        scratch_trainer_configs,
    ):
        paths, scratch_dir, hft_ops_exp_dir = scratch_trainer_configs

        # Write a base with epochs=30
        base_rel = f"_tmp_fingerprint_test_{scratch_dir.name.split('_')[-1]}/base.yaml"
        base_path = scratch_dir / "base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(
                {
                    "data": {"feature_count": 98},
                    "model": {
                        "model_type": "tlob",
                        "input_size": 98,
                        "num_classes": 3,
                    },
                    "train": {"batch_size": 128, "epochs": 30, "seed": 42},
                },
                f,
            )

        # Write a child that _base:'s the base
        child_path = scratch_dir / "child.yaml"
        with open(child_path, "w") as f:
            yaml.dump(
                {
                    "_base": "base.yaml",
                    "name": "child_exp",
                },
                f,
            )

        # Compute child_path rel to pipeline_root for the manifest
        child_rel = str(child_path.relative_to(paths.pipeline_root))

        m_path = _write_manifest(
            hft_ops_exp_dir, "exp_v1", trainer_config_path=child_rel
        )
        manifest = load_manifest(m_path)

        fp_before = compute_fingerprint(manifest, paths)

        # Mutate the base: epochs 30 → 40
        with open(base_path, "w") as f:
            yaml.dump(
                {
                    "data": {"feature_count": 98},
                    "model": {
                        "model_type": "tlob",
                        "input_size": 98,
                        "num_classes": 3,
                    },
                    "train": {"batch_size": 128, "epochs": 40, "seed": 42},
                },
                f,
            )

        fp_after = compute_fingerprint(manifest, paths)

        assert fp_before != fp_after, (
            "CRITICAL §3.3b regression: base mutation did NOT change the "
            "fingerprint. This is the ledger-conflation bug: shared-base "
            "changes now silently leave dependent experiment fingerprints "
            "identical. Fix: ensure `compute_fingerprint` calls "
            "`resolve_inheritance` on trainer configs."
        )

    def test_base_mutation_changes_inline_fingerprint(
        self,
        scratch_trainer_configs,
    ):
        """Same behavior for inline trainer_config with `_base:` — the
        inline form must also resolve through the trainer's merge.py before
        fingerprinting."""
        paths, scratch_dir, hft_ops_exp_dir = scratch_trainer_configs

        base_path = scratch_dir / "inline_base.yaml"
        with open(base_path, "w") as f:
            yaml.dump(
                {
                    "data": {"feature_count": 98},
                    "model": {
                        "model_type": "tlob",
                        "input_size": 98,
                        "num_classes": 3,
                    },
                    "train": {"batch_size": 128, "epochs": 30, "seed": 42},
                },
                f,
            )

        # Path to the base relative to trainer_configs root (how _base paths
        # are interpreted for inline trainer_config per
        # _absolutize_inline_base_paths convention)
        base_rel = str(
            base_path.relative_to(paths.trainer_dir / "configs")
        )

        inline = {"_base": base_rel, "name": "inline_child"}
        m_path = _write_manifest(
            hft_ops_exp_dir, "inline_v1", trainer_config_inline=inline
        )
        manifest = load_manifest(m_path)

        fp_before = compute_fingerprint(manifest, paths)

        # Mutate base
        with open(base_path, "w") as f:
            yaml.dump(
                {
                    "data": {"feature_count": 98},
                    "model": {
                        "model_type": "tlob",
                        "input_size": 98,
                        "num_classes": 3,
                    },
                    "train": {"batch_size": 128, "epochs": 40, "seed": 42},
                },
                f,
            )

        # IMPORTANT: manifest's inline `trainer_config` is a frozen snapshot
        # from load_manifest time. Force a re-load so the mutation is seen.
        manifest2 = load_manifest(m_path)
        fp_after = compute_fingerprint(manifest2, paths)

        assert fp_before != fp_after, (
            "Inline trainer_config with `_base:` must also resolve "
            "inheritance for fingerprinting (§3.3b)."
        )


class TestInlineAndPathEquivalence:
    """A manifest with `training.config: child.yaml` and a manifest with
    `training.trainer_config: <child's dict>` must produce IDENTICAL
    fingerprints (two entry points to the same effective config).
    """

    def test_path_and_inline_equivalent_no_base(
        self,
        scratch_trainer_configs,
    ):
        """Simplest case: no inheritance. Path-based and inline should match."""
        paths, scratch_dir, hft_ops_exp_dir = scratch_trainer_configs

        content = {
            "data": {"feature_count": 98},
            "model": {
                "model_type": "tlob",
                "input_size": 98,
                "num_classes": 3,
            },
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        # Path form
        path_cfg = scratch_dir / "no_base.yaml"
        with open(path_cfg, "w") as f:
            yaml.dump(content, f)
        rel = str(path_cfg.relative_to(paths.pipeline_root))

        m1 = _write_manifest(
            hft_ops_exp_dir, "form_path", trainer_config_path=rel
        )
        fp_path = compute_fingerprint(load_manifest(m1), paths)

        # Inline form with the exact same content
        m2 = _write_manifest(
            hft_ops_exp_dir, "form_inline", trainer_config_inline=dict(content)
        )
        fp_inline = compute_fingerprint(load_manifest(m2), paths)

        assert fp_path == fp_inline, (
            "Path-based and inline trainer configs of equivalent content "
            "must produce identical fingerprints — both load through the "
            "same _extract_fingerprint_fields pipeline."
        )


class TestContentAddressedFingerprint:
    """§3.3b contract: fingerprints are content-addressed, NOT path-addressed.

    Two different base files with IDENTICAL content must produce IDENTICAL
    fingerprints for their dependent configs. Post-Batch-1 audit item G6 —
    validates that the fingerprint pipeline truly hashes the resolved
    effective dict rather than smuggling the base filename into the hash.
    """

    def test_two_bases_same_content_same_fingerprint(
        self,
        scratch_trainer_configs,
    ):
        """Base-A and Base-B with identical content → same fingerprint.

        Proves that the fingerprint depends on the CONTENT of resolved dict,
        not on which file path was used as the base. If this ever fails, the
        §3.3b resolve-before-hash contract has regressed into a path-based
        hash — meaning researchers who rename a base file would see false
        "new experiment" entries in the ledger despite zero behavioral change.
        """
        paths, scratch_dir, hft_ops_exp_dir = scratch_trainer_configs

        base_content = {
            "data": {"feature_count": 98},
            "model": {
                "model_type": "tlob",
                "input_size": 98,
                "num_classes": 3,
            },
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        # Write TWO base files with identical content but different names
        base_a_path = scratch_dir / "base_a.yaml"
        base_b_path = scratch_dir / "base_b.yaml"
        for p in (base_a_path, base_b_path):
            with open(p, "w") as f:
                yaml.dump(base_content, f)

        # Two children, each inheriting from a different base (same content)
        child_a = scratch_dir / "child_a.yaml"
        child_b = scratch_dir / "child_b.yaml"
        with open(child_a, "w") as f:
            yaml.dump({"_base": "base_a.yaml", "name": "A"}, f)
        with open(child_b, "w") as f:
            yaml.dump({"_base": "base_b.yaml", "name": "B"}, f)

        rel_a = str(child_a.relative_to(paths.pipeline_root))
        rel_b = str(child_b.relative_to(paths.pipeline_root))

        m_a = _write_manifest(hft_ops_exp_dir, "exp_a", trainer_config_path=rel_a)
        m_b = _write_manifest(hft_ops_exp_dir, "exp_b", trainer_config_path=rel_b)

        fp_a = compute_fingerprint(load_manifest(m_a), paths)
        fp_b = compute_fingerprint(load_manifest(m_b), paths)

        assert fp_a == fp_b, (
            "CRITICAL §3.3b regression: two bases with identical content "
            "produced DIFFERENT fingerprints. The fingerprint has become "
            "path-addressed instead of content-addressed. Check that "
            "`_load_trainer_config_resolved` hashes the resolved dict and "
            "not the base file path."
        )

    def test_different_content_different_fingerprint(
        self,
        scratch_trainer_configs,
    ):
        """Sibling of above: if bases have DIFFERENT content, fingerprints differ.

        This guards against the other failure mode: content-blind hashing
        that always returns the same value regardless of input.
        """
        paths, scratch_dir, hft_ops_exp_dir = scratch_trainer_configs

        base_content_1 = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }
        base_content_2 = dict(base_content_1)
        base_content_2["train"] = dict(base_content_1["train"])
        base_content_2["train"]["epochs"] = 60   # differs!

        base_1 = scratch_dir / "content_1.yaml"
        base_2 = scratch_dir / "content_2.yaml"
        with open(base_1, "w") as f:
            yaml.dump(base_content_1, f)
        with open(base_2, "w") as f:
            yaml.dump(base_content_2, f)

        child_1 = scratch_dir / "child_c1.yaml"
        child_2 = scratch_dir / "child_c2.yaml"
        with open(child_1, "w") as f:
            yaml.dump({"_base": "content_1.yaml", "name": "C1"}, f)
        with open(child_2, "w") as f:
            yaml.dump({"_base": "content_2.yaml", "name": "C2"}, f)

        rel_1 = str(child_1.relative_to(paths.pipeline_root))
        rel_2 = str(child_2.relative_to(paths.pipeline_root))

        m_1 = _write_manifest(hft_ops_exp_dir, "exp_c1", trainer_config_path=rel_1)
        m_2 = _write_manifest(hft_ops_exp_dir, "exp_c2", trainer_config_path=rel_2)

        fp_1 = compute_fingerprint(load_manifest(m_1), paths)
        fp_2 = compute_fingerprint(load_manifest(m_2), paths)

        assert fp_1 != fp_2, (
            "Different base content must produce different fingerprints. "
            "If they match, the fingerprint has become content-blind."
        )
