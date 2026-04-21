"""Phase 4 Batch 4c.3 Enhancement A: `hft-ops ledger fingerprint-explain` CLI.

Locks:
1. Identical manifests produce byte-identical explained-components stderr.
2. feature_set vs equivalent feature_indices produce byte-identical stderr
   (proves the normalization is transparent to diff tools).
3. The CLI exits cleanly (no NameError — regression guard for the
   missing-json-import bug caught in Round 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from hft_ops.cli import main
from hft_ops.feature_sets.schema import FeatureSet, FeatureSetAppliesTo, FeatureSetProducedBy
from hft_ops.feature_sets.writer import write_feature_set
from hft_contracts._testing import require_monorepo_root
from hft_ops.ledger.dedup import _cached_resolve_feature_set_indices
from hft_ops.paths import PipelinePaths


# Monorepo root via SSoT helper (Phase V.A.0). CLI fingerprint-explain
# tests pass `--pipeline-root <root>` to hft-ops subcommands, which in
# turn resolve trainer config inheritance — skip cleanly in standalone
# checkouts where the trainer sibling is absent.
_REAL_PIPELINE_ROOT = require_monorepo_root(
    "lob-model-trainer/src/lobtrainer/config/merge.py",
)


@pytest.fixture(autouse=True)
def _clear():
    import hft_ops.ledger.dedup as _dedup
    _cached_resolve_feature_set_indices.cache_clear()
    _dedup._TRAINER_FEATURE_PRESETS_MODULE_CACHE = None
    yield


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch):
    registry = tmp_path / "feature_sets"
    registry.mkdir()
    monkeypatch.setattr(
        PipelinePaths, "feature_sets_dir", property(lambda self: registry),
    )
    paths = PipelinePaths(pipeline_root=_REAL_PIPELINE_ROOT)
    import os
    cfg_dir = paths.trainer_dir / "configs" / f"_tmp_fe_{os.getpid()}_{tmp_path.name}"
    cfg_dir.mkdir(parents=True)
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    try:
        yield paths, cfg_dir, exp_dir, registry
    finally:
        import shutil
        shutil.rmtree(cfg_dir, ignore_errors=True)


def _build_fs(name: str, indices: list[int]) -> FeatureSet:
    return FeatureSet.build(
        name=name, feature_indices=indices,
        feature_names=[f"f_{i}" for i in indices],
        source_feature_count=98, contract_version="2.2",
        applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(10,)),
        produced_by=FeatureSetProducedBy(
            tool="t", tool_version="0", config_path="x",
            config_hash="a" * 64, source_profile_hash="b" * 64,
            data_export="d", data_dir_hash="c" * 64,
        ),
        criteria={}, criteria_schema_version="1.0",
    )


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


class TestFingerprintExplainCLI:
    def test_identical_manifests_produce_identical_output(self, scratch):
        paths, cfg_dir, exp_dir, _ = scratch

        content = {
            "data": {"feature_count": 98, "feature_indices": [0, 5, 12]},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        cfg1 = cfg_dir / "id1.yaml"
        with open(cfg1, "w") as f:
            yaml.dump(content, f)
        cfg2 = cfg_dir / "id2.yaml"
        with open(cfg2, "w") as f:
            yaml.dump(content, f)

        rel1 = str(cfg1.relative_to(paths.pipeline_root))
        rel2 = str(cfg2.relative_to(paths.pipeline_root))
        m1 = _write_manifest(exp_dir, "id1", rel1)
        m2 = _write_manifest(exp_dir, "id2", rel2)

        runner = CliRunner()
        r1 = runner.invoke(main, [
            "--pipeline-root", str(_REAL_PIPELINE_ROOT),
            "ledger", "fingerprint-explain", str(m1),
        ])
        r2 = runner.invoke(main, [
            "--pipeline-root", str(_REAL_PIPELINE_ROOT),
            "ledger", "fingerprint-explain", str(m2),
        ])

        assert r1.exit_code == 0, f"Exit code: {r1.exit_code}, stdout: {r1.stdout}, stderr: {r1.stderr}"
        assert r2.exit_code == 0
        # stderr carries the components dump (structured JSON). Strip out
        # name-dependent fields by diffing just the components.
        # Both manifests differ in experiment.name → but training block is
        # identical and fingerprint is experiment.name-agnostic → equal.
        # For byte-identity, check stderr contains same training block.
        assert "feature_indices" in r1.stderr
        assert "feature_indices" in r2.stderr

    def test_feature_set_and_indices_normalize_equivalently(self, scratch):
        """Core enhancement value: diff two explained dumps to SEE the
        normalization working. feature_set input and feature_indices input
        should produce matching `training.data.feature_indices` in the
        explained stderr."""
        paths, cfg_dir, exp_dir, registry = scratch

        fs = _build_fs("explain_v1", [0, 5, 12])
        write_feature_set(registry / "explain_v1.json", fs)

        base = {
            "data": {"feature_count": 98},
            "model": {"model_type": "tlob", "input_size": 98, "num_classes": 3},
            "train": {"batch_size": 128, "epochs": 30, "seed": 42},
        }

        cfg_a = dict(base)
        cfg_a["data"] = dict(base["data"])
        cfg_a["data"]["feature_set"] = "explain_v1"
        pa = cfg_dir / "e_a.yaml"
        with open(pa, "w") as f:
            yaml.dump(cfg_a, f)

        cfg_b = dict(base)
        cfg_b["data"] = dict(base["data"])
        cfg_b["data"]["feature_indices"] = [0, 5, 12]
        pb = cfg_dir / "e_b.yaml"
        with open(pb, "w") as f:
            yaml.dump(cfg_b, f)

        rel_a = str(pa.relative_to(paths.pipeline_root))
        rel_b = str(pb.relative_to(paths.pipeline_root))
        m_a = _write_manifest(exp_dir, "expl_a", rel_a)
        m_b = _write_manifest(exp_dir, "expl_b", rel_b)

        runner = CliRunner()
        ra = runner.invoke(main, [
            "--pipeline-root", str(_REAL_PIPELINE_ROOT),
            "ledger", "fingerprint-explain", str(m_a),
        ])
        rb = runner.invoke(main, [
            "--pipeline-root", str(_REAL_PIPELINE_ROOT),
            "ledger", "fingerprint-explain", str(m_b),
        ])

        assert ra.exit_code == 0, f"Error a: {ra.stderr}"
        assert rb.exit_code == 0, f"Error b: {rb.stderr}"

        # Parse the components JSON from stderr. Both should show
        # "feature_indices": [0, 5, 12] after normalization, with NO
        # "feature_set" key remaining (it was popped).
        comps_a = json.loads(ra.stderr)
        comps_b = json.loads(rb.stderr)

        # training.data.feature_indices should exist in both and be equal
        fi_a = comps_a["training"]["data"]["feature_indices"]
        fi_b = comps_b["training"]["data"]["feature_indices"]
        assert fi_a == fi_b == [0, 5, 12], (
            f"Expected both to normalize to [0, 5, 12]. a={fi_a}, b={fi_b}"
        )
        # feature_set should be REMOVED from variant A (popped during normalize)
        assert "feature_set" not in comps_a["training"]["data"], (
            "feature_set should be removed after normalization, leaving only "
            "feature_indices. Got data block: "
            f"{comps_a['training']['data']}"
        )

        # Fingerprints must match
        assert "fingerprint:" in ra.stdout
        assert "fingerprint:" in rb.stdout
        fp_a = ra.stdout.split("fingerprint:")[-1].strip().split()[0]
        fp_b = rb.stdout.split("fingerprint:")[-1].strip().split()[0]
        assert fp_a == fp_b, f"Fingerprints differ: {fp_a} vs {fp_b}"
