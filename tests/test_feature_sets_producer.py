"""Tests for hft_ops.feature_sets.producer (Phase 4 Batch 4b).

The producer bridges hft-ops → hft-feature-evaluator. These tests
exercise:
- The soft failure mode when the evaluator is not installed (no-op
  for venvs that do not ship the evaluator).
- The end-to-end workflow via a synthetic evaluator stub (monkey-
  patches the lazy imports inside ``produce_feature_set``).
- Error paths: empty selection, missing profile hash, hash propagation.
"""

from __future__ import annotations

import sys
import textwrap
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hft_ops.feature_sets.producer import (
    EvaluatorNotInstalled,
    NoFeaturesSelectedError,
    produce_feature_set,
)
from hft_ops.paths import PipelinePaths


# ---------------------------------------------------------------------------
# Synthetic evaluator stub used by the end-to-end test
# ---------------------------------------------------------------------------


@dataclass
class _StubProfile:
    feature_name: str
    feature_index: int


@dataclass
class _StubSchema:
    feature_count: int = 98
    contract_version: str = "2.2"


@dataclass
class _StubLoader:
    schema: _StubSchema = field(default_factory=_StubSchema)


@dataclass
class _StubSelectionCriteria:
    """Mirrors the evaluator dataclass shape the producer consumes."""
    name: str = "default"
    criteria_schema_version: str = "1.0"
    min_passing_paths: int = 1

    @classmethod
    def from_yaml(cls, path: Any) -> "_StubSelectionCriteria":
        # Minimal parser — tests just need a valid instance.
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if "criteria" in d and len(d) == 1:
            d = d["criteria"]
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


class _StubEvaluationPipeline:
    """Minimal pipeline stub that mimics the run_v2 + last_profile_hash
    contract from Phase 4 Batch 4a."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.loader = _StubLoader()
        self._last_profile_hash: str | None = None

    @property
    def last_profile_hash(self) -> str | None:
        return self._last_profile_hash

    def run_v2(self) -> dict[str, _StubProfile]:
        profiles = {
            "alpha": _StubProfile("alpha", 5),
            "beta": _StubProfile("beta", 12),
            "gamma": _StubProfile("gamma", 0),
        }
        # Mimic Phase 4 Batch 4a contract: populate hash before return.
        self._last_profile_hash = "a" * 64
        return profiles


@dataclass
class _StubEvaluationConfig:
    export_dir: str

    @classmethod
    def from_yaml(cls, path: str) -> "_StubEvaluationConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(export_dir=d.get("export_dir", "data/exports/fake"))


def _stub_select_features(
    profiles: dict[str, _StubProfile],
    criteria: _StubSelectionCriteria,
) -> list[str]:
    """Stub that returns all names (sorted) for min_passing_paths==1,
    empty for min_passing_paths > len(profiles)."""
    if criteria.min_passing_paths > len(profiles):
        return []
    return sorted(profiles)


@pytest.fixture
def stub_evaluator_modules(monkeypatch):
    """Install synthetic hft_evaluator.{config,criteria,pipeline} modules.

    Only the symbols the producer imports are replaced. Real evaluator
    install state is not affected — monkeypatch restores on teardown.
    """
    fake_config = types.ModuleType("hft_evaluator.config")
    fake_config.EvaluationConfig = _StubEvaluationConfig

    fake_criteria = types.ModuleType("hft_evaluator.criteria")
    fake_criteria.SelectionCriteria = _StubSelectionCriteria
    fake_criteria.select_features = _stub_select_features

    fake_pipeline = types.ModuleType("hft_evaluator.pipeline")
    fake_pipeline.EvaluationPipeline = _StubEvaluationPipeline

    fake_parent = types.ModuleType("hft_evaluator")
    fake_parent.config = fake_config
    fake_parent.criteria = fake_criteria
    fake_parent.pipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "hft_evaluator", fake_parent)
    monkeypatch.setitem(sys.modules, "hft_evaluator.config", fake_config)
    monkeypatch.setitem(sys.modules, "hft_evaluator.criteria", fake_criteria)
    monkeypatch.setitem(sys.modules, "hft_evaluator.pipeline", fake_pipeline)
    yield


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n")
    return path


@pytest.fixture
def pipeline_root(tmp_path, monkeypatch):
    """Build a minimal pipeline-root fixture with the directories the
    producer touches (contracts/, data/exports/fake)."""
    root = tmp_path / "pipeline"
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "pipeline_contract.toml").write_text("# stub\n")
    (root / "data" / "exports" / "fake").mkdir(parents=True)
    # Drop a sentinel file inside the export dir so hash_directory_manifest
    # has something to hash.
    (root / "data" / "exports" / "fake" / "sentinel.bin").write_bytes(b"x")
    return PipelinePaths(pipeline_root=root)


# ---------------------------------------------------------------------------
# EvaluatorNotInstalled soft failure
# ---------------------------------------------------------------------------


class TestEvaluatorNotInstalled:
    def test_raises_clear_error_when_evaluator_missing(
        self, tmp_path, monkeypatch, pipeline_root
    ):
        # Simulate missing evaluator by blocking imports.
        for mod in (
            "hft_evaluator",
            "hft_evaluator.config",
            "hft_evaluator.criteria",
            "hft_evaluator.pipeline",
        ):
            monkeypatch.setitem(sys.modules, mod, None)

        eval_cfg = _write_yaml(tmp_path / "eval.yaml", "export_dir: data/exports/fake")
        crit_yaml = _write_yaml(tmp_path / "crit.yaml", "name: test")

        with pytest.raises(EvaluatorNotInstalled, match="not installed"):
            produce_feature_set(
                evaluator_config_path=eval_cfg,
                criteria_yaml_path=crit_yaml,
                name="test_v1",
                applies_to_assets=["NVDA"],
                applies_to_horizons=[10],
                pipeline_paths=pipeline_root,
            )

    def test_error_subclasses_importerror(self):
        assert issubclass(EvaluatorNotInstalled, ImportError)


# ---------------------------------------------------------------------------
# End-to-end with stub evaluator
# ---------------------------------------------------------------------------


class TestProduceEndToEnd:
    def test_produces_valid_feature_set(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        eval_cfg = _write_yaml(
            tmp_path / "eval.yaml",
            "export_dir: data/exports/fake",
        )
        crit_yaml = _write_yaml(
            tmp_path / "crit.yaml",
            """
            name: test_criteria
            min_passing_paths: 1
            """,
        )

        fs = produce_feature_set(
            evaluator_config_path=eval_cfg,
            criteria_yaml_path=crit_yaml,
            name="test_v1",
            applies_to_assets=["NVDA"],
            applies_to_horizons=[10, 60],
            pipeline_paths=pipeline_root,
            description="End-to-end test",
            notes="stub-produced",
            created_by="pytest",
        )
        fs.verify_integrity()

        assert fs.name == "test_v1"
        assert fs.feature_indices == (0, 5, 12)  # gamma=0, alpha=5, beta=12
        assert fs.feature_names == ("gamma", "alpha", "beta")  # index-order
        assert fs.applies_to.assets == ("NVDA",)
        assert fs.applies_to.horizons == (10, 60)
        assert fs.description == "End-to-end test"
        assert fs.source_feature_count == 98
        assert fs.contract_version == "2.2"
        assert fs.produced_by.source_profile_hash == "a" * 64
        assert fs.produced_by.tool == "hft-feature-evaluator"
        assert fs.criteria_schema_version == "1.0"

    def test_raises_when_no_features_selected(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        eval_cfg = _write_yaml(
            tmp_path / "eval.yaml",
            "export_dir: data/exports/fake",
        )
        crit_yaml = _write_yaml(
            tmp_path / "crit.yaml",
            """
            name: impossible
            min_passing_paths: 99
            """,
        )

        with pytest.raises(NoFeaturesSelectedError, match="zero features"):
            produce_feature_set(
                evaluator_config_path=eval_cfg,
                criteria_yaml_path=crit_yaml,
                name="empty_v1",
                applies_to_assets=["NVDA"],
                applies_to_horizons=[10],
                pipeline_paths=pipeline_root,
            )

    def test_raises_when_config_missing(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        crit_yaml = _write_yaml(tmp_path / "crit.yaml", "name: test")
        with pytest.raises(FileNotFoundError, match="Evaluator config"):
            produce_feature_set(
                evaluator_config_path=tmp_path / "missing.yaml",
                criteria_yaml_path=crit_yaml,
                name="test_v1",
                applies_to_assets=["NVDA"],
                applies_to_horizons=[10],
                pipeline_paths=pipeline_root,
            )

    def test_raises_when_criteria_missing(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        eval_cfg = _write_yaml(
            tmp_path / "eval.yaml",
            "export_dir: data/exports/fake",
        )
        with pytest.raises(FileNotFoundError, match="Criteria YAML"):
            produce_feature_set(
                evaluator_config_path=eval_cfg,
                criteria_yaml_path=tmp_path / "missing.yaml",
                name="test_v1",
                applies_to_assets=["NVDA"],
                applies_to_horizons=[10],
                pipeline_paths=pipeline_root,
            )

    def test_feature_set_passes_integrity_verification(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        eval_cfg = _write_yaml(tmp_path / "eval.yaml", "export_dir: data/exports/fake")
        crit_yaml = _write_yaml(tmp_path / "crit.yaml", "name: test")

        fs = produce_feature_set(
            evaluator_config_path=eval_cfg,
            criteria_yaml_path=crit_yaml,
            name="integrity_v1",
            applies_to_assets=["NVDA"],
            applies_to_horizons=[10],
            pipeline_paths=pipeline_root,
        )
        fs.verify_integrity()  # No exception

    def test_provenance_paths_are_relative_to_pipeline_root(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        # Write configs INSIDE pipeline_root so relative path works.
        configs_dir = pipeline_root.pipeline_root / "configs"
        configs_dir.mkdir()
        eval_cfg = _write_yaml(
            configs_dir / "eval.yaml",
            "export_dir: data/exports/fake",
        )
        crit_yaml = _write_yaml(configs_dir / "crit.yaml", "name: test")

        fs = produce_feature_set(
            evaluator_config_path=eval_cfg,
            criteria_yaml_path=crit_yaml,
            name="rel_v1",
            applies_to_assets=["NVDA"],
            applies_to_horizons=[10],
            pipeline_paths=pipeline_root,
        )
        # Paths should be relativized (no absolute prefix).
        assert not fs.produced_by.config_path.startswith("/")
        assert not fs.produced_by.data_export.startswith("/")
        assert "configs/eval.yaml" in fs.produced_by.config_path

    def test_different_criteria_same_indices_same_hash(
        self, tmp_path, pipeline_root, stub_evaluator_modules
    ):
        # PRODUCT-only hash: changing criteria (but producing same
        # indices) must yield IDENTICAL content_hash. This locks R1.
        eval_cfg = _write_yaml(tmp_path / "eval.yaml", "export_dir: data/exports/fake")
        crit_a = _write_yaml(
            tmp_path / "a.yaml",
            "name: alpha\nmin_passing_paths: 1",
        )
        crit_b = _write_yaml(
            tmp_path / "b.yaml",
            "name: beta\nmin_passing_paths: 1",
        )
        fs_a = produce_feature_set(
            evaluator_config_path=eval_cfg,
            criteria_yaml_path=crit_a,
            name="a_v1",
            applies_to_assets=["NVDA"],
            applies_to_horizons=[10],
            pipeline_paths=pipeline_root,
        )
        fs_b = produce_feature_set(
            evaluator_config_path=eval_cfg,
            criteria_yaml_path=crit_b,
            name="b_v1",
            applies_to_assets=["NVDA"],
            applies_to_horizons=[10],
            pipeline_paths=pipeline_root,
        )
        # Same product (all 3 features) → same hash despite different
        # criteria names. This is the load-bearing R1 invariant.
        assert fs_a.content_hash == fs_b.content_hash
        # Name differs (user-assigned identifiers).
        assert fs_a.name != fs_b.name
        # Criteria differ (recipe metadata) — stored but NOT hashed.
        assert fs_a.criteria != fs_b.criteria
