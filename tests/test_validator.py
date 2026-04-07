"""Tests for manifest validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.validator import (
    ValidationResult,
    _compute_feature_count,
    validate_manifest,
)
from hft_ops.paths import PipelinePaths


class TestComputeFeatureCount:
    def test_lob_only(self):
        cfg = {"lob_levels": 10}
        assert _compute_feature_count(cfg) == 40

    def test_all_features(self):
        cfg = {
            "lob_levels": 10,
            "include_derived": True,
            "include_mbo": True,
            "include_signals": True,
        }
        assert _compute_feature_count(cfg) == 98  # 40 + 8 + 36 + 14

    def test_no_signals(self):
        cfg = {
            "lob_levels": 10,
            "include_derived": True,
            "include_mbo": True,
            "include_signals": False,
        }
        assert _compute_feature_count(cfg) == 84  # 40 + 8 + 36

    def test_five_levels(self):
        cfg = {"lob_levels": 5}
        assert _compute_feature_count(cfg) == 20


class TestValidateManifest:
    def test_valid_manifest(
        self, sample_manifest_yaml: Path, tmp_pipeline: Path
    ):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)
        result = validate_manifest(manifest, paths)

        errors = [str(e) for e in result.errors]
        assert result.is_valid, f"Expected valid, got errors: {errors}"

    def test_missing_extractor_config(self, tmp_pipeline: Path):
        manifest_data = {
            "experiment": {
                "name": "bad_test",
                "contract_version": "2.2",
            },
            "stages": {
                "extraction": {
                    "enabled": True,
                    "config": "feature-extractor-MBO-LOB/configs/nonexistent.toml",
                },
                "training": {"enabled": False},
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "bad.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f)

        manifest = load_manifest(manifest_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        result = validate_manifest(manifest, paths)

        assert not result.is_valid
        assert any("not found" in str(e) for e in result.errors)

    def test_feature_count_mismatch(
        self,
        sample_manifest_yaml: Path,
        sample_trainer_yaml: Path,
        tmp_pipeline: Path,
    ):
        """Trainer says feature_count=50, extractor computes 98 => error."""
        with open(sample_trainer_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["feature_count"] = 50
        cfg["model"]["input_size"] = 50
        with open(sample_trainer_yaml, "w") as f:
            yaml.dump(cfg, f)

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)
        result = validate_manifest(manifest, paths)

        assert not result.is_valid
        error_msgs = [str(e) for e in result.errors]
        assert any("feature_count" in msg.lower() or "Feature count" in msg for msg in error_msgs)

    def test_window_size_mismatch(
        self,
        sample_manifest_yaml: Path,
        sample_trainer_yaml: Path,
        tmp_pipeline: Path,
    ):
        with open(sample_trainer_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["sequence"]["window_size"] = 200  # extractor has 100
        with open(sample_trainer_yaml, "w") as f:
            yaml.dump(cfg, f)

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)
        result = validate_manifest(manifest, paths)

        assert not result.is_valid
        assert any("window_size" in str(e) for e in result.errors)

    def test_horizon_resolution(
        self,
        sample_manifest_yaml: Path,
        tmp_pipeline: Path,
    ):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(sample_manifest_yaml)
        assert manifest.stages.training.horizon_value == 100

        result = validate_manifest(manifest, paths)
        assert result.is_valid

    def test_horizon_not_found(
        self,
        sample_manifest_yaml: Path,
        sample_extractor_toml: Path,
        tmp_pipeline: Path,
    ):
        """horizon_value=999 not in extractor horizons => error."""
        manifest_data = {
            "experiment": {
                "name": "bad_horizon",
                "contract_version": "2.2",
            },
            "stages": {
                "extraction": {
                    "enabled": True,
                    "config": str(
                        sample_extractor_toml.relative_to(tmp_pipeline)
                    ),
                },
                "training": {
                    "enabled": True,
                    "config": "lob-model-trainer/configs/experiments/test.yaml",
                    "horizon_value": 999,
                },
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "bad_hz.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f)

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(manifest_path)
        result = validate_manifest(manifest, paths)

        assert not result.is_valid
        assert any("horizon_value=999" in str(e) for e in result.errors)

    def test_training_without_data_source(self, tmp_pipeline: Path):
        manifest_data = {
            "experiment": {"name": "no_data"},
            "stages": {
                "extraction": {
                    "enabled": False,
                    "output_dir": "",
                },
                "training": {"enabled": True, "config": ""},
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "no_data.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f)

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        manifest = load_manifest(manifest_path)
        result = validate_manifest(manifest, paths)

        assert not result.is_valid


class TestValidationResult:
    def test_empty_is_valid(self):
        r = ValidationResult()
        assert r.is_valid
        assert len(r.errors) == 0
        assert len(r.warnings) == 0

    def test_warning_still_valid(self):
        r = ValidationResult()
        r.warning("minor issue")
        assert r.is_valid
        assert len(r.warnings) == 1

    def test_error_invalid(self):
        r = ValidationResult()
        r.error("critical issue")
        assert not r.is_valid
        assert len(r.errors) == 1
