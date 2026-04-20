"""Shared test fixtures for hft-ops tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.fixture(autouse=True)
def _clean_strict_index_env(monkeypatch):
    """Phase 8B MUST-FIX (Agent 3): autouse fixture that clears
    ``HFT_OPS_STRICT_INDEX`` and ``CI`` env vars at the START of every
    hft-ops test. Without this, a test that sets ``CI=true`` via
    monkeypatch (for strict-mode coverage) could leak state into the
    next test if pytest runs them in the same process and the test-
    specific monkeypatch tears down incorrectly. Also protects against
    the case where a developer runs hft-ops tests inside a CI-like
    shell (local dev with ``CI=true`` in their shell profile).

    The cleanup is deliberately AUTOUSE so every test starts in a known
    non-strict state; individual tests that WANT strict mode use their
    own ``monkeypatch.setenv("CI", "true")`` and that explicit set
    supersedes this fixture within their scope.
    """
    monkeypatch.delenv("HFT_OPS_STRICT_INDEX", raising=False)
    monkeypatch.delenv("CI", raising=False)
    yield


@pytest.fixture
def tmp_pipeline(tmp_path: Path) -> Path:
    """Create a minimal mock pipeline directory structure for testing."""
    root = tmp_path / "HFT-pipeline-v2"
    root.mkdir()

    contracts = root / "contracts"
    contracts.mkdir()
    (contracts / "pipeline_contract.toml").write_text(
        '[contract]\nschema_version = "2.2"\n'
    )

    hft_contracts = root / "hft-contracts"
    hft_contracts.mkdir()

    for module in [
        "feature-extractor-MBO-LOB",
        "MBO-LOB-analyzer",
        "lob-dataset-analyzer",
        "lob-model-trainer",
        "lob-backtester",
        "lob-models",
        "MBO-LOB-reconstructor",
    ]:
        mod_dir = root / module
        mod_dir.mkdir()
        scripts_dir = mod_dir / "scripts"
        scripts_dir.mkdir()
        configs_dir = mod_dir / "configs"
        configs_dir.mkdir()

    (root / "lob-dataset-analyzer" / "scripts" / "run_analysis.py").write_text(
        "# stub\n"
    )
    (root / "MBO-LOB-analyzer" / "scripts" / "run_analysis.py").write_text(
        "# stub\n"
    )
    (root / "lob-model-trainer" / "scripts" / "train.py").write_text(
        "# stub\n"
    )
    (root / "lob-backtester" / "scripts" / "backtest_deeplob.py").write_text(
        "# stub\n"
    )

    hft_ops = root / "hft-ops"
    hft_ops.mkdir()
    (hft_ops / "experiments").mkdir()
    (hft_ops / "ledger").mkdir()

    return root


@pytest.fixture
def sample_extractor_toml(tmp_pipeline: Path) -> Path:
    """Create a sample extractor TOML config."""
    content = """\
[experiment]
name = "Test Export"
version = "1.0.0"
tags = ["test"]

[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
tick_size = 0.01

[data]
input_dir = "../data/NVDA"
output_dir = "../data/exports/nvda_test"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true
mbo_window_size = 1000

[sampling]
strategy = "event_based"
event_count = 1000

[sequence]
window_size = 100
stride = 10
max_buffer_size = 50000

[labels]
strategy = "triple_barrier"
max_horizons = [50, 100, 200]
profit_target_pct = 0.002
stop_loss_pct = 0.001

[split]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
"""
    config_path = (
        tmp_pipeline / "feature-extractor-MBO-LOB" / "configs" / "test.toml"
    )
    config_path.write_text(content)
    return config_path


@pytest.fixture
def sample_trainer_yaml(tmp_pipeline: Path) -> Path:
    """Create a sample trainer YAML config."""
    config: Dict[str, Any] = {
        "name": "Test Training",
        "data": {
            "data_dir": "../data/exports/nvda_test",
            "feature_count": 98,
            "labeling_strategy": "triple_barrier",
            "num_classes": 3,
            "horizon_idx": 1,
            "sequence": {
                "window_size": 100,
                "stride": 10,
            },
            "normalization": {
                "strategy": "zscore_per_day",
                "eps": 1e-8,
                "clip_value": 10.0,
            },
        },
        "model": {
            "model_type": "tlob",
            "input_size": 98,
            "num_classes": 3,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 64,
            "learning_rate": 0.0001,
            "epochs": 50,
            "early_stopping_patience": 10,
        },
        "output_dir": "outputs/test",
        "tags": ["test"],
    }
    config_path = (
        tmp_pipeline
        / "lob-model-trainer"
        / "configs"
        / "experiments"
        / "test.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


@pytest.fixture
def sample_manifest_yaml(
    tmp_pipeline: Path,
    sample_extractor_toml: Path,
    sample_trainer_yaml: Path,
) -> Path:
    """Create a sample experiment manifest YAML."""
    ext_rel = sample_extractor_toml.relative_to(tmp_pipeline)
    train_rel = sample_trainer_yaml.relative_to(tmp_pipeline)

    manifest: Dict[str, Any] = {
        "experiment": {
            "name": "test_experiment",
            "description": "Test experiment for unit tests",
            "hypothesis": "Testing the orchestrator works",
            "contract_version": "2.2",
            "tags": ["test", "nvda"],
        },
        "pipeline_root": "..",
        "stages": {
            "extraction": {
                "enabled": True,
                "skip_if_exists": True,
                "config": str(ext_rel),
                "output_dir": "data/exports/nvda_test",
            },
            "raw_analysis": {
                "enabled": False,
            },
            "dataset_analysis": {
                "enabled": True,
                "profile": "quick",
                "split": "train",
            },
            "training": {
                "enabled": True,
                "config": str(train_rel),
                "overrides": {
                    "data.data_dir": "${stages.extraction.output_dir}",
                },
                "horizon_value": 100,
                "output_dir": "hft-ops/ledger/runs/${experiment.name}_${timestamp}",
            },
            "backtesting": {
                "enabled": True,
                "model_checkpoint": "${stages.training.output_dir}/checkpoints/best.pt",
                "data_dir": "${stages.extraction.output_dir}",
                "horizon_idx": "${resolved.horizon_idx}",
                "params": {
                    "initial_capital": 100000,
                    "position_size": 0.1,
                    "spread_bps": 1.0,
                    "slippage_bps": 0.5,
                },
            },
        },
    }

    manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "test.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)
    return manifest_path
