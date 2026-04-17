"""ValidationRunner integration tests.

Covers the 5-scenario policy matrix from the plan:
- (a) PASS → training proceeds (status COMPLETED, captured_metrics populated)
- (b) FAIL + on_fail=warn (DEFAULT) → COMPLETED with warning, pipeline continues
- (c) FAIL + on_fail=abort → FAILED, pipeline stops
- (d) PASS + allow_zero_ic_names bypass works
- (e) enabled=false → stage skipped by CLI (not the runner itself)

Also covers fingerprint stability: changes to validation config must NOT
change the experiment fingerprint (validation is an observation, not a
treatment).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml


N_TIMESTEPS = 5
HORIZONS = [1, 5, 10]


def _write_synthetic_export(
    export_dir: Path,
    *,
    n_days: int = 8,
    n_seqs_per_day: int = 500,
    signal_features: tuple = (0, 3),
    signal_strength: float = 0.6,
    label_std_bps: float = 10.0,
) -> None:
    """Build a synthetic off-exchange export inside export_dir.

    Mirrors the fast_gate test harness. Features 0 and 3 carry signal, all
    others are noise. Categorical indices {29, 30, 32, 33} are constants
    per the off-exchange contract.
    """
    train_dir = export_dir / "train"
    train_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_features = 34

    for d_i in range(n_days):
        date = f"2026-01-{d_i + 1:02d}"
        labels = rng.normal(0.0, label_std_bps, size=(n_seqs_per_day, len(HORIZONS)))
        sequences = rng.normal(
            0.0, 1.0, size=(n_seqs_per_day, N_TIMESTEPS, n_features)
        ).astype(np.float32)

        for t in range(N_TIMESTEPS):
            for f_idx in signal_features:
                noise = rng.normal(0, 1.0, size=n_seqs_per_day)
                sequences[:, t, f_idx] = (
                    signal_strength * labels[:, 0] + noise
                ).astype(np.float32)

        # Categoricals
        for cat in (29, 30, 33):
            sequences[:, :, cat] = 1.0
        sequences[:, :, 32] = float(d_i % 4)

        np.save(train_dir / f"{date}_sequences.npy", sequences)
        np.save(train_dir / f"{date}_labels.npy", labels.astype(np.float64))
        md = {
            "day": date,
            "n_sequences": n_seqs_per_day,
            "window_size": N_TIMESTEPS,
            "n_features": n_features,
            "schema_version": "1.0",
            "contract_version": "off_exchange_1.0",
            "label_strategy": "point_return",
            "label_encoding": "continuous_bps",
            "horizons": HORIZONS,
            "bin_size_seconds": 60,
            "normalization": {
                "strategy": "per_day_zscore",
                "applied": False,
                "params_file": f"{date}_normalization.json",
            },
            "provenance": {"processor_version": "0.1.0"},
            "export_timestamp": "2026-04-14T00:00:00Z",
        }
        with open(train_dir / f"{date}_metadata.json", "w") as f:
            json.dump(md, f)


def _make_manifest(
    tmp_pipeline: Path,
    export_rel_path: str,
    *,
    on_fail: str = "warn",
    min_ic: float = 0.05,
    min_ic_count: int = 2,
    min_stability: float = 2.0,
    min_return_std_bps: float = 5.0,
    allow_zero_ic_names=None,
    enabled: bool = True,
    target_horizon: str = "1",
) -> Path:
    allow_zero_ic_names = allow_zero_ic_names or []
    manifest_dict = {
        "experiment": {
            "name": "validation_test",
            "contract_version": "2.2",
        },
        "pipeline_root": "..",
        "stages": {
            "extraction": {
                "enabled": False,
                # Export already exists; validation reads from output_dir
                "output_dir": export_rel_path,
            },
            "dataset_analysis": {"enabled": False},
            "validation": {
                "enabled": enabled,
                "on_fail": on_fail,
                "target_horizon": target_horizon,
                "min_ic": min_ic,
                "min_ic_count": min_ic_count,
                "min_stability": min_stability,
                "min_return_std_bps": min_return_std_bps,
                "sample_size": 3000,
                "n_folds": 4,
                "allow_zero_ic_names": allow_zero_ic_names,
            },
            "training": {"enabled": False},
            "backtesting": {"enabled": False},
        },
    }
    path = tmp_pipeline / "hft-ops" / "experiments" / "val_test.yaml"
    with open(path, "w") as f:
        yaml.dump(manifest_dict, f)
    return path


@pytest.fixture
def synthetic_ops_env(tmp_pipeline: Path):
    """tmp_pipeline + a synthetic export at data/exports/val_test/."""
    export_dir = tmp_pipeline / "data" / "exports" / "val_test"
    _write_synthetic_export(export_dir, signal_strength=0.6)
    return tmp_pipeline, "data/exports/val_test"


# ---------------------------------------------------------------------------
# Scenario (a): strong signal → PASS → training proceeds
# ---------------------------------------------------------------------------


class TestValidationPassScenario:
    def test_pass_captures_metrics(self, synthetic_ops_env):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        tmp_pipeline, rel = synthetic_ops_env
        m_path = _make_manifest(tmp_pipeline, rel, on_fail="warn")
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        runner = ValidationRunner()
        errors = runner.validate_inputs(manifest, ops)
        assert errors == [], f"unexpected input errors: {errors}"

        result = runner.run(manifest, ops)

        assert result.status == StageStatus.COMPLETED, (
            f"strong-signal export should PASS; result: {result}"
        )
        assert result.captured_metrics["validation_verdict"] == "PASS"
        assert result.captured_metrics["best_feature_ic"] > 0.05
        assert result.captured_metrics["ic_count"] >= 2
        assert np.isfinite(result.captured_metrics["stability"])
        assert "validation_report" in result.captured_metrics

        # Gate report file must exist on disk
        report_path = Path(result.captured_metrics["gate_report_path"])
        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)
        assert report["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Scenario (b): fail + warn → COMPLETED with warning
# ---------------------------------------------------------------------------


class TestValidationFailWarnScenario:
    def test_fail_warn_keeps_going(self, tmp_pipeline):
        """Zero-signal export + on_fail=warn → stage COMPLETED, warning logged."""
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        export_dir = tmp_pipeline / "data" / "exports" / "zero_sig"
        _write_synthetic_export(export_dir, signal_features=(), signal_strength=0.0)

        m_path = _make_manifest(
            tmp_pipeline,
            "data/exports/zero_sig",
            on_fail="warn",
        )
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        result = ValidationRunner().run(manifest, ops)

        # Stage status is COMPLETED (warn), but verdict is FAIL
        assert result.status == StageStatus.COMPLETED, (
            f"warn policy must not fail the stage; got {result}"
        )
        assert result.captured_metrics["validation_verdict"] == "FAIL"
        assert "[WARN]" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Scenario (c): fail + abort → FAILED stage
# ---------------------------------------------------------------------------


class TestValidationFailAbortScenario:
    def test_fail_abort_stops(self, tmp_pipeline):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        export_dir = tmp_pipeline / "data" / "exports" / "zero_sig_abort"
        _write_synthetic_export(export_dir, signal_features=(), signal_strength=0.0)

        m_path = _make_manifest(
            tmp_pipeline,
            "data/exports/zero_sig_abort",
            on_fail="abort",
        )
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        result = ValidationRunner().run(manifest, ops)

        assert result.status == StageStatus.FAILED
        assert result.captured_metrics["validation_verdict"] == "FAIL"
        assert "IC gate FAILED" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Scenario (d): record_only — always COMPLETED regardless of verdict
# ---------------------------------------------------------------------------


class TestValidationRecordOnlyScenario:
    def test_record_only_never_fails(self, tmp_pipeline):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        export_dir = tmp_pipeline / "data" / "exports" / "zero_record"
        _write_synthetic_export(export_dir, signal_features=(), signal_strength=0.0)

        m_path = _make_manifest(
            tmp_pipeline,
            "data/exports/zero_record",
            on_fail="record_only",
        )
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        result = ValidationRunner().run(manifest, ops)
        assert result.status == StageStatus.COMPLETED
        assert result.captured_metrics["validation_verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Scenario (e): bypass list excludes context features from IC count
# ---------------------------------------------------------------------------


class TestValidationBypassListScenario:
    def test_bypass_list_captured_in_report(self, synthetic_ops_env):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.validation import ValidationRunner

        tmp_pipeline, rel = synthetic_ops_env
        m_path = _make_manifest(
            tmp_pipeline,
            rel,
            allow_zero_ic_names=["dark_share", "time_regime"],
        )
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        result = ValidationRunner().run(manifest, ops)
        report = result.captured_metrics["validation_report"]
        assert "dark_share" in report["allow_zero_ic_names"]
        assert "time_regime" in report["allow_zero_ic_names"]


# ---------------------------------------------------------------------------
# Fingerprint stability: validation config must NOT affect fingerprint
# ---------------------------------------------------------------------------


class TestValidationFingerprintStability:
    """Validation is an OBSERVATION, not a TREATMENT — changing thresholds
    or on_fail policy must not change the experiment fingerprint.
    """

    def test_validation_config_change_preserves_fingerprint(
        self,
        synthetic_ops_env,
    ):
        from hft_ops.ledger.dedup import compute_fingerprint
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths

        tmp_pipeline, rel = synthetic_ops_env
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        m1 = _make_manifest(
            tmp_pipeline, rel, on_fail="warn", min_ic=0.05
        )
        manifest1 = load_manifest(m1)
        fp1 = compute_fingerprint(manifest1, paths)

        m2 = _make_manifest(
            tmp_pipeline, rel, on_fail="abort", min_ic=0.10
        )
        manifest2 = load_manifest(m2)
        fp2 = compute_fingerprint(manifest2, paths)

        assert fp1 == fp2, (
            "Validation config changes must NOT affect fingerprint. "
            f"fp1={fp1[:16]}, fp2={fp2[:16]}. "
            "Check compute_fingerprint excludes 'validation' stage."
        )

    def test_validation_enabled_flag_preserves_fingerprint(
        self,
        synthetic_ops_env,
    ):
        """Even toggling validation.enabled should not affect fingerprint."""
        from hft_ops.ledger.dedup import compute_fingerprint
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths

        tmp_pipeline, rel = synthetic_ops_env
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        m1 = _make_manifest(tmp_pipeline, rel, enabled=True)
        m2 = _make_manifest(tmp_pipeline, rel, enabled=False)

        fp1 = compute_fingerprint(load_manifest(m1), paths)
        fp2 = compute_fingerprint(load_manifest(m2), paths)

        assert fp1 == fp2


# ---------------------------------------------------------------------------
# Input validation / error paths
# ---------------------------------------------------------------------------


class TestValidationInputErrors:
    def test_bad_on_fail_rejected_at_load(self, tmp_pipeline):
        """Loader must reject unknown on_fail values fail-fast."""
        from hft_ops.manifest.loader import load_manifest

        manifest_dict = {
            "experiment": {"name": "bad", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "validation": {
                    "enabled": True,
                    "on_fail": "blow_up",  # invalid
                },
                "training": {"enabled": False},
            },
        }
        p = tmp_pipeline / "hft-ops" / "experiments" / "bad.yaml"
        with open(p, "w") as f:
            yaml.dump(manifest_dict, f)

        with pytest.raises(ValueError, match="on_fail"):
            load_manifest(p)

    def test_missing_data_dir_fails_input_validation(self, tmp_pipeline):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.validation import ValidationRunner

        manifest_dict = {
            "experiment": {"name": "no_export", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {"enabled": False, "output_dir": ""},
                "validation": {
                    "enabled": True,
                    "on_fail": "warn",
                    "target_horizon": "1",
                },
                "training": {"enabled": False},
            },
        }
        p = tmp_pipeline / "hft-ops" / "experiments" / "noex.yaml"
        with open(p, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(p)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)

        errors = ValidationRunner().validate_inputs(manifest, ops)
        assert any("output_dir" in e for e in errors)

    def test_cli_registers_validation_runner(self):
        """Sanity: cli.py must register ValidationRunner in stage_runners.

        This is the orphan-bug regression guard — pre-fix the runner existed
        only as a schema-level concept, with no runner in the CLI loop.
        """
        cli_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "hft_ops"
            / "cli.py"
        )
        source = cli_path.read_text()
        assert "ValidationRunner()" in source, (
            "cli.py must instantiate ValidationRunner() in stage_runners "
            "(orphan-bug regression check). Phase 2b fix."
        )
        # Also check it appears in both the run and sweep paths.
        assert source.count("ValidationRunner()") >= 2, (
            "ValidationRunner must be registered in BOTH `run` and `sweep run` "
            f"loops; found {source.count('ValidationRunner()')} occurrence(s)."
        )
        # Order invariant: validation must come BEFORE training.
        run_validation_idx = source.index('"validation"')
        run_training_idx = source.index('"training"')
        assert run_validation_idx < run_training_idx, (
            "validation stage must be registered BEFORE training so a "
            "failing gate under on_fail=abort prevents wasted compute."
        )

    def test_dry_run_skips(self, synthetic_ops_env):
        from hft_ops.config import OpsConfig
        from hft_ops.manifest.loader import load_manifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        tmp_pipeline, rel = synthetic_ops_env
        m_path = _make_manifest(tmp_pipeline, rel)
        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        ops = OpsConfig(paths=paths, dry_run=True, verbose=False)

        result = ValidationRunner().run(manifest, ops)
        assert result.status == StageStatus.SKIPPED
