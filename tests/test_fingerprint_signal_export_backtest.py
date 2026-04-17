"""Phase 5 FULL-A Block 2 regression guards: fingerprint coverage extension.

Locks the Phase 5 FULL-A (2026-04-17) additions to ``compute_fingerprint``:
  * signal_export.script / split / extra_args hashed when NON-DEFAULT
  * backtesting.script / extra_args hashed when NON-DEFAULT
  * validation still excluded (observation, not treatment)
  * **BACK-COMPAT GUARD**: manifests using default signal_export/backtesting
    (which is all existing production manifests today) produce IDENTICAL
    fingerprints pre/post this block — new `components["signal_export"]` /
    `components["backtest_script"]` keys only appear when a researcher
    overrides defaults.

See ``/Users/knight/.claude/plans/gentle-brewing-quail.md`` Phase 5 FULL-A
Block 2 + CRITICAL-FIX 1 for rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hft_ops.ledger.dedup import compute_fingerprint, compute_fingerprint_explain
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


_REAL_PIPELINE_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_manifest(
    exp_dir: Path,
    name: str,
    signal_export_overrides: Dict[str, Any] | None = None,
    backtesting_overrides: Dict[str, Any] | None = None,
    validation_overrides: Dict[str, Any] | None = None,
) -> Path:
    """Write a minimal manifest for fingerprint coverage tests.

    Training stage is enabled with a minimal inline trainer_config so
    compute_fingerprint exercises the normal training path; the stages under
    test (signal_export, backtesting, validation) get overrides merged in.
    """
    signal_export = {"enabled": False}
    if signal_export_overrides is not None:
        signal_export.update(signal_export_overrides)

    backtesting = {
        "enabled": True,  # default True per schema.py
    }
    if backtesting_overrides is not None:
        backtesting.update(backtesting_overrides)

    validation = {"enabled": False}
    if validation_overrides is not None:
        validation.update(validation_overrides)

    manifest = {
        "experiment": {"name": name, "contract_version": "2.2"},
        "pipeline_root": "..",
        "stages": {
            "extraction": {"enabled": False, "output_dir": "data/exports/fake"},
            "dataset_analysis": {"enabled": False},
            "validation": validation,
            "training": {
                "enabled": True,
                "output_dir": f"outputs/{name}",
                "trainer_config": {
                    "data": {"feature_count": 98},
                    "model": {"model_type": "tlob"},
                    "train": {"batch_size": 128, "epochs": 30, "seed": 42},
                },
            },
            "signal_export": signal_export,
            "backtesting": backtesting,
        },
    }
    path = exp_dir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return path


@pytest.fixture
def scratch(tmp_path: Path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    paths = PipelinePaths(pipeline_root=_REAL_PIPELINE_ROOT)
    yield paths, exp_dir


# -----------------------------------------------------------------------------
# CRITICAL-FIX 1: Back-compat guard — default script → no fp change
# -----------------------------------------------------------------------------


class TestBackCompatInvariant:
    """Default-script + no extra_args manifests produce unchanged fingerprints.

    This guards against the class of bug where adding fingerprint coverage
    silently re-fingerprints every existing ledger record.
    """

    def test_default_backtesting_script_no_extra_args_produces_no_backtest_script_key(self, scratch):
        """The most common case: backtesting.enabled=True with default script.

        No `components["backtest_script"]` key should be added — only the
        existing `components["backtest"]` (asdict of params) contributes.
        """
        paths, exp_dir = scratch
        m = _write_manifest(exp_dir, "default_bt",
                            backtesting_overrides={"script": "scripts/backtest_deeplob.py"})
        fp, components = compute_fingerprint_explain(load_manifest(m), paths)
        assert "backtest" in components, "existing backtest-params hash should still be there"
        assert "backtest_script" not in components, (
            "default backtest script + empty extra_args should NOT contribute a "
            "new fingerprint component (back-compat invariant)"
        )

    def test_signal_export_disabled_produces_no_key(self, scratch):
        paths, exp_dir = scratch
        m = _write_manifest(exp_dir, "no_se")
        fp, components = compute_fingerprint_explain(load_manifest(m), paths)
        assert "signal_export" not in components, (
            "signal_export.enabled=False MUST NOT contribute to fingerprint"
        )


# -----------------------------------------------------------------------------
# Block 2 primary contract: non-default values flip the fp
# -----------------------------------------------------------------------------


class TestSignalExportCoverage:
    def test_non_default_signal_export_script_flips_fingerprint(self, scratch):
        """A sweep axis on signal_export.script must produce distinct fingerprints."""
        paths, exp_dir = scratch
        m1 = _write_manifest(exp_dir, "se_default",
                             signal_export_overrides={"enabled": True,
                                                      "script": "scripts/export_signals.py"})
        m2 = _write_manifest(exp_dir, "se_hmhp",
                             signal_export_overrides={"enabled": True,
                                                      "script": "scripts/export_hmhp_signals.py"})
        fp1, c1 = compute_fingerprint_explain(load_manifest(m1), paths)
        fp2, c2 = compute_fingerprint_explain(load_manifest(m2), paths)
        # m1 uses default → no signal_export key
        assert "signal_export" not in c1
        # m2 uses non-default → key present
        assert "signal_export" in c2
        assert c2["signal_export"].get("script") == "scripts/export_hmhp_signals.py"
        assert fp1 != fp2, "distinct signal_export.script MUST produce distinct fingerprints"

    def test_signal_export_extra_args_flips_fingerprint(self, scratch):
        paths, exp_dir = scratch
        m1 = _write_manifest(exp_dir, "se_no_args",
                             signal_export_overrides={"enabled": True})
        m2 = _write_manifest(exp_dir, "se_calibrate",
                             signal_export_overrides={"enabled": True,
                                                      "extra_args": ["--calibrate", "variance_match"]})
        fp1, _ = compute_fingerprint_explain(load_manifest(m1), paths)
        fp2, c2 = compute_fingerprint_explain(load_manifest(m2), paths)
        assert "signal_export" in c2
        assert c2["signal_export"]["extra_args"] == ["--calibrate", "variance_match"]
        assert fp1 != fp2

    def test_signal_export_non_default_split_flips_fingerprint(self, scratch):
        """Default split is 'test'; non-default splits (val, train) flip fp.

        Default split + default script + no extra_args → no signal_export key
        (back-compat — existing HMHP manifests with enabled=True and default
        split remain semantically identical for fingerprinting).
        """
        paths, exp_dir = scratch
        m_test = _write_manifest(exp_dir, "se_test_split",
                                 signal_export_overrides={"enabled": True, "split": "test"})
        m_val = _write_manifest(exp_dir, "se_val_split",
                                signal_export_overrides={"enabled": True, "split": "val"})
        fp_test, c_test = compute_fingerprint_explain(load_manifest(m_test), paths)
        fp_val, c_val = compute_fingerprint_explain(load_manifest(m_val), paths)
        assert "signal_export" not in c_test, (
            "default split='test' with default script → no signal_export key (back-compat)"
        )
        assert "signal_export" in c_val
        assert c_val["signal_export"].get("split") == "val"
        assert fp_test != fp_val


class TestBacktestScriptCoverage:
    def test_non_default_backtest_script_flips_fingerprint(self, scratch):
        paths, exp_dir = scratch
        m1 = _write_manifest(exp_dir, "bt_default",
                             backtesting_overrides={"script": "scripts/backtest_deeplob.py"})
        m2 = _write_manifest(exp_dir, "bt_regression",
                             backtesting_overrides={"script": "scripts/run_regression_backtest.py"})
        fp1, c1 = compute_fingerprint_explain(load_manifest(m1), paths)
        fp2, c2 = compute_fingerprint_explain(load_manifest(m2), paths)
        assert "backtest_script" not in c1, "default backtest script → no new component"
        assert "backtest_script" in c2
        assert c2["backtest_script"].get("script") == "scripts/run_regression_backtest.py"
        assert fp1 != fp2

    def test_backtest_extra_args_flips_fingerprint(self, scratch):
        paths, exp_dir = scratch
        m1 = _write_manifest(exp_dir, "bt_no_args", backtesting_overrides={})
        m2 = _write_manifest(exp_dir, "bt_with_args",
                             backtesting_overrides={"extra_args": ["--threshold", "2.5"]})
        fp1, c1 = compute_fingerprint_explain(load_manifest(m1), paths)
        fp2, c2 = compute_fingerprint_explain(load_manifest(m2), paths)
        assert "backtest_script" not in c1
        assert "backtest_script" in c2
        assert c2["backtest_script"]["extra_args"] == ["--threshold", "2.5"]
        assert fp1 != fp2


class TestValidationStillExcluded:
    """Confirms Phase 5 FULL-A Block 2 did NOT change validation exclusion
    (validation.min_ic mutation must still produce identical fingerprint).
    """

    def test_validation_min_ic_mutation_preserves_fingerprint(self, scratch):
        paths, exp_dir = scratch
        m1 = _write_manifest(exp_dir, "val_05",
                             validation_overrides={"enabled": True, "min_ic": 0.05})
        m2 = _write_manifest(exp_dir, "val_03",
                             validation_overrides={"enabled": True, "min_ic": 0.03})
        fp1, _ = compute_fingerprint_explain(load_manifest(m1), paths)
        fp2, _ = compute_fingerprint_explain(load_manifest(m2), paths)
        assert fp1 == fp2, (
            "validation.min_ic is an OBSERVATION threshold, NOT a treatment — "
            "it MUST NOT influence the fingerprint (§rationale in dedup.py L684-688)"
        )
