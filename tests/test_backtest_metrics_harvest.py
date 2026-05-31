"""
S1 — backtest-metrics harvest (H6, regression path).

Tests `hft_ops.stages.backtesting._harvest_backtest_metrics`: read the
regression backtester's deterministic `<output_dir>/<NAME>.json`, reduce to the
backtester's best threshold (replicating run_regression_backtest.py:610-619),
remap PascalCase -> flat snake_case, finite-guard every numeric, fail-soft.

Fixtures are faithful to the REAL on-disk regression summary schema
(top-level `{name, exchange, zero_dte_enabled, results:[...]}`; each `results[]`
element PascalCase `TotalReturn/SharpeRatio/MaxDrawdown/WinRate` + option_*),
ground-truthed against `lob-backtester/outputs/backtests/cycle12_r20_hmhp_r__seed_42.json`
— whose best-by-option_return_pct row carries NaN sharpe/win_rate (a losing
experiment's do-nothing threshold), the case that mandates the isfinite guard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from hft_ops.stages.backtesting import _harvest_backtest_metrics

NAN = float("nan")
REGRESSION_SCRIPT = "lob-backtester/scripts/run_regression_backtest.py"
READABILITY_SCRIPT = "lob-backtester/scripts/run_readability_backtest.py"
DEEPLOB_SCRIPT = "scripts/backtest_deeplob.py"


def _threshold(label, *, total_return, sharpe, max_dd, win_rate,
               option_return_pct=None, option_win_rate=None, n_entries=100):
    """Build one regression `results[]` per-threshold dict (real key names)."""
    d = {
        "label": label,
        "min_return_bps": 8,
        "max_spread_bps": 5.0,
        "strategy_name": "regression_magnitude",
        "SharpeRatio": sharpe,
        "SortinoRatio": sharpe,
        "MaxDrawdown": max_dd,
        "CalmarRatio": 0.0,
        "TotalReturn": total_return,
        "WinRate": win_rate,
        "ProfitFactor": 1.0,
        "Expectancy": 0.0,
        "n_entries": n_entries,
        "trade_rate": 0.5,
        "avg_hold_events": 10.0,
    }
    if option_return_pct is not None:
        d["option_return_pct"] = option_return_pct
        d["option_final_equity"] = 100.0
        d["option_n_trades"] = n_entries
    if option_win_rate is not None:
        d["option_win_rate"] = option_win_rate
    return d


def _write_summary(output_dir: Path, name: str, *, zero_dte: bool, results: list) -> None:
    summary = {
        "name": name,
        "exchange": "XNAS",
        "signal_dir": "outputs/.../signals/test",
        "signal_metadata": {"horizons": [10, 60, 300]},
        "holding_policy": "fixed_horizon",
        "zero_dte_enabled": zero_dte,
        "results": results,
    }
    (output_dir / f"{name}.json").write_text(json.dumps(summary))


# --------------------------------------------------------------------------
# Regression — happy paths
# --------------------------------------------------------------------------
def test_regression_zero_dte_picks_best_by_option_return_pct(tmp_path):
    """zero_dte=True -> best threshold = max option_return_pct; profitable row
    has finite metrics -> all snake_case keys present."""
    name = "exp_profitable"
    _write_summary(tmp_path, name, zero_dte=True, results=[
        _threshold("aggressive", total_return=-0.05, sharpe=-3.0, max_dd=0.06,
                   win_rate=0.40, option_return_pct=-2.0, option_win_rate=0.45),
        _threshold("balanced", total_return=0.02, sharpe=1.5, max_dd=0.01,
                   win_rate=0.55, option_return_pct=3.5, option_win_rate=0.60),  # best
        _threshold("conservative", total_return=0.0, sharpe=NAN, max_dd=0.0,
                   win_rate=NAN, option_return_pct=0.0, option_win_rate=NAN),
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out == {
        "total_return": 0.02,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.01,
        "win_rate": 0.55,
        "option_return_pct": 3.5,
        "option_win_rate": 0.60,
    }, out


def test_regression_zero_dte_losing_experiment_nan_at_best_row(tmp_path):
    """GROUND-TRUTH (cycle12_r20_hmhp_r__seed_42): a losing experiment's best
    threshold by option_return_pct is the do-nothing 0% row, which carries NaN
    sharpe/win_rate. The isfinite guard MUST drop the NaN keys (record the
    finite 0.0 returns, omit sharpe/win_rate) — never write NaN to the ledger."""
    name = "cycle12_r20_like__seed_42"
    _write_summary(tmp_path, name, zero_dte=True, results=[
        _threshold("ultra_conv_15bps", total_return=0.0, sharpe=NAN, max_dd=0.0,
                   win_rate=NAN, option_return_pct=0.0, option_win_rate=NAN),  # best (0 > neg)
        _threshold("aggressive", total_return=-0.0594, sharpe=-25.84, max_dd=0.0611,
                   win_rate=0.3662, option_return_pct=-6.08, option_win_rate=0.4254),
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out == {"total_return": 0.0, "max_drawdown": 0.0, "option_return_pct": 0.0}, out
    # explicit: no NaN ever reaches the ledger
    assert all(math.isfinite(v) for v in out.values())
    assert "sharpe_ratio" not in out and "win_rate" not in out and "option_win_rate" not in out


def test_regression_non_zero_dte_picks_best_by_total_return_no_option_keys(tmp_path):
    """zero_dte=False -> best = max TotalReturn; NO option_* keys emitted even
    if present in the dict."""
    name = "exp_no_zero_dte"
    _write_summary(tmp_path, name, zero_dte=False, results=[
        _threshold("a", total_return=-0.01, sharpe=-1.0, max_dd=0.02, win_rate=0.45),
        _threshold("b", total_return=0.03, sharpe=2.0, max_dd=0.005, win_rate=0.58),  # best
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out == {
        "total_return": 0.03, "sharpe_ratio": 2.0, "max_drawdown": 0.005, "win_rate": 0.58,
    }, out
    assert "option_return_pct" not in out


def test_regression_omits_total_trades(tmp_path):
    """total_trades is ABSENT from regression results[] (only n_entries, a
    misleading proxy) -> harvest must NOT emit total_trades."""
    name = "exp_tt"
    _write_summary(tmp_path, name, zero_dte=False, results=[
        _threshold("a", total_return=0.01, sharpe=1.0, max_dd=0.01, win_rate=0.5, n_entries=710),
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert "total_trades" not in out
    assert out["total_return"] == 0.01


def test_regression_finite_guard_drops_only_nonfinite_keys(tmp_path):
    """A best row with mixed finite/NaN/Inf -> only finite keys survive."""
    name = "exp_mixed"
    _write_summary(tmp_path, name, zero_dte=False, results=[
        _threshold("only", total_return=0.04, sharpe=NAN, max_dd=float("inf"), win_rate=0.5),
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out == {"total_return": 0.04, "win_rate": 0.5}, out


def test_regression_best_selection_robust_to_nan_selection_key(tmp_path):
    """A threshold with NaN option_return_pct must NEVER win the best-selection
    (treat non-finite as -inf); the finite row is picked."""
    name = "exp_nan_key"
    _write_summary(tmp_path, name, zero_dte=True, results=[
        _threshold("nan_key", total_return=9.99, sharpe=9.0, max_dd=0.0,
                   win_rate=0.99, option_return_pct=NAN, option_win_rate=0.99),
        _threshold("finite", total_return=0.01, sharpe=1.0, max_dd=0.01,
                   win_rate=0.50, option_return_pct=1.0, option_win_rate=0.55),  # must win
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out["option_return_pct"] == 1.0
    assert out["total_return"] == 0.01  # the finite row, not the NaN-key 9.99 row


# --------------------------------------------------------------------------
# Regression — fail-soft
# --------------------------------------------------------------------------
def test_missing_summary_returns_empty(tmp_path):
    out = _harvest_backtest_metrics(tmp_path, "does_not_exist", REGRESSION_SCRIPT)
    assert out == {}


def test_malformed_json_returns_empty(tmp_path):
    (tmp_path / "bad.json").write_text("{not valid json,,,")
    out = _harvest_backtest_metrics(tmp_path, "bad", REGRESSION_SCRIPT)
    assert out == {}


def test_empty_results_list_returns_empty(tmp_path):
    _write_summary(tmp_path, "empty", zero_dte=True, results=[])
    out = _harvest_backtest_metrics(tmp_path, "empty", REGRESSION_SCRIPT)
    assert out == {}


def test_results_missing_key_returns_empty(tmp_path):
    (tmp_path / "noresults.json").write_text(json.dumps({"name": "x", "zero_dte_enabled": True}))
    out = _harvest_backtest_metrics(tmp_path, "noresults", REGRESSION_SCRIPT)
    assert out == {}


def test_all_nonfinite_best_row_returns_empty(tmp_path):
    name = "exp_allnan"
    _write_summary(tmp_path, name, zero_dte=False, results=[
        _threshold("a", total_return=NAN, sharpe=NAN, max_dd=NAN, win_rate=NAN),
    ])
    out = _harvest_backtest_metrics(tmp_path, name, REGRESSION_SCRIPT)
    assert out == {}


# --------------------------------------------------------------------------
# Non-regression scripts — deferred / unsupported -> {}
# --------------------------------------------------------------------------
def test_readability_script_deferred_returns_empty(tmp_path):
    """Readability (classification) harvest is a deferred follow-on -> {}."""
    out = _harvest_backtest_metrics(tmp_path, "some_classification_run", READABILITY_SCRIPT)
    assert out == {}


def test_deeplob_script_returns_empty(tmp_path):
    """deeplob writes no summary -> {}."""
    out = _harvest_backtest_metrics(tmp_path, "x", DEEPLOB_SCRIPT)
    assert out == {}


def test_empty_script_returns_empty(tmp_path):
    """script='' (the new C2 default) -> {} (no harvest)."""
    out = _harvest_backtest_metrics(tmp_path, "x", "")
    assert out == {}


def test_harvest_never_raises_even_on_garbage_inputs(tmp_path):
    """Fail-soft contract: the harvest is observation-tier and must never raise
    (a harvest failure must not fail the backtest stage)."""
    # output_dir that doesn't exist + odd script name
    weird = tmp_path / "nope"
    assert _harvest_backtest_metrics(weird, "x", REGRESSION_SCRIPT) == {}
    assert _harvest_backtest_metrics(weird, "", "??!!") == {}


# ==========================================================================
# S2 — wire the harvest into BacktestRunner.run (producer captured_metrics)
# ==========================================================================
class TestBacktestRunnerWire:
    """The runner harvests into result.captured_metrics['backtest_metrics'] on
    subprocess success (and sets result.output_dir); on failure/dry-run it does
    not harvest. run_subprocess is mocked (no real backtester invoked)."""

    @staticmethod
    def _run(tmp_path, monkeypatch, *, returncode, write_summary, dry_run=False,
             script=REGRESSION_SCRIPT, name="exp_x"):
        import subprocess
        from hft_ops.config import OpsConfig, PipelinePaths
        from hft_ops.manifest.schema import (
            ExperimentManifest, ExperimentHeader, Stages, BacktestingStage,
        )
        from hft_ops.stages import backtesting as bt_mod

        paths = PipelinePaths(pipeline_root=tmp_path)
        output_dir = paths.backtester_dir / "outputs" / "backtests"
        output_dir.mkdir(parents=True, exist_ok=True)
        if write_summary:
            _write_summary(output_dir, name, zero_dte=False, results=[
                _threshold("a", total_return=0.03, sharpe=2.0, max_dd=0.005, win_rate=0.58),
            ])
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name=name),
            stages=Stages(backtesting=BacktestingStage(enabled=True, script=script)),
        )
        ops = OpsConfig(paths=paths, dry_run=dry_run, verbose=False)
        monkeypatch.setattr(
            bt_mod, "run_subprocess",
            lambda *a, **k: subprocess.CompletedProcess(
                args=[], returncode=returncode, stdout="ok", stderr="",
            ),
        )
        result = bt_mod.BacktestRunner().run(manifest, ops)
        return result, str(output_dir)

    def test_success_harvests_metrics_and_sets_output_dir(self, tmp_path, monkeypatch):
        from hft_ops.stages.base import StageStatus
        result, output_dir = self._run(tmp_path, monkeypatch, returncode=0, write_summary=True)
        assert result.status == StageStatus.COMPLETED
        assert result.captured_metrics["backtest_metrics"] == {
            "total_return": 0.03, "sharpe_ratio": 2.0, "max_drawdown": 0.005, "win_rate": 0.58,
        }
        assert result.output_dir == output_dir

    def test_failure_does_not_harvest(self, tmp_path, monkeypatch):
        from hft_ops.stages.base import StageStatus
        result, _ = self._run(tmp_path, monkeypatch, returncode=1, write_summary=True)
        assert result.status == StageStatus.FAILED
        assert "backtest_metrics" not in result.captured_metrics

    def test_success_missing_summary_records_empty_and_still_completes(self, tmp_path, monkeypatch):
        from hft_ops.stages.base import StageStatus
        result, _ = self._run(tmp_path, monkeypatch, returncode=0, write_summary=False)
        assert result.status == StageStatus.COMPLETED  # harvest fail-soft must not flip status
        assert result.captured_metrics["backtest_metrics"] == {}

    def test_dry_run_skips_without_harvest(self, tmp_path, monkeypatch):
        from hft_ops.stages.base import StageStatus
        result, _ = self._run(tmp_path, monkeypatch, returncode=0, write_summary=True, dry_run=True)
        assert result.status == StageStatus.SKIPPED
        assert "backtest_metrics" not in result.captured_metrics


# ==========================================================================
# S3 — orchestrator wire: _record_experiment moves the producer's
# captured_metrics['backtest_metrics'] onto ExperimentRecord.backtest_metrics
# (the persisted, index-projected, compare-readable field). This is the single
# point covering single-run + sweep-serial + sweep-parallel (all route here).
# ==========================================================================
class TestRecordExperimentBacktestWire:
    @staticmethod
    def _record(tmp_path, *, name, backtesting_metrics=None, include_backtesting=True):
        from hft_ops.cli import _record_experiment
        from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
        from hft_ops.paths import PipelinePaths
        from hft_ops.stages.base import StageResult, StageStatus
        from hft_ops.ledger.ledger import ExperimentLedger

        paths = PipelinePaths(pipeline_root=tmp_path)
        training = StageResult(stage_name="training")
        training.status = StageStatus.COMPLETED
        training.captured_metrics = {"test_ic": 0.38}
        results = {"training": training}
        if include_backtesting:
            bt = StageResult(stage_name="backtesting")
            bt.status = StageStatus.COMPLETED
            bt.captured_metrics = (
                {} if backtesting_metrics is None
                else {"backtest_metrics": backtesting_metrics}
            )
            results["backtesting"] = bt
        manifest = ExperimentManifest(experiment=ExperimentHeader(name=name))
        eid = _record_experiment(
            manifest, paths, fingerprint="d" * 64,
            results=results, total_duration=1.0,
        )
        return ExperimentLedger(paths.ledger_dir).get(eid)

    def test_backtest_metrics_persisted_on_record_and_index(self, tmp_path):
        m = {"total_return": 0.03, "sharpe_ratio": 1.5, "max_drawdown": 0.01, "win_rate": 0.55}
        record = self._record(tmp_path, name="rec_bt", backtesting_metrics=m)
        assert record.backtest_metrics == m
        # index_entry projection (what `hft-ops compare` reads) carries the whitelist
        idx = record.index_entry()
        assert idx["backtest_metrics"]["total_return"] == 0.03
        assert idx["backtest_metrics"]["sharpe_ratio"] == 1.5

    def test_empty_backtest_metrics_leaves_default(self, tmp_path):
        record = self._record(tmp_path, name="rec_empty", backtesting_metrics={})
        assert record.backtest_metrics == {}

    def test_no_backtesting_stage_leaves_default(self, tmp_path):
        record = self._record(tmp_path, name="rec_none", include_backtesting=False)
        assert record.backtest_metrics == {}


# ==========================================================================
# S4 — C2: drop the broken silent default (`scripts/backtest_deeplob.py`) at
# BOTH the schema field AND the loader fallback (R2 C2-1: a 2nd hand-mirror);
# fail loud at VALIDATE time when an enabled backtesting stage has no script
# (R2 C2-2: `resolve("")` -> pipeline_root EXISTS, so the existence check
# silently passes -> mid-run "is a directory"; the guard must fire first).
# ==========================================================================
class TestC2ScriptRequiredFailLoud:
    def test_schema_default_script_is_empty(self):
        from hft_ops.manifest.schema import BacktestingStage
        assert BacktestingStage().script == ""

    def test_loader_default_script_is_empty(self):
        """A backtesting block omitting `script:` must NOT silently get the
        broken deeplob default — the loader was an independent 2nd mirror."""
        from hft_ops.manifest.loader import _build_backtesting
        stage = _build_backtesting({"enabled": True})
        assert stage.script == ""

    def test_validate_inputs_empty_script_fails_loud(self, tmp_path):
        from hft_ops.config import OpsConfig, PipelinePaths
        from hft_ops.manifest.schema import (
            ExperimentManifest, ExperimentHeader, Stages, BacktestingStage,
        )
        from hft_ops.stages.backtesting import BacktestRunner

        paths = PipelinePaths(pipeline_root=tmp_path)
        paths.backtester_dir.mkdir(parents=True, exist_ok=True)  # isolate the script error
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="x"),
            stages=Stages(backtesting=BacktestingStage(enabled=True, script="")),
        )
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)
        errors = BacktestRunner().validate_inputs(manifest, ops)
        assert any("script is required" in e for e in errors), errors
        assert any("run_regression_backtest.py" in e for e in errors), errors
        # MUST NOT degrade into the misleading "is a directory" path: the empty
        # script must be caught by the required-guard, not slip past the
        # existence check (resolve("") == pipeline_root, which exists).
        assert not any("Backtest script not found" in e for e in errors), errors

    def test_validate_inputs_valid_script_no_required_error(self, tmp_path):
        from hft_ops.config import OpsConfig, PipelinePaths
        from hft_ops.manifest.schema import (
            ExperimentManifest, ExperimentHeader, Stages, BacktestingStage,
        )
        from hft_ops.stages.backtesting import BacktestRunner

        paths = PipelinePaths(pipeline_root=tmp_path)
        script_rel = "lob-backtester/scripts/run_regression_backtest.py"
        script_abs = paths.resolve(script_rel)
        script_abs.parent.mkdir(parents=True, exist_ok=True)  # also creates backtester_dir
        script_abs.write_text("# stub")
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name="x"),
            stages=Stages(backtesting=BacktestingStage(enabled=True, script=script_rel)),
        )
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)
        errors = BacktestRunner().validate_inputs(manifest, ops)
        assert not any("script is required" in e for e in errors), errors
        assert not any("Backtest script not found" in e for e in errors), errors


# ==========================================================================
# ASSEMBLED PATH (re-validation gap closure): producer -> orchestrator ->
# ledger, stitched with REAL components (only the subprocess mocked). S2 and
# S3 are tested in isolation; this proves the wire between the REAL
# BacktestRunner.run StageResult and the REAL _record_experiment, end to end.
# ==========================================================================
class TestAssembledProducerToLedger:
    def test_real_run_result_flows_to_ledger_record_and_index(self, tmp_path, monkeypatch):
        import subprocess
        from hft_ops.config import OpsConfig, PipelinePaths
        from hft_ops.manifest.schema import (
            ExperimentManifest, ExperimentHeader, Stages, BacktestingStage,
        )
        from hft_ops.stages import backtesting as bt_mod
        from hft_ops.stages.base import StageResult, StageStatus
        from hft_ops.cli import _record_experiment
        from hft_ops.ledger.ledger import ExperimentLedger

        paths = PipelinePaths(pipeline_root=tmp_path)
        output_dir = paths.backtester_dir / "outputs" / "backtests"
        output_dir.mkdir(parents=True, exist_ok=True)
        name = "assembled_exp"
        _write_summary(output_dir, name, zero_dte=False, results=[
            _threshold("a", total_return=0.021, sharpe=1.42, max_dd=0.013, win_rate=0.552),
        ])
        manifest = ExperimentManifest(
            experiment=ExperimentHeader(name=name),
            stages=Stages(backtesting=BacktestingStage(enabled=True, script=REGRESSION_SCRIPT)),
        )
        ops = OpsConfig(paths=paths, dry_run=False, verbose=False)
        monkeypatch.setattr(
            bt_mod, "run_subprocess",
            lambda *a, **k: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr="",
            ),
        )
        # (1) REAL producer harvests into its StageResult.captured_metrics
        bt_result = bt_mod.BacktestRunner().run(manifest, ops)
        assert bt_result.status == StageStatus.COMPLETED
        assert bt_result.captured_metrics["backtest_metrics"]["total_return"] == 0.021

        # (2) the REAL StageResult flows through the REAL orchestrator wire
        training = StageResult(stage_name="training")
        training.status = StageStatus.COMPLETED
        training.captured_metrics = {"test_ic": 0.3}
        results = {"training": training, "backtesting": bt_result}
        eid = _record_experiment(
            manifest, paths, fingerprint="e" * 64,
            results=results, total_duration=1.0,
        )

        # (3) the on-disk ledger record + index projection carry the metrics
        record = ExperimentLedger(paths.ledger_dir).get(eid)
        assert record.backtest_metrics == {
            "total_return": 0.021, "sharpe_ratio": 1.42,
            "max_drawdown": 0.013, "win_rate": 0.552,
        }
        assert record.index_entry()["backtest_metrics"]["total_return"] == 0.021
