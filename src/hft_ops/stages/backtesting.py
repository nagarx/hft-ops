"""
Backtesting stage runner.

Invokes a backtester script (configurable via manifest) with the model
checkpoint, signal directory, and backtest parameters. Supports multiple
scripts: ``backtest_deeplob.py`` (default), ``run_readability_backtest.py``,
``run_regression_backtest.py``, ``run_spread_signal_backtest.py``.

The standard ``params`` block (``initial_capital``, ``position_size``,
``spread_bps``, etc.) is passed as CLI args to scripts that accept them
(all current scripts). Script-specific config is passed via ``extra_args``
or ``params_file``.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    _format_subprocess_failure,
    run_subprocess,
    _tail,
)

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# H6 — backtest-metrics harvest (regression path).
#
# Greenfield: BacktestRunner.run previously set NO captured_metrics, so every
# orchestrated backtest recorded ExperimentRecord.backtest_metrics == {} (the
# live 98/98 monitorability blackout). These module-private helpers read the
# backtester's deterministic regression summary and reduce it to a flat
# snake_case dict that the ledger index whitelist + `hft-ops compare` consume.
# Mirrors the signal_export.py `_harvest_*` convention. Observation-tier +
# fail-soft (never raises -> never fails the backtest stage).
# --------------------------------------------------------------------------

# Regression summary (lob-backtester run_regression_backtest.py) per-threshold
# PascalCase -> flat snake_case. PURE key-rename (raw values preserved) so that
# `hft-ops compare` (which reads the index-projected backtest_metrics whitelist
# {total_return, sharpe_ratio, max_drawdown, win_rate, total_trades}) sees them.
_REGRESSION_METRIC_MAP = {
    "TotalReturn": "total_return",
    "SharpeRatio": "sharpe_ratio",
    "MaxDrawdown": "max_drawdown",
    "WinRate": "win_rate",
}
# 0DTE option scalars — emitted only when the summary's zero_dte_enabled is set.
_REGRESSION_OPTION_MAP = {
    "option_return_pct": "option_return_pct",
    "option_win_rate": "option_win_rate",
}


def _finite_or_none(value: Any) -> Any:
    """Return ``value`` iff it is a finite real number, else ``None``.

    Guards NaN/Inf out of the ledger: ``atomic_write_json`` uses
    ``json.dump(allow_nan=True)``, so a NaN would silently serialize as the
    non-standard ``NaN`` literal (breaks strict parsers) AND NaN-compares-False
    silently corrupts ``hft-ops compare`` ranking. Verified necessary even on
    the regression path: a losing experiment's best-by-``option_return_pct`` row
    is the do-nothing threshold, which carries NaN sharpe/win_rate
    (cycle12_r20_hmhp_r__seed_42.json, 2026-05-31).
    """
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _finite_sort_key(result: Dict[str, Any], key: str) -> float:
    """Best-threshold selection key: non-finite -> ``-inf`` so it never wins."""
    v = result.get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("-inf")


def _harvest_regression_metrics(summary_path: Path) -> Dict[str, Any]:
    """Reduce a regression ``<NAME>.json`` summary to a flat snake_case dict.

    Replicates the backtester's own best-threshold selection
    (run_regression_backtest.py:610-619 — documented-intentional mirror; the
    sibling script is not importable across the repo boundary): for 0DTE runs
    pick the threshold with the max ``option_return_pct``, else the max
    ``TotalReturn``. Remaps the chosen row's PascalCase keys, dropping any
    non-finite value. ``total_trades`` is intentionally OMITTED (absent from the
    regression summary; ``n_entries`` counts entries not legs — a misleading
    proxy).

    SEMANTIC NOTE (read before interpreting ``total_return``): this is the
    backtester's OWN best-threshold headline (faithful to
    run_regression_backtest.py:610-619), NOT a "did the model trade well" score.
    For a LOSING experiment (all current dark records are losers per the
    validated findings), the best-by-``option_return_pct`` row is the do-nothing
    ultra-conservative threshold (0 trades), so ``total_return == 0.0`` means
    "the best achievable outcome was to NOT trade", NOT "neutral performance".
    Use ``hft-ops ledger show`` for per-threshold detail. A future follow-on
    (with the readability harvest) may add an ``n_entries``/``traded``
    disambiguator to the index whitelist so ``hft-ops compare`` can distinguish
    do-nothing-0.0 from neutral-0.0 (needs an hft-contracts index-schema change).
    """
    data = json.loads(summary_path.read_text())
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return {}
    zero_dte = bool(data.get("zero_dte_enabled"))
    has_option = any(
        isinstance(r, dict) and "option_return_pct" in r for r in results
    )
    if zero_dte and has_option:
        best = max(results, key=lambda r: _finite_sort_key(r, "option_return_pct"))
    else:
        best = max(results, key=lambda r: _finite_sort_key(r, "TotalReturn"))
    if not isinstance(best, dict):
        return {}
    out: Dict[str, Any] = {}
    for src, dst in _REGRESSION_METRIC_MAP.items():
        v = _finite_or_none(best.get(src))
        if v is not None:
            out[dst] = v
    if zero_dte:
        for src, dst in _REGRESSION_OPTION_MAP.items():
            v = _finite_or_none(best.get(src))
            if v is not None:
                out[dst] = v
    return out


def _harvest_backtest_metrics(
    output_dir: Path, run_name: str, script_name: str
) -> Dict[str, Any]:
    """Harvest a flat snake_case ``backtest_metrics`` dict from the backtester's
    on-disk output. Observation-tier + FAIL-SOFT: any error logs a WARN and
    returns ``{}`` — it MUST NEVER raise (a harvest failure must not fail the
    backtest stage, which already succeeded).

    Dispatches on the backtester script: ``run_regression_backtest.py`` writes a
    deterministic ``<output_dir>/<run_name>.json`` (the only path implemented
    here — recovers the 93/98 regression dark records + all future regression).
    Readability (classification) harvest is a deferred follow-on (random run_id
    index-scan + NaN-sanitizer + single-match rule); ``backtest_deeplob.py`` /
    spread write no harvestable summary. All non-regression cases return ``{}``.
    """
    try:
        name = Path(script_name).name
        if "regression" in name:
            summary = Path(output_dir) / f"{run_name}.json"
            if not summary.exists():
                _logger.warning(
                    "backtest-metrics harvest: regression summary not found at "
                    "%s — backtest_metrics will be empty for run '%s'.",
                    summary, run_name,
                )
                return {}
            return _harvest_regression_metrics(summary)
        if "readability" in name:
            _logger.warning(
                "backtest-metrics harvest: readability (classification) harvest "
                "is a deferred follow-on; backtest_metrics empty for run '%s'.",
                run_name,
            )
            return {}
        # deeplob / spread / unknown / empty-script: no harvestable summary.
        return {}
    except Exception as exc:  # noqa: BLE001 — fail-soft observation tier
        _logger.warning(
            "backtest-metrics harvest failed (backtest_metrics empty) for run "
            "'%s' via '%s': %s", run_name, script_name, exc,
        )
        return {}


class BacktestRunner:
    """Runs backtesting via the script specified in the manifest."""

    @property
    def stage_name(self) -> str:
        return "backtesting"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.backtesting

        backtester_dir = config.paths.backtester_dir
        if not backtester_dir.exists():
            errors.append(f"Backtester directory not found: {backtester_dir}")

        # Validate the configured script exists.
        #
        # V.1.5 Frame-5 Task-1c fix (2026-04-23): script path is PIPELINE-ROOT-
        # RELATIVE by convention (matches `extraction.config`, `data.data_dir`,
        # `stage.checkpoint` — all resolved via `config.paths.resolve()`).
        # Previous `backtester_dir / stage.script` produced DOUBLED prefix
        # (`/...lob-backtester/lob-backtester/scripts/...`) when manifests used
        # the canonical pipeline-root-relative `lob-backtester/scripts/...`
        # path. Bug had never surfaced because backtesting stage had never been
        # exercised live via orchestrator. Unified with
        # `config.paths.resolve(stage.script)` to match pipeline-wide convention.
        # C2 (2026-05-31): an enabled backtesting stage MUST specify a script —
        # there is no default. This guard MUST run BEFORE the existence check:
        # config.paths.resolve("") returns pipeline_root (which EXISTS), so an
        # empty script would silently pass the existence check and only fail
        # mid-run with a confusing "is a directory" subprocess error. Fail loud
        # here with an actionable message (validate_inputs is only called for
        # enabled stages — see cli.py stage loop).
        if not stage.script:
            errors.append(
                "stages.backtesting.script is required (no default); use "
                "lob-backtester/scripts/run_regression_backtest.py (regression) "
                "or lob-backtester/scripts/run_readability_backtest.py "
                "(classification)."
            )
        else:
            script_path = config.paths.resolve(stage.script)
            if not script_path.exists():
                errors.append(
                    f"Backtest script not found: {script_path} "
                    f"(configured via stages.backtesting.script='{stage.script}')"
                )

        if stage.model_checkpoint:
            checkpoint = config.paths.resolve(stage.model_checkpoint)
            if not checkpoint.exists() and not manifest.stages.training.enabled:
                errors.append(f"Model checkpoint not found: {checkpoint}")

        if stage.params_file:
            params_file = config.paths.resolve(stage.params_file)
            if not params_file.exists():
                errors.append(f"Backtest params_file not found: {params_file}")

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.backtesting
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = (
                f"dry-run: would run backtesting via {stage.script}"
            )
            return result

        # V.1.5 Frame-5 Task-1c fix: see matching comment at validate-site above.
        script = config.paths.resolve(stage.script)

        cmd = [sys.executable, str(script)]

        # Phase 7.5-B.1 (2026-04-23) — closes Bug #5 of the Frame 5 Task 1 audit.
        # Runner previously passed `--experiment`, `--data-dir`, `--signals-dir`,
        # `--horizon-idx`, `--params-file`, `--spread-bps`, `--slippage-bps`,
        # `--threshold`, `--no-short`, `--device` — ALL rejected by argparse
        # on the current backtester scripts (`run_readability_backtest.py`,
        # `run_regression_backtest.py`, `run_spread_signal_backtest.py`) which
        # expect `--signals`, `--max-spread-bps`, `--commission`, `--name`,
        # `--exchange`, `--manifest`, `--output-dir`, plus script-specific flags.
        #
        # Bug had never surfaced because the backtesting stage had never been
        # exercised live via the orchestrator (Frame 5 Task 1 discovery: 0/34
        # ledger records are live). The legacy dead fields (`data_dir`,
        # `horizon_idx`, `params_file`, `model_checkpoint`, `slippage_bps`,
        # `threshold`, `no_short`, `device`) are KEPT in the schema for
        # back-compat (2 existing manifests reference them) but STRIPPED from
        # the cmd construction here — marked deprecated in schema docstrings
        # with 2026-10-31 removal deadline (Phase 7.5-B.2 follow-up).
        #
        # Script-specific flags (readability `--min-agreement`, regression
        # `--zero-dte`, exchange `--exchange=ARCX`, etc.) MUST be passed via
        # manifest's `stage.extra_args` list. The runner passes a minimal
        # common-denominator set of flags; extra_args is the documented escape
        # hatch.

        # REQUIRED args (accepted by ALL 3 backtester scripts)
        if stage.signals_dir:
            signals = str(config.paths.resolve(stage.signals_dir))
            cmd.extend(["--signals", signals])

        # Script name — maps to backtester argparse `--name` (for output-dir
        # naming + gate report identification).
        cmd.extend(["--name", manifest.experiment.name])

        # Ledger linkage — backtester scripts accept `--manifest` to record
        # the authoring manifest path in their output for cross-tool traceability.
        if manifest.manifest_path:
            cmd.extend(["--manifest", manifest.manifest_path])

        # Numeric params with CORRECT flag names (scripts use `--max-spread-bps`,
        # NOT `--spread-bps`; scripts have NO `--slippage-bps` or `--device` —
        # drop those entirely).
        params = stage.params
        cmd.extend(["--initial-capital", str(params.initial_capital)])
        cmd.extend(["--position-size", str(params.position_size)])
        cmd.extend(["--max-spread-bps", str(params.spread_bps)])

        # #PY-180 hardening (2026-05-13): explicitly pass --output-dir to close
        # Wave 2 Agent F finding (producer-side dump location "works by accident").
        # Pre-fix: scripts default to "outputs/backtests/" (CWD-relative); subprocess
        # cwd=paths.backtester_dir → lands at <backtester_dir>/outputs/backtests/
        # IFF nobody changes the script default OR the cwd. Explicit absolute path
        # removes the silent dependency on those two invariants.
        # Per-trade .npy files dumped via `atomic_write_npy` post-Sub-cycle-4a
        # (de99f45) land at `<output_dir>/{run_name}__option_trade_pnls__{label}.npy`
        # for downstream R-16c analyzer ingestion.
        output_dir = config.paths.backtester_dir / "outputs" / "backtests"
        cmd.extend(["--output-dir", str(output_dir)])

        # Pass-through for script-specific args (readability `--min-agreement`
        # / `--min-confidence`, regression `--zero-dte` / `--commission`, all
        # exchange overrides, etc.). Operators set these explicitly in
        # manifest YAML's `stages.backtesting.extra_args`. extra_args is AFTER
        # the standard flags so operators can override `--output-dir` per-arm
        # if needed (rare; the explicit standard path is the recommended default).
        cmd.extend(stage.extra_args)

        script_basename = Path(stage.script).name

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.backtester_dir,
                verbose=config.verbose,
                env=config.env_overrides or None,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                # H6: harvest the backtester's metrics into captured_metrics so
                # the orchestrator persists them onto ExperimentRecord.
                # backtest_metrics (the live monitorability fix — previously the
                # stage recorded nothing -> 98/98 ledger records had {}).
                # _harvest_backtest_metrics is FAIL-SOFT (never raises) so it
                # cannot flip a successful backtest to FAILED.
                result.output_dir = str(output_dir)
                result.captured_metrics["backtest_metrics"] = _harvest_backtest_metrics(
                    output_dir, manifest.experiment.name, stage.script
                )
            else:
                result.status = StageStatus.FAILED
                # Phase α-2 / #PY-80 (2026-05-10) — surface stderr.
                result.error_message = _format_subprocess_failure(proc, script_basename)
        except Exception as e:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        return result

    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        return []
