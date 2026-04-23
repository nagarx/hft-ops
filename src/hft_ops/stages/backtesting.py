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

import sys
import time
from pathlib import Path
from typing import List

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)


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

        # Pass-through for script-specific args (readability `--min-agreement`
        # / `--min-confidence`, regression `--zero-dte` / `--commission`, all
        # exchange overrides, etc.). Operators set these explicitly in
        # manifest YAML's `stages.backtesting.extra_args`.
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
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"{script_basename} exited with code {proc.returncode}"
                )
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
