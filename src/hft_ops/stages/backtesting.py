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

        # Validate the configured script exists
        script_path = backtester_dir / stage.script
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

        script = config.paths.backtester_dir / stage.script

        cmd = [sys.executable, str(script)]

        cmd.extend(["--experiment", manifest.experiment.name])

        if stage.data_dir:
            data_dir = str(config.paths.resolve(stage.data_dir))
            cmd.extend(["--data-dir", data_dir])

        if stage.signals_dir:
            signals_dir = str(config.paths.resolve(stage.signals_dir))
            cmd.extend(["--signals-dir", signals_dir])

        if stage.horizon_idx is not None:
            cmd.extend(["--horizon-idx", str(stage.horizon_idx)])

        if stage.params_file:
            params_file = str(config.paths.resolve(stage.params_file))
            cmd.extend(["--params-file", params_file])

        params = stage.params
        cmd.extend(["--initial-capital", str(params.initial_capital)])
        cmd.extend(["--position-size", str(params.position_size)])
        cmd.extend(["--spread-bps", str(params.spread_bps)])
        cmd.extend(["--slippage-bps", str(params.slippage_bps)])

        if params.threshold > 0:
            cmd.extend(["--threshold", str(params.threshold)])

        if params.no_short:
            cmd.append("--no-short")

        if params.device != "cpu":
            cmd.extend(["--device", params.device])

        cmd.extend(stage.extra_args)

        script_basename = Path(stage.script).name

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.backtester_dir,
                verbose=config.verbose,
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
