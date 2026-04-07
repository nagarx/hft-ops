"""
Training stage runner.

Invokes lob-model-trainer/scripts/train.py with the trainer YAML config,
applying overrides from the manifest. Resolves horizon_value to horizon_idx
from the export metadata at runtime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)


def _resolve_horizon_idx(
    horizon_value: int,
    export_dir: Path,
) -> Optional[int]:
    """Resolve horizon_value to horizon_idx from export metadata.

    Reads the first metadata JSON in export_dir, extracts the horizons
    list, and returns the index of horizon_value within it.

    Returns None if resolution fails.
    """
    meta_files = sorted(export_dir.glob("*_metadata.json"))
    if not meta_files:
        return None

    try:
        with open(meta_files[0], "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    horizons = metadata.get("horizons", metadata.get("max_horizons", []))
    if horizon_value in horizons:
        return horizons.index(horizon_value)

    return None


def _apply_overrides(
    config_path: Path,
    overrides: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Apply dotted-key overrides to a YAML config and write to output_path.

    Args:
        config_path: Original trainer YAML.
        overrides: Dict of dotted keys to values (e.g., {"data.data_dir": "/path"}).
        output_path: Where to write the modified YAML.

    Returns:
        The output_path.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


class TrainingRunner:
    """Runs model training via train.py."""

    @property
    def stage_name(self) -> str:
        return "training"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.training

        if not stage.config:
            errors.append("training.config is required")
        else:
            config_path = config.paths.resolve(stage.config)
            if not config_path.exists():
                errors.append(f"Trainer config not found: {config_path}")

        trainer_dir = config.paths.trainer_dir
        if not trainer_dir.exists():
            errors.append(f"Trainer directory not found: {trainer_dir}")

        script = trainer_dir / "scripts" / "train.py"
        if not script.exists():
            errors.append(f"train.py not found: {script}")

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.training
        result = StageResult(stage_name=self.stage_name)

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = "dry-run: would run training"
            return result

        original_config = config.paths.resolve(stage.config)
        overrides = dict(stage.overrides)

        if stage.horizon_value is not None and manifest.stages.extraction.output_dir:
            export_dir = config.paths.resolve(manifest.stages.extraction.output_dir)
            horizon_idx = _resolve_horizon_idx(stage.horizon_value, export_dir)
            if horizon_idx is not None:
                overrides["data.horizon_idx"] = horizon_idx
                result.captured_metrics["resolved_horizon_idx"] = horizon_idx

        output_dir = stage.output_dir
        if output_dir:
            output_dir = str(config.paths.resolve(output_dir))
            overrides["output_dir"] = output_dir

        if overrides:
            resolved_dir = config.paths.runs_dir / manifest.experiment.name
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / "resolved_trainer_config.yaml"
            effective_config = _apply_overrides(
                original_config, overrides, resolved_config
            )
        else:
            effective_config = original_config

        # Store effective config path for _record_experiment to load
        result.captured_metrics["_effective_config_path"] = str(effective_config)

        script = config.paths.trainer_dir / "scripts" / "train.py"
        cmd = [
            sys.executable, str(script),
            "--config", str(effective_config),
        ]

        if output_dir:
            cmd.extend(["--output-dir", output_dir])

        cmd.extend(stage.extra_args)

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.trainer_dir,
                verbose=config.verbose,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                if output_dir:
                    result.output_dir = output_dir
                self._capture_training_metrics(result)
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"train.py exited with code {proc.returncode}"
                )
        except Exception as e:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        return result

    def _capture_training_metrics(self, result: StageResult) -> None:
        """Try to capture training metrics from the output directory."""
        if not result.output_dir:
            return

        output_dir = Path(result.output_dir)
        history_file = output_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
                if isinstance(history, dict):
                    val_accs = history.get("val_accuracy", [])
                    if val_accs:
                        result.captured_metrics["best_val_accuracy"] = max(val_accs)
                    val_f1s = history.get("val_macro_f1", [])
                    if val_f1s:
                        result.captured_metrics["best_val_macro_f1"] = max(val_f1s)
            except (json.JSONDecodeError, OSError):
                pass

    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.training

        if not stage.output_dir:
            return errors

        output_dir = config.paths.resolve(stage.output_dir)
        if not output_dir.exists():
            errors.append(f"Training output directory not found: {output_dir}")
            return errors

        checkpoints_dir = output_dir / "checkpoints"
        if not checkpoints_dir.exists() or not any(checkpoints_dir.glob("*.pt")):
            errors.append(f"No model checkpoints found in {checkpoints_dir}")

        return errors
