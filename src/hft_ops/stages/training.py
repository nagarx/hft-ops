"""
Training stage runner.

Invokes lob-model-trainer/scripts/train.py with the trainer YAML config,
applying overrides from the manifest. Resolves horizon_value to horizon_idx
from the export metadata at runtime.
"""

from __future__ import annotations

import json
import math
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
# Phase 7 Stage 7.4 Round 4 item #1 (C1-complete). Re-use the authoritative
# regression val_* key taxonomy declared by the PostTrainingGateRunner
# instead of duplicating the list here. Prevents drift when the gate's
# primary-metric fallback order changes.
from hft_ops.stages.post_training_gate import (
    _REGRESSION_VAL_MAX_KEYS,
    _REGRESSION_VAL_MIN_KEYS,
)

# Classification-only max-better val_* keys. (val_signal_rate is emitted by
# the TLOB / opportunity classification strategies; val_macro_f1 and
# val_accuracy by every classification strategy.) val_loss lives in
# _REGRESSION_VAL_MIN_KEYS because it's min-better for any task.
_CLASSIFICATION_VAL_MAX_KEYS = (
    "val_accuracy",
    "val_macro_f1",
    "val_signal_rate",
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


def _apply_overrides_to_dict(
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply dotted-key overrides to a config dict in-place.

    Args:
        cfg: Config dict to modify.
        overrides: Dict of dotted keys to values (e.g., {"data.data_dir": "/path"}).

    Returns:
        The same dict (for chaining). Mutated in-place.
    """
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return cfg


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

    _apply_overrides_to_dict(cfg, overrides)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


def _absolutize_inline_base_paths(
    cfg: Dict[str, Any],
    trainer_configs_root: Path,
) -> None:
    """Convert relative ``_base:`` values in an inline trainer_config to absolute paths.

    Phase 1 wrapper-less manifests embed the trainer_config dict directly in
    the manifest. When that dict uses ``_base:`` for composition, the path is
    intuitively relative to ``<trainer_dir>/configs/`` (where bases live) —
    but we materialize the config to a temp file under ``hft-ops/runs/...``,
    and ``lobtrainer.config.merge.resolve_inheritance`` resolves ``_base``
    relative to the FILE's directory. Without this rewrite, bases would be
    searched under the temp directory and fail to load.

    This function walks the config dict and rewrites each relative ``_base``
    to an absolute path rooted at ``trainer_configs_root``. Absolute paths
    are left untouched. The walk is recursive so nested ``_base`` (added in
    Phase 3 for multi-base composition) is covered too.

    Args:
        cfg: The inline trainer_config dict. Mutated in-place.
        trainer_configs_root: Absolute path to ``<trainer_dir>/configs/``.
    """

    def _absolutize(value: Any) -> Any:
        if isinstance(value, str):
            p = Path(value)
            if p.is_absolute():
                return str(p)
            return str((trainer_configs_root / p).resolve())
        if isinstance(value, list):
            return [_absolutize(v) for v in value]
        return value

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "_base" in node:
                node["_base"] = _absolutize(node["_base"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(cfg)


def _materialize_inline_config(
    inline_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    output_path: Path,
    trainer_configs_root: Optional[Path] = None,
) -> Path:
    """Write an inline trainer_config dict (with overrides applied) to a YAML file.

    This is the wrapper-less path: the manifest contains the trainer config
    inline, and we serialize it to a file that train.py can consume. If the
    inline config uses ``_base:`` inheritance, relative base paths are
    absolutized (rooted at ``trainer_configs_root``) so they resolve correctly
    from the temp file's location.

    Args:
        inline_cfg: The trainer_config dict from the manifest (deep-copied here).
        overrides: Dict of dotted keys to values.
        output_path: Where to write the resolved YAML.
        trainer_configs_root: Absolute path to ``<trainer_dir>/configs/``.
            When supplied, relative ``_base:`` entries are absolutized against
            this root. When None (test harness, manifests without inheritance),
            ``_base`` entries are written verbatim.

    Returns:
        The output_path.
    """
    # Deep copy so manifest state is not mutated
    import copy
    cfg = copy.deepcopy(inline_cfg)
    if trainer_configs_root is not None:
        _absolutize_inline_base_paths(cfg, trainer_configs_root)
    _apply_overrides_to_dict(cfg, overrides)

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

        has_path = bool(stage.config)
        has_inline = stage.trainer_config is not None

        if not has_path and not has_inline:
            errors.append(
                "training: either 'config' (path) or 'trainer_config' (inline) is required"
            )
        elif has_path and has_inline:
            # Loader already catches this; defensive double-check.
            errors.append(
                "training: 'config' and 'trainer_config' are mutually exclusive"
            )
        elif has_path:
            config_path = config.paths.resolve(stage.config)
            if not config_path.exists():
                errors.append(f"Trainer config not found: {config_path}")
        # has_inline case: no file to check; dict already validated by loader.

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

        # Materialize effective config: inline path vs file path
        if stage.trainer_config is not None:
            # Inline path: write trainer_config dict (with overrides) to temp YAML.
            # Absolutize ``_base:`` references so train.py's resolve_inheritance()
            # can find bases under <trainer_dir>/configs/ from the temp location.
            resolved_dir = config.paths.runs_dir / manifest.experiment.name
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / "resolved_trainer_config.yaml"
            trainer_configs_root = (config.paths.trainer_dir / "configs").resolve()
            effective_config = _materialize_inline_config(
                stage.trainer_config,
                overrides,
                resolved_config,
                trainer_configs_root=trainer_configs_root,
            )
            result.captured_metrics["_trainer_config_source"] = "inline"
        elif overrides:
            original_config = config.paths.resolve(stage.config)
            resolved_dir = config.paths.runs_dir / manifest.experiment.name
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / "resolved_trainer_config.yaml"
            effective_config = _apply_overrides(
                original_config, overrides, resolved_config
            )
            result.captured_metrics["_trainer_config_source"] = "path_with_overrides"
        else:
            effective_config = config.paths.resolve(stage.config)
            result.captured_metrics["_trainer_config_source"] = "path_raw"

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
                env=config.env_overrides or None,
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
        """Capture training metrics from the output directory.

        Phase 7 Stage 7.4 Round 1 (2026-04-19) extended this to read both
        ``training_history.json`` AND ``test_metrics.json``. Round 4
        (2026-04-20) closes the C1-complete gap: the Round 1 version only
        iterated three keys (``val_accuracy``, ``val_macro_f1``,
        ``val_loss``) for per-epoch history, leaving every regression-
        specific val_* key silently invisible to ``PostTrainingGateRunner``
        prior-best lookup on ledger records. Round 4 unifies the
        iteration over:

        - classification max-better: ``_CLASSIFICATION_VAL_MAX_KEYS``
        - regression max-better:     ``_REGRESSION_VAL_MAX_KEYS``
          (re-imported from post_training_gate as SSoT)
        - min-better (any task):     ``_REGRESSION_VAL_MIN_KEYS``
          (includes val_loss, val_mae, val_rmse)

        All series values are filtered to finite scalars before max/min
        reduction to prevent NaN-poisoned ledger entries (rule §2, §8).

        Conventions:
        - per-epoch history is either dict-of-lists (legacy retroactive
          records) OR list-of-dicts (current MetricLogger callback output;
          see ``lobtrainer.training.callbacks:552``).
        - regression runs also write ``test_metrics.json`` — a flat dict of
          ``{test_<metric>: float}`` (established by
          ``lobtrainer.training.simple_trainer:223-225``; Round 4 item #6
          added the same for the PyTorch ``Trainer`` path at
          ``lob-model-trainer/scripts/train.py``). All finite float values
          are merged with the exact key preserved.

        Merge order is history-first, test-second — the final test-split
        metric wins on any key collision since it is the canonical final
        number used by ``_find_prior_best_experiment``.
        """
        if not result.output_dir:
            return

        output_dir = Path(result.output_dir)

        # Per-epoch validation history.
        max_keys = _CLASSIFICATION_VAL_MAX_KEYS + _REGRESSION_VAL_MAX_KEYS
        min_keys = _REGRESSION_VAL_MIN_KEYS

        def _finite(value: Any) -> bool:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        history_file = output_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
                if isinstance(history, dict):
                    # Legacy dict-of-lists format (retroactive records).
                    for key in max_keys:
                        series = [v for v in history.get(key, []) if _finite(v)]
                        if series:
                            result.captured_metrics[f"best_{key}"] = max(series)
                    for key in min_keys:
                        series = [v for v in history.get(key, []) if _finite(v)]
                        if series:
                            result.captured_metrics[f"best_{key}"] = min(series)
                elif isinstance(history, list):
                    # Current list-of-dicts per-epoch format.
                    for key in max_keys:
                        series = [
                            epoch.get(key)
                            for epoch in history
                            if isinstance(epoch, dict) and _finite(epoch.get(key))
                        ]
                        if series:
                            result.captured_metrics[f"best_{key}"] = max(series)
                    for key in min_keys:
                        series = [
                            epoch.get(key)
                            for epoch in history
                            if isinstance(epoch, dict) and _finite(epoch.get(key))
                        ]
                        if series:
                            result.captured_metrics[f"best_{key}"] = min(series)
            except (json.JSONDecodeError, OSError):
                pass

        # Regression-style test-split scalar metrics (Phase 7 Stage 7.4).
        # Flat ``{metric: float}`` dict convention. Merge finite values
        # only — NaN/Inf would poison ledger comparisons (rule §2, §8).
        test_metrics_file = output_dir / "test_metrics.json"
        if test_metrics_file.exists():
            try:
                with open(test_metrics_file, "r") as f:
                    test_metrics = json.load(f)
                if isinstance(test_metrics, dict):
                    for key, value in test_metrics.items():
                        if _finite(value):
                            result.captured_metrics[key] = float(value)
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
