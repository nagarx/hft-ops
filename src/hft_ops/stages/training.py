"""
Training stage runner.

Invokes lob-model-trainer/scripts/train.py with the trainer YAML config,
applying overrides from the manifest. Resolves horizon_value to horizon_idx
from the export metadata at runtime.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hft_ops.config import OpsConfig
from hft_ops.ledger.dedup import _load_trainer_merge_module
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.paths import PipelinePaths
from hft_ops.stages._override_discipline import (
    KNOWN_TRAINER_PREFIXES,
    apply_override_loud,
    validate_trainer_override_prefixes,
)
from hft_ops.stages.base import (
    _format_subprocess_failure,
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

logger = logging.getLogger(__name__)


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
    *,
    validate_prefixes: bool = False,
    source: str = "_apply_overrides_to_dict",
) -> Dict[str, Any]:
    """Apply dotted-key overrides to a config dict in-place.

    Args:
        cfg: Config dict to modify.
        overrides: Dict of dotted keys to values (e.g., {"data.data_dir": "/path"}).
        validate_prefixes: Phase R-17 F5 (default False for back-compat). When
            True, validates each override's first segment against
            ``KNOWN_TRAINER_PREFIXES`` BEFORE applying any mutation (fail-fast
            per hft-rules §5). Closes #PY-131 typo class for trainer-config
            override paths. Production callers should pass ``True``; legacy
            test fixtures with synthetic keys may keep default ``False``.
        source: human-readable source for fail-loud error messages.

    Returns:
        The same dict (for chaining). Mutated in-place.

    Raises:
        UnknownOverrideKeyError: when ``validate_prefixes=True`` and any
            override key has unknown top-level prefix (typo detection).
    """
    # Phase R-17 F5: fail-fast prefix validation BEFORE any mutation,
    # so partial-mutation-then-raise is impossible.
    if validate_prefixes:
        validate_trainer_override_prefixes(overrides, source=source)

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return cfg


def _resolve_trainer_inheritance(
    cfg: Dict[str, Any],
    config_path: Path,
    paths: Optional[PipelinePaths],
) -> Dict[str, Any]:
    """Resolve `_base:` inheritance in a trainer YAML dict.

    N1 fix (forensic audit 2026-04-26 / verified 2026-05-05): the trainer
    materialization code paths (``_apply_overrides`` + ``_materialize_inline_config``)
    used to write the cfg dict verbatim — leaving ``_base:`` keys present in
    the materialized YAML. Downstream consumers like
    ``hft_ops.stages.contract_preflight.preflight_trainer_config`` then read
    the raw YAML and miss ``model.model_type`` / ``data.feature_count`` /
    ``data.sequence.window_size`` (which live in the base files), raising
    ValueError before ``validate_input_contract`` even runs. Calling this
    helper before override + write fixes the gap structurally.

    Reuses the torch-free SSoT shim ``_load_trainer_merge_module`` (loads
    ``<trainer_dir>/src/lobtrainer/config/merge.py`` directly via
    ``spec_from_file_location``, bypassing the package ``__init__.py``).
    Soft fallback policy mirrors ``hft_ops.ledger.dedup._resolve_inline_trainer_config``:

    - **No `_base:` key**: return cfg unchanged (no-op).
    - **Merge module unavailable**: WARN log + return cfg unchanged
      (degraded-CI tolerance; downstream may fail with clearer error).
    - **Hard errors** (inheritance cycle, depth exceeded, malformed `_base:`,
      missing base file): PROPAGATE — these are configuration bugs.

    Args:
        cfg: Trainer config dict (may contain `_base:`).
        config_path: Absolute path to the trainer YAML on disk (or a
            fictitious path inside ``<trainer_dir>/configs/`` for inline
            configs — relative `_base:` paths in the dict resolve relative
            to ``config_path.parent``).
        paths: PipelinePaths instance (provides ``trainer_dir``). When None
            (test fixtures without paths context), helper is a no-op.

    Returns:
        The resolved-and-merged dict (no `_base:` key). MAY be the same
        object as ``cfg`` if no `_base:` was present.

    Raises:
        ValueError: inheritance cycle, depth exceeded, or malformed `_base:`.
        FileNotFoundError: referenced base config not found.
    """
    if paths is None:
        # N1-B (post-review diagnostic): log when paths is None AND `_base:`
        # is present. This is the silent-pre-N1 path that downstream callers
        # may interpret as `model.model_type missing`. Observability per
        # hft-rules §8.
        if "_base" in cfg:
            logger.debug(
                "_resolve_trainer_inheritance: paths=None and `_base:` "
                "present — resolution skipped. cfg=%s. Downstream may "
                "raise `missing model.model_type`.", config_path
            )
        return cfg
    if "_base" not in cfg:
        return cfg
    merge_mod = _load_trainer_merge_module(paths)
    if merge_mod is None:
        # Degraded CI environment — trainer merge module not loadable.
        # Caller will likely surface a clearer error downstream.
        # N1-C (post-review diagnostic): WARN so the next "missing
        # model.model_type" error is interpretable.
        logger.warning(
            "_resolve_trainer_inheritance: trainer merge module unavailable "
            "for %s — cfg NOT resolved. Downstream `missing model.model_type` "
            "errors may be caused by this. Inspect <trainer_dir>/src/"
            "lobtrainer/config/merge.py.", config_path
        )
        return cfg
    # Phase α-1.3 / #PY-83-cluster (2026-05-10): use Path.absolute() not
    # Path.resolve() — caller-side of α-1.2 cycle-detection invariant.
    # resolve_inheritance internally uses .absolute() at merge.py:135 +
    # :158/:160; caller MUST match for consistency.
    return merge_mod.resolve_inheritance(cfg, config_path.absolute())


def _apply_overrides(
    config_path: Path,
    overrides: Dict[str, Any],
    output_path: Path,
    *,
    paths: Optional[PipelinePaths] = None,
) -> Path:
    """Apply dotted-key overrides to a YAML config and write to output_path.

    Args:
        config_path: Original trainer YAML.
        overrides: Dict of dotted keys to values (e.g., {"data.data_dir": "/path"}).
        output_path: Where to write the modified YAML.
        paths: Optional PipelinePaths. When provided, ``_base:`` inheritance
            is resolved BEFORE overrides are applied (N1 fix — prevents
            silent ``model.model_type`` loss in materialized YAML). When None
            (legacy callers / test fixtures with flat configs), behaves
            as pre-N1-fix.

    Returns:
        The output_path.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # N1 fix: resolve `_base:` BEFORE applying overrides + writing. Without
    # this, the materialized YAML retains `_base:` and downstream consumers
    # (e.g., contract_preflight) miss model.model_type which lives in a base.
    cfg = _resolve_trainer_inheritance(cfg, config_path, paths)

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
            # Phase α-1.3 / #PY-83-cluster (2026-05-10): use .absolute() not
            # .resolve() — these rewritten _base values re-enter
            # resolve_inheritance cycle-detection at the next recursion level.
            # Must preserve symlink-source per α-1.2 invariant.
            return str((trainer_configs_root / p).absolute())
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
    *,
    paths: Optional[PipelinePaths] = None,
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
        paths: Optional PipelinePaths. When provided, ``_base:`` inheritance is
            resolved (after absolutize, before overrides + write) so the
            materialized YAML has model.model_type populated for downstream
            preflight. N1 fix.

    Returns:
        The output_path.
    """
    # Deep copy so manifest state is not mutated
    import copy
    cfg = copy.deepcopy(inline_cfg)
    if trainer_configs_root is not None:
        _absolutize_inline_base_paths(cfg, trainer_configs_root)

    # N1 fix: resolve `_base:` inheritance BEFORE applying overrides + writing.
    # For inline configs without an on-disk source, use a fictitious path
    # inside trainer_configs_root so relative-path resolution (which uses
    # `config_path.parent`) gets the correct root. Mirrors the pattern in
    # `hft_ops.ledger.dedup._resolve_inline_trainer_config:551-559`.
    if "_base" in cfg and trainer_configs_root is not None:
        fake_config_path = trainer_configs_root / "__inline_trainer_config__.yaml"
        cfg = _resolve_trainer_inheritance(cfg, fake_config_path, paths)

    # Phase R-17 F5: production caller validates prefixes (mirror _apply_overrides path).
    _apply_overrides_to_dict(
        cfg, overrides,
        validate_prefixes=True,
        source="_materialize_inline_config",
    )

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
                # Phase R-17 F4 (2026-05-11): #PY-128 closure — replaced direct
                # `overrides["data.horizon_idx"] = horizon_idx` with apply_override_loud.
                # user_set_check=False during initial migration (WARN-only); Phase R-18
                # promotes to True after manifest audit.
                apply_override_loud(
                    overrides,
                    "data.horizon_idx",
                    horizon_idx,
                    source=(
                        f"training.py:393 resolved from horizon_value="
                        f"{stage.horizon_value} via export metadata"
                    ),
                    user_set_check=False,
                )
                result.captured_metrics["resolved_horizon_idx"] = horizon_idx

        output_dir = stage.output_dir
        if output_dir:
            output_dir = str(config.paths.resolve(output_dir))
            # Phase R-17 F4 (2026-05-11): NEW-BUG-10 closure — sister site to L393.
            # Replaces direct `overrides["output_dir"] = output_dir`.
            apply_override_loud(
                overrides,
                "output_dir",
                output_dir,
                source=f"training.py:399 resolved from stage.output_dir={stage.output_dir}",
                user_set_check=False,
            )

        # Materialize effective config: inline path vs file path
        if stage.trainer_config is not None:
            # Inline path: write trainer_config dict (with overrides) to temp YAML.
            # Absolutize ``_base:`` references so train.py's resolve_inheritance()
            # can find bases under <trainer_dir>/configs/ from the temp location.
            resolved_dir = config.paths.runs_dir / manifest.experiment.name
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / "resolved_trainer_config.yaml"
            # Phase α-1.3 / #PY-83-cluster (2026-05-10): use .absolute() not
            # .resolve() — trainer_configs_root propagates downstream into
            # _absolutize_inline_base_paths and _materialize_inline_config →
            # fake_config_path → resolve_inheritance, all of which must
            # preserve symlink-source per α-1.2 cycle-detection invariant.
            trainer_configs_root = (config.paths.trainer_dir / "configs").absolute()
            effective_config = _materialize_inline_config(
                stage.trainer_config,
                overrides,
                resolved_config,
                trainer_configs_root=trainer_configs_root,
                paths=config.paths,
            )
            result.captured_metrics["_trainer_config_source"] = "inline"
        elif overrides:
            original_config = config.paths.resolve(stage.config)
            resolved_dir = config.paths.runs_dir / manifest.experiment.name
            resolved_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = resolved_dir / "resolved_trainer_config.yaml"
            effective_config = _apply_overrides(
                original_config, overrides, resolved_config,
                paths=config.paths,
            )
            result.captured_metrics["_trainer_config_source"] = "path_with_overrides"
        else:
            effective_config = config.paths.resolve(stage.config)
            result.captured_metrics["_trainer_config_source"] = "path_raw"

        # Store effective config path for _record_experiment to load
        result.captured_metrics["_effective_config_path"] = str(effective_config)

        # Phase V.A.8 MVP (2026-04-21): InputContract pre-flight.
        # Catch misconfigured YAMLs BEFORE the GPU subprocess launches —
        # save GPU-hours for misconfigurations like TLOB with window_size=1
        # or DeepLOB with feature_count=128. Hardcoded per-model constraint
        # table synced from lob-models ModelRegistry (see
        # `contract_preflight._INPUT_CONTRACTS` for drift-risk note and
        # Phase VI replacement path).
        #
        # Fail-loud convention per Agent 2 M1 fix: ValueError wrapped to
        # StageResult(FAILED) with gate_report conforming to
        # hft_contracts.gate_report.GateReportDict (status="fail"). The
        # generic captured_metrics["gate_report"] harvest in
        # cli.py::_record_experiment surfaces this into
        # ExperimentRecord.gate_reports["training_preflight"] for fast
        # `ledger list --gate-status fail` filtering.
        from hft_ops.stages.contract_preflight import preflight_trainer_config
        preflight_start = time.monotonic()
        try:
            # N1 fix: pass paths so preflight can defensively resolve `_base:`
            # if the materialized YAML still carries it (which it should NOT
            # post Sites B+C fix, but defense-in-depth catches direct callers
            # or future regressions).
            preflight_trainer_config(Path(effective_config), paths=config.paths)
        except ValueError as exc:
            result.duration_seconds = time.monotonic() - preflight_start
            result.status = StageStatus.FAILED
            result.error_message = f"Input contract pre-flight failed: {exc}"
            result.captured_metrics["gate_report"] = {
                "status": "fail",
                "reason": "input_contract_violation",
                "summary": str(exc)[:256],
            }
            return result

        script = config.paths.trainer_dir / "scripts" / "train.py"
        cmd = [
            sys.executable, str(script),
            "--config", str(effective_config),
        ]

        if output_dir:
            cmd.extend(["--output-dir", output_dir])

        # Phase X.1 v2 (2026-05-04): pass through operator opt-in for checkpoint
        # fingerprint strict-mode. Default is warn-only at load_checkpoint per
        # Phase X.4 promotion plan; this flag promotes to raise.
        if getattr(stage, "strict_checkpoint_fingerprint", False):
            cmd.append("--strict-checkpoint-fingerprint")

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
                # Phase α-2 / #PY-80 (2026-05-10) — surface argparse + traceback
                # stderr to the orchestrator's main loop (cli.py prints
                # error_message; without this, stderr was buried in result.stderr).
                result.error_message = _format_subprocess_failure(proc, "train.py")
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
