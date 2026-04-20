"""
Feature extraction stage runner.

Invokes the Rust feature-extractor-MBO-LOB binary (export_dataset) as a
subprocess. Supports Phase 8A.0 content-addressed extraction cache (consults
``data/exports/_cache/`` before extracting; populates on success) AND legacy
``skip_if_exists`` (deprecated; superseded by cache — see manifest schema
DeprecationWarning).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_contracts import SCHEMA_VERSION

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.scheduler.extraction_cache import (
    CacheKeyInputs,
    compute_cache_key,
    populate,
    prepare_cache_key_inputs,
    resolve_or_link,
)
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    run_subprocess,
    _tail,
)

logger = logging.getLogger(__name__)


class ExtractionRunner:
    """Runs feature extraction via cargo run --bin export_dataset.

    Cache-consult order (Phase 8A.0):
      1. If ``OpsConfig.cache_extraction`` and cache-key inputs gatherable:
         - Compute content-addressed cache_key
         - ``resolve_or_link(cache_key, output_dir, cache_root)``
         - Outcome ``hit`` → SKIPPED, ``captured_metrics[cache_hit]=True``
         - Outcome ``poisoned`` → fall through, re-extract, re-populate
         - Outcome ``miss`` → fall through, extract, populate on success
      2. Legacy ``skip_if_exists`` (only when cache is disabled or inputs
         cannot be gathered).
      3. Run extractor subprocess.
      4. On success + cache enabled + inputs available, ``populate(...)``.
    """

    @property
    def stage_name(self) -> str:
        return "extraction"

    def validate_inputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.extraction
        if not stage.config:
            errors.append("extraction.config is required")
        else:
            config_path = config.paths.resolve(stage.config)
            if not config_path.exists():
                errors.append(f"Extractor config not found: {config_path}")

        if not config.paths.extractor_dir.exists():
            errors.append(f"Extractor directory not found: {config.paths.extractor_dir}")

        return errors

    def run(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> StageResult:
        stage = manifest.stages.extraction
        result = StageResult(stage_name=self.stage_name)
        output_dir: Optional[Path] = (
            config.paths.resolve(stage.output_dir) if stage.output_dir else None
        )

        cache_root = config.paths.exports_dir / "_cache"
        cache_key_inputs: Optional[CacheKeyInputs] = None
        cache_key: Optional[str] = None

        # -------- Phase 8A.0 cache consult (before extraction) -----------
        if config.cache_extraction and output_dir is not None:
            try:
                cache_key_inputs = prepare_cache_key_inputs(
                    extractor_config_path=config.paths.resolve(stage.config),
                    extractor_dir=config.paths.extractor_dir,
                    reconstructor_dir=config.paths.reconstructor_dir,
                    hft_statistics_dir=None,  # auto-detect via .cargo/config.toml
                    contract_version=SCHEMA_VERSION,
                    data_dir=config.paths.data_dir,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "Cache disabled for this run — exception during key-input "
                    "gathering: %s",
                    exc,
                )
                cache_key_inputs = None

            if cache_key_inputs is not None:
                cache_key = compute_cache_key(cache_key_inputs)
                outcome = resolve_or_link(cache_key, output_dir, cache_root)
                result.captured_metrics["cache_key"] = cache_key
                result.captured_metrics["cache_hit"] = outcome.status == "hit"

                if outcome.status == "hit":
                    logger.info(
                        "[cache hit: %s, saved ~%.1fs, linked %d files via %s] "
                        "extraction → %s",
                        cache_key[:12],
                        outcome.seconds_saved,
                        outcome.linked_files,
                        outcome.link_type,
                        output_dir,
                    )
                    result.status = StageStatus.SKIPPED
                    result.output_dir = str(output_dir)
                    result.captured_metrics["cache_seconds_saved"] = outcome.seconds_saved
                    result.captured_metrics["cache_linked_files"] = outcome.linked_files
                    result.captured_metrics["cache_link_type"] = outcome.link_type
                    return result
                elif outcome.status == "poisoned":
                    logger.warning(
                        "[cache poisoned: %s] falling through to extraction",
                        cache_key[:12],
                    )
                    # Continue to extract + re-populate below
                else:
                    logger.info(
                        "[cache miss: key=%s] extracting",
                        cache_key[:12],
                    )

        # -------- Legacy skip_if_exists (fallback when cache disabled) ---
        if stage.skip_if_exists and output_dir is not None:
            if output_dir.exists() and any(output_dir.glob("*_metadata.json")):
                result.status = StageStatus.SKIPPED
                result.output_dir = str(output_dir)
                return result

        if config.dry_run:
            result.status = StageStatus.SKIPPED
            result.error_message = "dry-run: would run extraction"
            return result

        # -------- Subprocess invocation ----------------------------------
        config_path = config.paths.resolve(stage.config)
        cmd = [
            "cargo", "run", "--release",
            "--bin", "export_dataset",
            "--features", "parallel",
            "--",
            "--config", str(config_path),
        ]

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.extractor_dir,
                verbose=config.verbose,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                if stage.output_dir:
                    result.output_dir = str(config.paths.resolve(stage.output_dir))
            else:
                result.status = StageStatus.FAILED
                result.error_message = (
                    f"export_dataset exited with code {proc.returncode}"
                )
        except Exception as e:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        # -------- Populate cache on success ------------------------------
        if (
            result.status == StageStatus.COMPLETED
            and config.cache_extraction
            and cache_key_inputs is not None
            and cache_key is not None
            and output_dir is not None
            and output_dir.exists()
        ):
            try:
                populate(
                    cache_key,
                    output_dir,
                    cache_root,
                    extractor_duration_seconds=result.duration_seconds,
                    cache_key_inputs=cache_key_inputs,
                )
                logger.info(
                    "[cache populated: %s, size=%d files] for future reuse",
                    cache_key[:12],
                    sum(1 for _ in output_dir.rglob("*") if _.is_file()),
                )
            except Exception as exc:
                # Non-fatal — extraction succeeded, cache is opportunistic.
                logger.warning(
                    "Cache populate failed (extraction succeeded): %s", exc
                )

        return result

    def validate_outputs(
        self,
        manifest: ExperimentManifest,
        config: OpsConfig,
    ) -> List[str]:
        errors: List[str] = []
        stage = manifest.stages.extraction
        if not stage.output_dir:
            return errors

        output_dir = config.paths.resolve(stage.output_dir)
        if not output_dir.exists():
            errors.append(f"Extraction output directory not found: {output_dir}")
            return errors

        meta_files = sorted(output_dir.glob("*_metadata.json"))
        if not meta_files:
            errors.append(f"No metadata JSON files in {output_dir}")

        seq_files = sorted(output_dir.glob("*_sequences.npy"))
        if not seq_files:
            errors.append(f"No sequence .npy files in {output_dir}")

        label_files = sorted(output_dir.glob("*_labels.npy"))
        if not label_files:
            errors.append(f"No label .npy files in {output_dir}")

        return errors
