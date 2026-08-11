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
    resolve_build_provenance,
    resolve_or_link,
)
from hft_ops.stages.base import (
    StageResult,
    StageStatus,
    _format_subprocess_failure,
    enforce_output_contract,
    run_subprocess,
    _tail,
)

logger = logging.getLogger(__name__)

# Cargo feature set for export_dataset. SINGLE SOURCE — used by BOTH the
# pre-hash `cargo build` (which fixes which binary the cache key describes) and
# the `cargo run` that performs the extraction. They MUST agree: a feature
# mismatch produces a different binary, so cargo would rebuild between the hash
# and the run and re-open the very TOCTOU the pre-hash build closes.
_EXTRACTOR_CARGO_FEATURES = "parallel"


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
            errors.append(
                f"Extractor directory not found: {config.paths.extractor_dir}"
            )

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
            # TOCTOU CLOSURE (2026-08-11) — build BEFORE hashing.
            #
            # ``compiled_binary_sha256`` is one of the 9 cache-key inputs and is
            # read from ``target/release/export_dataset`` right here, while the
            # extraction below runs ``cargo run --release``, which REBUILDS if any
            # source changed. Those are two different artifacts whenever a source
            # edit is pending, and the window is exactly the one that matters:
            # MBO-LOB-reconstructor is consumed through a path ``[patch]``
            # (feature-extractor-MBO-LOB/.cargo/config.toml, gitignored), so an
            # uncommitted edit there reaches this build immediately. Edit the
            # reconstructor, do not rebuild by hand, run the stage, take a cache
            # MISS, and the freshly-rebuilt binary's output is populated under a
            # key that names the PRE-EDIT binary. The reverse is equally wrong: a
            # pre-fix cached extraction stays reachable until something happens to
            # rebuild the binary.
            #
            # Building first makes the hashed artifact the artifact that runs;
            # ``cargo run`` below then finds it up to date and simply executes it.
            #
            # DELIBERATELY NOT FIXED by adding a ``reconstructor_dirty`` key input.
            # CacheKeyInputs' own docstring requires a MAJOR
            # CACHE_MANIFEST_SCHEMA_VERSION bump for any new field, which would
            # invalidate every existing cache entry — to buy a STRICTLY WEAKER
            # signal. Git-dirtiness is a proxy; the compiled binary hash is the
            # ground truth (a source edit can only change the output by changing
            # the binary), and dirtiness would additionally false-invalidate on a
            # docs-only reconstructor edit.
            #
            # A failed build is self-limiting and needs no special handling: the
            # ``cargo run`` below performs the same build, fails identically, the
            # stage is marked FAILED, and ``populate()`` is never reached — so no
            # entry can be written under any key.
            if not config.dry_run:
                try:
                    build_proc = run_subprocess(
                        [
                            "cargo",
                            "build",
                            "--release",
                            "--bin",
                            "export_dataset",
                            "--features",
                            _EXTRACTOR_CARGO_FEATURES,
                        ],
                        cwd=config.paths.extractor_dir,
                        verbose=config.verbose,
                        env=config.env_overrides or None,
                    )
                    if build_proc.returncode != 0:
                        logger.warning(
                            "Pre-hash `cargo build` failed (rc=%d) — the cache key "
                            "will describe whatever binary is currently on disk. "
                            "The extraction below runs the same build and will "
                            "surface the real error.\n%s",
                            build_proc.returncode,
                            _tail(build_proc.stderr or ""),
                        )
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(
                        "Pre-hash `cargo build` could not be run (%s); cache key "
                        "may describe a stale binary.",
                        exc,
                    )

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
                    result.captured_metrics["cache_seconds_saved"] = (
                        outcome.seconds_saved
                    )
                    result.captured_metrics["cache_linked_files"] = outcome.linked_files
                    result.captured_metrics["cache_link_type"] = outcome.link_type
                    # P1a producer provenance (finding A-PROV): a cache HIT means
                    # the current producer git shas equal the cached entry's (the
                    # shas are part of the cache key), so capturing them now is
                    # correct lineage for the linked data. Fail-open observation —
                    # never blocks the run; harvested by cli._record_experiment.
                    result.captured_metrics["producer_commits"] = (
                        resolve_build_provenance(
                            extractor_dir=config.paths.extractor_dir,
                            reconstructor_dir=config.paths.reconstructor_dir,
                        )
                    )
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
            "cargo",
            "run",
            "--release",
            "--bin",
            "export_dataset",
            "--features",
            # MUST match the pre-hash `cargo build` above — see
            # _EXTRACTOR_CARGO_FEATURES.
            _EXTRACTOR_CARGO_FEATURES,
            "--",
            "--config",
            str(config_path),
        ]

        start = time.monotonic()
        try:
            proc = run_subprocess(
                cmd,
                cwd=config.paths.extractor_dir,
                verbose=config.verbose,
                env=config.env_overrides or None,
            )
            result.duration_seconds = time.monotonic() - start
            result.stdout = _tail(proc.stdout or "")
            result.stderr = _tail(proc.stderr or "")

            if proc.returncode == 0:
                result.status = StageStatus.COMPLETED
                if stage.output_dir:
                    result.output_dir = str(config.paths.resolve(stage.output_dir))
                # P1a producer provenance (finding A-PROV): capture the producer
                # git state at extraction time — the correct instant for build
                # lineage of the data we just produced. Fail-open observation —
                # never blocks; harvested by cli._record_experiment.
                result.captured_metrics["producer_commits"] = resolve_build_provenance(
                    extractor_dir=config.paths.extractor_dir,
                    reconstructor_dir=config.paths.reconstructor_dir,
                )
            else:
                result.status = StageStatus.FAILED
                # Phase α-2 / #PY-80 (2026-05-10) — surface stderr.
                result.error_message = _format_subprocess_failure(
                    proc, "export_dataset"
                )
        except Exception as e:
            result.duration_seconds = time.monotonic() - start
            result.status = StageStatus.FAILED
            result.error_message = str(e)

        # -------- Enforce the output contract BEFORE cache publication ----
        # The driver (cli.py) also enforces this for every stage, but that
        # happens after `run()` returns — too late to stop a bad export from
        # being published into the content-addressed cache below, where it
        # would be silently re-linked into every future run with the same
        # cache key. Enforcing here flips `result.status` to FAILED, which the
        # `== COMPLETED` guard on the populate block then short-circuits.
        enforce_output_contract(self, manifest, config, result)

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
                logger.warning("Cache populate failed (extraction succeeded): %s", exc)

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

        # RECURSIVE globs (2026-08-03). These were non-recursive until the
        # postcondition was first actually WIRED into the stage driver, at
        # which point they turned out to be broken: an export dir is
        # `<export>/{train,val,test}/<day>_*.{npy,json}` — the per-day
        # artifacts live one level DOWN, never at the top level, which holds
        # only `dataset_manifest.json` / `export_config.toml`. Measured on
        # `data/exports/e5_timebased_60s_v3p0`: top-level glob → 0 / 0 / 0;
        # `**/` glob → 230 / 230 / 230. So the old form returned three
        # spurious violations for EVERY valid export on disk, and failing
        # closed on it would have broken every extraction run.
        #
        # `*_labels.npy` intentionally also matches `*_regression_labels.npy`
        # (fnmatch `*` spans `<day>_regression`), so the check is satisfied by
        # BOTH classification exports (`_labels.npy`) and regression exports
        # (which omit `_labels.npy` entirely and emit only
        # `_regression_labels.npy` — see root CLAUDE.md §Cross-Module Data
        # Contracts). Verified: 230 matches on a pure-regression export.
        meta_files = sorted(output_dir.glob("**/*_metadata.json"))
        if not meta_files:
            errors.append(f"No metadata JSON files under {output_dir}")

        seq_files = sorted(output_dir.glob("**/*_sequences.npy"))
        if not seq_files:
            errors.append(f"No sequence .npy files under {output_dir}")

        label_files = sorted(output_dir.glob("**/*_labels.npy"))
        if not label_files:
            errors.append(f"No label .npy files under {output_dir}")

        return errors
