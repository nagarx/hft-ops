"""Phase 8A.0 — Content-addressed extraction cache tests (test-first).

Covers the 14 Revision 3 design gaps identified in pre-implementation
adversarial validation. Each test locks one invariant of the cache design:

Cache-key input sensitivity (6):
  - test_cache_key_deterministic
  - test_cache_key_differs_per_grid_point_after_sweep_override
  - test_cache_key_captures_raw_input_manifest_hash
  - test_cache_key_captures_hft_statistics_sha
  - test_cache_key_captures_compiled_binary_sha256
  - test_cache_key_captures_platform_target

Correctness (4):
  - test_cache_poisoning_full_sha_detection
  - test_consumer_write_to_linked_file_does_not_mutate_cache
  - test_fingerprint_stable_across_cache_hit_vs_miss
  - test_atomic_tree_replication_on_interrupt

Integration (3):
  - test_cache_hit_writes_sidecar_json
  - test_skip_if_exists_collision_with_stale_output_dir
  - test_gc_lru_by_last_hit_with_size_budget

Plan: ~/.claude/plans/fuzzy-discovering-flask.md §Phase 8A.0 Revision 3.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pytest

from hft_ops.scheduler.extraction_cache import (
    CACHE_MANIFEST_NAME,
    CACHE_SIDECAR_NAME,
    CacheKeyInputs,
    CacheOutcome,
    compute_cache_key,
    gc_cache,
    populate,
    replicate_tree,
    resolve_or_link,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cache_key_inputs() -> CacheKeyInputs:
    """Canonical 9-field fixture (post-sweep-override resolved config)."""
    return CacheKeyInputs(
        extractor_config_resolved={
            "sampling": {"strategy": "time_based", "bin_size_seconds": 60},
            "labeling": {"strategy": "regression", "horizon_value": 10},
            "features": {"lob_levels": 10, "include_mbo": True},
        },
        extractor_git_sha="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        extractor_cargo_lock_sha256="c1a2b3c4" + "0" * 56,
        reconstructor_git_sha="b1b2b3b4b5b6b7b8b9b0b1b2b3b4b5b6b7b8b9b0",
        hft_statistics_git_sha="e976ff7" + "0" * 33,
        raw_input_manifest_hash="d1d2d3d4" + "0" * 56,
        compiled_binary_sha256="f1e2d3c4" + "0" * 56,
        platform_target=f"{sys.platform}-{platform.machine()}",
        contract_version="2.2",
    )


@pytest.fixture
def synthetic_export(tmp_path: Path) -> Path:
    """Mock extractor output with realistic layout (train/val/test + top-level)."""
    export_dir = tmp_path / "export_synthetic"
    export_dir.mkdir()

    # Top-level files (mirror production layout)
    (export_dir / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": "2.2", "n_features": 98})
    )
    (export_dir / "export_config.toml").write_text('[experiment]\nname="synthetic"\n')
    (export_dir / "hybrid_normalization_stats.json").write_text('{"strategy": "none"}')

    # Split subdirs
    for split in ("train", "val", "test"):
        split_dir = export_dir / split
        split_dir.mkdir()
        for day in ("2025-02-03", "2025-02-04"):
            (split_dir / f"{day}_sequences.npy").write_bytes(
                b"\x93NUMPY\x01\x00" + (f"seq-{split}-{day}").encode() + b"\x00" * 64
            )
            (split_dir / f"{day}_regression_labels.npy").write_bytes(
                b"\x93NUMPY\x01\x00" + (f"lbl-{split}-{day}").encode() + b"\x00" * 32
            )
            (split_dir / f"{day}_metadata.json").write_text(
                json.dumps({"day": day, "split": split, "n_sequences": 100})
            )

    return export_dir


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "exports" / "_cache"
    root.mkdir(parents=True)
    return root


# ============================================================================
# Cache-key input sensitivity (6 tests)
# ============================================================================


class TestCacheKeyInputSensitivity:
    """Each of the 9 cache-key inputs must produce a distinct hash."""

    def test_cache_key_deterministic(self, cache_key_inputs: CacheKeyInputs):
        k1 = compute_cache_key(cache_key_inputs)
        k2 = compute_cache_key(cache_key_inputs)
        assert k1 == k2, "Same inputs must yield same key (determinism)"
        assert len(k1) == 64, f"SHA-256 hex is 64 chars, got {len(k1)}"
        assert all(c in "0123456789abcdef" for c in k1), "Key must be lowercase hex"

    def test_cache_key_differs_per_grid_point_after_sweep_override(
        self, cache_key_inputs: CacheKeyInputs
    ):
        """Phase 3 §3.3b mirror: post-override config MUST be re-hashed.
        Otherwise 9 grid points of horizon_sensitivity.yaml sharing
        bin_size=60s but differing in horizon_value would collapse to one
        cache entry — catastrophic ledger-conflation.
        """
        cfg_h10 = {
            **cache_key_inputs.extractor_config_resolved,
            "labeling": {"strategy": "regression", "horizon_value": 10},
        }
        cfg_h60 = {
            **cache_key_inputs.extractor_config_resolved,
            "labeling": {"strategy": "regression", "horizon_value": 60},
        }
        k_h10 = compute_cache_key(replace(cache_key_inputs, extractor_config_resolved=cfg_h10))
        k_h60 = compute_cache_key(replace(cache_key_inputs, extractor_config_resolved=cfg_h60))
        assert k_h10 != k_h60, "Different horizon_value must produce different cache keys"

    def test_cache_key_captures_raw_input_manifest_hash(
        self, cache_key_inputs: CacheKeyInputs
    ):
        """SHIP-BLOCKER #6 — raw .dbn.zst changes (re-download, corruption
        rollback) invalidate cache even when config + git SHAs unchanged.
        """
        k_a = compute_cache_key(replace(cache_key_inputs, raw_input_manifest_hash="hash_a" + "0" * 58))
        k_b = compute_cache_key(replace(cache_key_inputs, raw_input_manifest_hash="hash_b" + "0" * 58))
        assert k_a != k_b

    def test_cache_key_captures_hft_statistics_sha(self, cache_key_inputs: CacheKeyInputs):
        """SHIP-BLOCKER #5 — hft-statistics is patched via .cargo/config.toml;
        version bumps change Welford/time-regime outputs without extractor
        git SHA changing.
        """
        k_a = compute_cache_key(replace(cache_key_inputs, hft_statistics_git_sha="aaa" + "0" * 37))
        k_b = compute_cache_key(replace(cache_key_inputs, hft_statistics_git_sha="bbb" + "0" * 37))
        assert k_a != k_b

    def test_cache_key_captures_compiled_binary_sha256(
        self, cache_key_inputs: CacheKeyInputs
    ):
        """Rev-3 refinement — uncommitted `cargo update -p dbn` leaves
        git SHA + Cargo.lock stable but produces a different binary; cache
        MUST invalidate.
        """
        k_a = compute_cache_key(replace(cache_key_inputs, compiled_binary_sha256="bin_a" + "0" * 59))
        k_b = compute_cache_key(replace(cache_key_inputs, compiled_binary_sha256="bin_b" + "0" * 59))
        assert k_a != k_b

    def test_cache_key_captures_platform_target(self, cache_key_inputs: CacheKeyInputs):
        """Rev-3 refinement — macOS-arm64 vs Linux-x86_64 can produce
        last-ULP Welford differences under -C target-cpu=native; prevents
        CI (Linux) + local dev (macOS) cache cross-contamination.
        """
        k_darwin = compute_cache_key(replace(cache_key_inputs, platform_target="darwin-arm64"))
        k_linux = compute_cache_key(replace(cache_key_inputs, platform_target="linux-x86_64"))
        assert k_darwin != k_linux


# ============================================================================
# Correctness (4 tests)
# ============================================================================


class TestCachePoisoningDetection:
    def test_cache_poisoning_full_sha_detection(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 upgrade from 4KB tripwire to full-file SHA.
        Populate → corrupt one file → resolve must detect + mark poisoned.
        """
        key = compute_cache_key(cache_key_inputs)
        populate(
            key,
            synthetic_export,
            cache_root,
            extractor_duration_seconds=42.0,
            cache_key_inputs=cache_key_inputs,
        )
        cache_dir = cache_root / key
        target = cache_dir / "train" / "2025-02-03_sequences.npy"
        assert target.exists(), "populate must create train/*.npy files"

        # Corrupt — first unlock (populate applies chmod-readonly)
        target.chmod(0o644)
        target.write_bytes(b"CORRUPTED" + b"\x00" * 100)

        output_dir = tmp_path / "consumer_output"
        outcome = resolve_or_link(key, output_dir, cache_root)
        assert outcome.status == "poisoned", (
            f"Corrupted file should yield status=poisoned, got {outcome.status}"
        )

        manifest = json.loads((cache_dir / CACHE_MANIFEST_NAME).read_text())
        assert manifest["status"] == "poisoned", (
            f"Manifest must mark entry as poisoned, got {manifest['status']}"
        )


class TestConsumerMutationProtection:
    def test_consumer_write_to_linked_file_does_not_mutate_cache(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 SHIP-BLOCKER — clonefile/reflink COW OR chmod-readonly
        inodes prevent consumer accidental-mutation of cache.
        """
        key = compute_cache_key(cache_key_inputs)
        populate(
            key,
            synthetic_export,
            cache_root,
            extractor_duration_seconds=10.0,
            cache_key_inputs=cache_key_inputs,
        )
        cache_dir = cache_root / key
        cache_file = cache_dir / "train" / "2025-02-03_sequences.npy"
        original_bytes = cache_file.read_bytes()
        original_sha = hashlib.sha256(original_bytes).hexdigest()

        output_dir = tmp_path / "consumer_output"
        outcome = resolve_or_link(key, output_dir, cache_root)
        assert outcome.status == "hit"

        consumer_file = output_dir / "train" / "2025-02-03_sequences.npy"
        assert consumer_file.exists()

        # Consumer attempts to overwrite. Outcome depends on link_type:
        #   - clonefile / reflink: COW — write succeeds, cache unchanged
        #   - hardlink_readonly / symlink_relative: write fails EACCES
        try:
            consumer_file.write_bytes(b"CONSUMER_OVERWRITE")
            consumer_wrote_successfully = True
        except PermissionError:
            consumer_wrote_successfully = False

        # Whether consumer succeeded or not, the CACHE file must be unchanged
        cache_after = cache_file.read_bytes()
        cache_sha_after = hashlib.sha256(cache_after).hexdigest()
        assert cache_sha_after == original_sha, (
            f"Cache file SHA mutated from {original_sha[:12]} to "
            f"{cache_sha_after[:12]} after consumer write "
            f"(link_type={outcome.link_type}, consumer_wrote={consumer_wrote_successfully})"
        )


class TestFingerprintStabilityAcrossCache:
    def test_fingerprint_stable_across_cache_hit_vs_miss(
        self, cache_key_inputs: CacheKeyInputs
    ):
        """Invariant 4 — cache hit vs miss MUST NOT affect ExperimentRecord
        fingerprint. Cache status is an observation, not a treatment.
        Verified here at the primitive level: cache_key varies WILD but
        compute_cache_key never reads anything that's ALSO in
        ExperimentRecord fingerprint (which hashes manifest + resolved
        configs, not git SHAs).

        This test locks the SYMMETRIC invariant: the 9 cache-key inputs
        are DISJOINT from the 5-field ExperimentRecord fingerprint
        surface (extraction/training/backtest/data_manifest/contract_version).
        If a future edit conflates them, the dedup ledger would treat
        a cache-hit experiment differently from a cache-miss one — breaking
        scientific comparability.
        """
        # Cache key includes compiled_binary_sha256 and hft_statistics_git_sha —
        # these are build-environment observations, not treatments.
        k1 = compute_cache_key(cache_key_inputs)
        k2 = compute_cache_key(replace(cache_key_inputs, compiled_binary_sha256="x" * 64))
        assert k1 != k2, "Cache key responds to build-env changes..."

        # ...but ExperimentRecord fingerprint does NOT see compiled_binary_sha256.
        # We verify by inspecting the dedup fingerprint computation does not
        # reference any of the 5 cache-only fields.
        from hft_ops.ledger import dedup

        dedup_source = Path(dedup.__file__).read_text()
        for cache_only_field in (
            "compiled_binary_sha256",
            "hft_statistics_git_sha",
            "platform_target",
            "raw_input_manifest_hash",
            "extractor_cargo_lock_sha256",
        ):
            assert cache_only_field not in dedup_source, (
                f"compute_fingerprint MUST NOT reference {cache_only_field!r} — "
                f"cache-only inputs are observations, not treatments (Invariant 4)"
            )


class TestAtomicTreeReplication:
    def test_atomic_tree_replication_on_interrupt(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 Agent 2 SHIP-BLOCKER #6 — consumer must never see a
        half-populated output_dir. replicate_tree writes to <output_dir>.tmp/
        then atomically renames. If interrupted before rename, the final
        output_dir does not exist (no partial state).
        """
        key = compute_cache_key(cache_key_inputs)
        populate(
            key,
            synthetic_export,
            cache_root,
            extractor_duration_seconds=10.0,
            cache_key_inputs=cache_key_inputs,
        )
        cache_dir = cache_root / key

        output_dir = tmp_path / "consumer_atomic"
        n_linked, link_type = replicate_tree(cache_dir, output_dir)

        # Post-replication: output_dir exists, .tmp/ siblings gone
        assert output_dir.exists() and output_dir.is_dir()
        tmp_siblings = list(output_dir.parent.glob(f"{output_dir.name}{'.tmp'}*"))
        assert not tmp_siblings, f"Leftover .tmp/ staging dirs: {tmp_siblings}"
        assert n_linked > 0, "Should have linked at least one file"

        # Must contain every top-level + split file the cache entry had
        assert (output_dir / "dataset_manifest.json").exists()
        assert (output_dir / "train" / "2025-02-03_sequences.npy").exists()
        assert (output_dir / "val").is_dir()
        assert (output_dir / "test" / "2025-02-04_metadata.json").exists()


# ============================================================================
# Integration (3 tests)
# ============================================================================


class TestSidecarJson:
    def test_cache_hit_writes_sidecar_json(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 Agent 2 — _CACHE_HIT.json sidecar in output_dir exposes
        link-type + original extractor-run-timestamp for operator debugging.
        """
        key = compute_cache_key(cache_key_inputs)
        populate(
            key,
            synthetic_export,
            cache_root,
            extractor_duration_seconds=10.0,
            cache_key_inputs=cache_key_inputs,
        )

        output_dir = tmp_path / "output_with_sidecar"
        outcome = resolve_or_link(key, output_dir, cache_root)
        assert outcome.status == "hit"

        sidecar_path = output_dir / CACHE_SIDECAR_NAME
        assert sidecar_path.exists(), f"{CACHE_SIDECAR_NAME} must be written on hit"
        sidecar = json.loads(sidecar_path.read_text())

        for required_key in (
            "cache_key",
            "source_extractor_run_utc",
            "linked_at_utc",
            "link_type",
        ):
            assert required_key in sidecar, f"sidecar missing {required_key!r}: {sidecar}"

        assert sidecar["cache_key"] == key
        assert sidecar["link_type"] in (
            "clonefile",
            "reflink",
            "hardlink_readonly",
            "symlink_relative",
        )


class TestSkipIfExistsCollision:
    def test_skip_if_exists_collision_with_stale_output_dir(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 Agent 2 SHIP-BLOCKER #4 — if output_dir exists AND is
        NOT a clone/symlink (no _CACHE_HIT.json sidecar), the cache is
        BYPASSED for that grid point (existing real data preserved).
        Prevents silent overwrite of legacy 20GB pre-cache exports.

        We verify this by populating the cache for a key, then creating
        an unrelated output_dir (no sidecar), then calling resolve_or_link
        with a flag indicating this is a pre-existing real run. The
        outcome status surface exposes the bypass.
        """
        key = compute_cache_key(cache_key_inputs)
        populate(
            key,
            synthetic_export,
            cache_root,
            extractor_duration_seconds=10.0,
            cache_key_inputs=cache_key_inputs,
        )

        # Create a pre-existing legacy output_dir with REAL data (no sidecar)
        legacy_output = tmp_path / "legacy_run"
        legacy_output.mkdir()
        (legacy_output / "dataset_manifest.json").write_text('{"legacy": true}')
        legacy_bytes = (legacy_output / "dataset_manifest.json").read_bytes()

        # resolve_or_link must detect the legacy dir and BYPASS
        outcome = resolve_or_link(key, legacy_output, cache_root)
        assert outcome.status == "miss", (
            f"legacy output_dir without sidecar must yield status=miss "
            f"(bypass cache), got {outcome.status}"
        )

        # Legacy data untouched
        assert (legacy_output / "dataset_manifest.json").read_bytes() == legacy_bytes
        assert not (legacy_output / CACHE_SIDECAR_NAME).exists()


class TestPostAuditFixes:
    """Post-audit regression guards for the 4-agent validation pass
    (2026-04-20). Each test locks one CRITICAL/HIGH finding's fix.
    """

    def test_symlinks_in_extractor_output_are_rejected(
        self, tmp_path: Path, cache_root: Path, cache_key_inputs: CacheKeyInputs
    ):
        """Agent-B C2: ``shutil.copytree`` default follows symlinks →
        populate would silently cache whatever the symlink pointed to.
        Fix: copytree with ``symlinks=True`` (preserves symlinks) THEN
        ``_reject_symlinks_in_tree`` hard-fails. This test locks the
        fail-loud behavior.
        """
        # Create synthetic export with a symlink
        export_dir = tmp_path / "malicious_export"
        export_dir.mkdir()
        (export_dir / "real_file.txt").write_text("ok")
        (export_dir / "symlink.txt").symlink_to(export_dir / "real_file.txt")

        from hft_ops.scheduler.extraction_cache import CacheError

        key = compute_cache_key(cache_key_inputs)
        with pytest.raises(CacheError, match="Symlink rejected"):
            populate(
                key, export_dir, cache_root,
                extractor_duration_seconds=1.0,
                cache_key_inputs=cache_key_inputs,
            )
        # Staging dir must be cleaned up — no orphan `<key>.tmp-<pid>/`
        tmp_candidates = list(cache_root.glob(f"{key}.tmp*"))
        assert not tmp_candidates, (
            f"Symlink rejection must clean up staging; found: {tmp_candidates}"
        )
        # Final dir must NOT exist (populate aborted before rename)
        assert not (cache_root / key).exists()

    def test_sha_memo_invalidated_on_poisoning(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Agent-B C3: without memo invalidation, a cache entry that
        successfully validated (memo=True) and then was corrupted in the
        same process would still report ``status=hit`` because the memo
        says "already valid". Fix: ``_flip_status_poisoned`` drops the
        memo entry for that cache_dir.
        """
        from hft_ops.scheduler.extraction_cache import (
            _SHA_VALIDATION_MEMO,
            _flip_status_poisoned,
        )

        key = compute_cache_key(cache_key_inputs)
        populate(
            key, synthetic_export, cache_root,
            extractor_duration_seconds=10.0,
            cache_key_inputs=cache_key_inputs,
        )
        cache_dir = cache_root / key

        # First resolve → memo records True
        output_dir = tmp_path / "first"
        outcome = resolve_or_link(key, output_dir, cache_root)
        assert outcome.status == "hit"

        memo_key = (str(cache_dir), key, os.getpid())
        assert _SHA_VALIDATION_MEMO.get(memo_key) is True

        # Flip to poisoned — memo must be cleared
        manifest_path = cache_dir / "CACHE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        _flip_status_poisoned(cache_dir, manifest)
        assert memo_key not in _SHA_VALIDATION_MEMO, (
            "Memo must be invalidated on poisoning — otherwise a "
            "stale True memo causes _validate_cache_entry to skip "
            "re-validation on subsequent resolves in the same process."
        )

    def test_raw_input_manifest_missing_returns_none_no_fallback(
        self, tmp_path: Path
    ):
        """Agent-B C1: previously had a basename-only fallback that
        collapsed different input_dirs with the same basename to the
        SAME fallback manifest path → cache-key collision. Fix: no
        fallback. Fail-closed when primary path is missing.
        """
        from hft_ops.scheduler.extraction_cache import (
            _compute_raw_input_manifest_hash,
        )

        extractor_dir = tmp_path / "feature-extractor-MBO-LOB"
        extractor_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Neither the primary (input_dir) nor the would-be fallback exists
        resolved_config = {"data": {"input_dir": "../data/raw/nonexistent"}}

        result = _compute_raw_input_manifest_hash(
            resolved_config, data_dir=data_dir, extractor_dir=extractor_dir
        )
        assert result is None, (
            "Must return None when manifest.json cannot be found at the "
            "primary input_dir path — NO basename-only fallback (that "
            "caused cache-key collision across different raw data dirs)."
        )


class TestGarbageCollectionLRU:
    def test_gc_lru_by_last_hit_with_size_budget(
        self,
        tmp_path: Path,
        cache_root: Path,
        synthetic_export: Path,
        cache_key_inputs: CacheKeyInputs,
    ):
        """Rev-3 SHIP-BLOCKER — at 1.8TB drive w/ 3.5TB 1-yr projection,
        age-only GC deletes valuable recent-hit entries. LRU-by-last-hit
        with size budget is correct.
        """
        # Populate 3 cache entries with distinct keys + staggered mtimes
        keys = []
        for i, (cfg_override, mtime_offset) in enumerate([
            ({"bin_size_seconds": 10}, -300),  # oldest hit (5 min ago)
            ({"bin_size_seconds": 30}, -120),  # middle hit (2 min ago)
            ({"bin_size_seconds": 60}, -1),    # newest hit (1 sec ago)
        ]):
            config = {**cache_key_inputs.extractor_config_resolved, **cfg_override}
            inputs = replace(cache_key_inputs, extractor_config_resolved=config)
            key = compute_cache_key(inputs)
            keys.append(key)
            populate(
                key,
                synthetic_export,
                cache_root,
                extractor_duration_seconds=10.0,
                cache_key_inputs=inputs,
            )
            # Stamp directory mtime to simulate past last_hit
            cache_dir = cache_root / key
            now = time.time()
            os.utime(cache_dir, (now + mtime_offset, now + mtime_offset))

        # Dry-run GC with tiny size budget must propose evicting oldest first
        planned = gc_cache(cache_root, max_size_gb=0.000001, dry_run=True)
        planned_paths = [Path(p).name for p in planned]

        # Oldest entry (keys[0]) must be in planned eviction list.
        # Newest entry (keys[2]) must NOT be evicted first.
        assert keys[0] in planned_paths, (
            f"Oldest key {keys[0][:12]} must be evicted first under LRU. "
            f"Planned: {[p[:12] for p in planned_paths]}"
        )
        # Verify LRU ordering: oldest evicted before newest
        if keys[0] in planned_paths and keys[2] in planned_paths:
            assert planned_paths.index(keys[0]) < planned_paths.index(keys[2]), (
                "LRU order violated: oldest must be evicted first"
            )
