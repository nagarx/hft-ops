"""Phase 8A.0 — Content-addressed extraction cache.

Eliminates redundant extractor subprocess invocations when grid points share
extraction-config inputs. Cache key hashes 9 orthogonal inputs so behavior
tracks the full TREATMENT (resolved config — post-sweep-override, post-_base
merge) AND the BUILD ENVIRONMENT (git SHAs + dep hashes + compiled binary +
platform target):

  1. extractor_config_resolved — POST-expansion config dict
  2. extractor_git_sha          — HEAD of feature-extractor-MBO-LOB
  3. extractor_cargo_lock_sha   — full Cargo.lock SHA-256 (transitive deps)
  4. reconstructor_git_sha      — HEAD of MBO-LOB-reconstructor
  5. hft_statistics_git_sha     — HEAD of hft-statistics (Welford/regime SSoT)
  6. raw_input_manifest_hash    — databento-ingest manifest SHA-256
  7. compiled_binary_sha256     — target/release/export_dataset
  8. platform_target            — e.g. "darwin-arm64" (Welford last-ULP)
  9. contract_version           — hft_contracts.SCHEMA_VERSION

See plan file ``~/.claude/plans/fuzzy-discovering-flask.md`` §Phase 8A.0
Revision 3 for the full 14-findings design audit.

----------------------------------------------------------------------------
Layout
----------------------------------------------------------------------------

Cache root: ``<pipeline_root>/data/exports/_cache/``

Per-entry structure mirrors the extractor's actual output (train/val/test
subdirs + top-level ``dataset_manifest.json`` / ``export_config.toml`` /
optional ``hybrid_normalization_stats.json``). Cache is ADAPTIVE — snapshots
whatever the extractor produced; does not assume a fixed file set.

::

    <cache_root>/<sha256_64hex>/
      CACHE_MANIFEST.json          — populate() writes; RO after finalize
      dataset_manifest.json
      export_config.toml
      hybrid_normalization_stats.json   (if present in source)
      train/
        <day>_sequences.npy
        <day>_regression_labels.npy    (regression) or <day>_labels.npy (class)
        <day>_metadata.json
        <day>_normalization.json
        <day>_forward_prices.npy
      val/  ...
      test/ ...

Finalized entries are chmod-readonly (mode 0o444 files, 0o555 dirs) to
block consumer accidental-mutation. Works uniformly for:

  - APFS (macOS): ``cp -c -R`` (clonefile) — COW means consumer writes
    already create copies; chmod-readonly is defense-in-depth.
  - Btrfs / XFS (Linux): ``cp --reflink=always -R`` — same COW semantics.
  - ext4 (Linux): ``os.link`` hardlink — chmod-readonly is LOAD-BEARING
    (hardlinks share inodes; without readonly, consumer write silently
    corrupts cache).
  - Cross-filesystem: relative symlink — opening the symlink in write
    mode follows to target; chmod-readonly on target makes write fail
    EACCES, protecting cache.

----------------------------------------------------------------------------
Observability + GC
----------------------------------------------------------------------------

On each ``resolve_or_link`` HIT, we:

  - Validate size (cheap) + full-file SHA-256 (per-process-memoized; ~2s
    once per process per cache key; zero-cost on subsequent hits).
  - Touch cache_dir mtime via ``os.utime(cache_dir, None)`` — authoritative
    LRU signal (no JSON-write contention; filesystem-atomic).
  - Opportunistically update ``CACHE_MANIFEST.json::last_hit_at_utc``
    (best-effort; not load-bearing; filesystem-only-partial-write OK).
  - Write ``<output_dir>/_CACHE_HIT.json`` sidecar exposing cache_key +
    link_type + original extractor timestamp for operator debugging.

GC policy: LRU-by-mtime + size budget + pin-list (``_PINNED.json``).
Combined filters apply with AND semantics.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from hft_contracts.atomic_io import atomic_write_json
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

logger = logging.getLogger(__name__)


__all__ = [
    "CACHE_MANIFEST_SCHEMA_VERSION",
    "CACHE_MANIFEST_NAME",
    "CACHE_SIDECAR_NAME",
    "PINNED_FILE_NAME",
    "CacheKeyInputs",
    "CacheOutcome",
    "CacheError",
    "CachePoisonedError",
    "compute_cache_key",
    "prepare_cache_key_inputs",
    "resolve_or_link",
    "populate",
    "replicate_tree",
    "gc_cache",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_MANIFEST_SCHEMA_VERSION: str = "1.0"
CACHE_MANIFEST_NAME: str = "CACHE_MANIFEST.json"
CACHE_SIDECAR_NAME: str = "_CACHE_HIT.json"
PINNED_FILE_NAME: str = "_PINNED.json"

_TMP_SUFFIX: str = ".tmp"  # suffix for atomic-rename staging dirs
_READONLY_FILE_MODE: int = 0o444
_READONLY_DIR_MODE: int = 0o555

# Per-process SHA memoization cache.
# Key: (cache_dir_str, cache_key, pid) → bool (valid).
# Cleared on process restart. Keyed by cache_dir (not just cache_key) so two
# tests with the same cache_key in different tmp_paths don't collide, AND so
# a test-mutation-then-revalidation pattern (e.g. poisoning tests) doesn't
# falsely persist "invalid" memo into an unrelated test using the same key.
_SHA_VALIDATION_MEMO: Dict[Tuple[str, str, int], bool] = {}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CacheError(ValueError):
    """Root exception for extraction-cache errors."""


class CachePoisonedError(CacheError):
    """Cache entry failed SHA-256 validation on a file. Entry is marked
    ``status=poisoned`` in CACHE_MANIFEST.json; operator must decide whether
    to clear or re-extract (the latter is automatic via fall-through).
    """


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKeyInputs:
    """The 9 canonical inputs to the cache-key SHA-256.

    Field order is part of the canonical-JSON contract: changing it breaks
    hash stability. New inputs MUST be added as a new field (additive) AND
    bump ``CACHE_MANIFEST_SCHEMA_VERSION`` MAJOR (breaks existing cache).
    """

    extractor_config_resolved: Dict[str, Any]
    extractor_git_sha: str
    extractor_cargo_lock_sha256: str
    reconstructor_git_sha: str
    hft_statistics_git_sha: str
    raw_input_manifest_hash: str
    compiled_binary_sha256: str
    platform_target: str
    contract_version: str


@dataclass(frozen=True)
class CacheOutcome:
    """Result of a ``resolve_or_link`` call.

    status semantics:
      - ``hit``  — cache entry found, SHA-validated, linked/replicated to
        output_dir with sidecar written.
      - ``miss`` — no cache entry (or output_dir is legacy non-cache data
        per TestSkipIfExistsCollision); caller should run the extractor.
      - ``poisoned`` — cache entry exists but SHA validation failed; entry
        marked ``status=poisoned`` in CACHE_MANIFEST.json; caller should
        re-extract (auto-invalidates bad entry on next populate).
    """

    status: Literal["hit", "miss", "poisoned"]
    cache_key: str
    cache_dir: Optional[Path]
    output_dir: Path
    linked_files: int
    seconds_saved: float
    link_type: Optional[Literal["clonefile", "reflink", "hardlink_readonly", "symlink_relative"]]


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def compute_cache_key(inputs: CacheKeyInputs) -> str:
    """Return 64-char SHA-256 hex of canonical-JSON blob of 9 inputs.

    Uses ``hft_contracts.canonical_hash`` SSoT — ZERO re-derivation.
    """
    payload = {
        "extractor_config_resolved": inputs.extractor_config_resolved,
        "extractor_git_sha": inputs.extractor_git_sha,
        "extractor_cargo_lock_sha256": inputs.extractor_cargo_lock_sha256,
        "reconstructor_git_sha": inputs.reconstructor_git_sha,
        "hft_statistics_git_sha": inputs.hft_statistics_git_sha,
        "raw_input_manifest_hash": inputs.raw_input_manifest_hash,
        "compiled_binary_sha256": inputs.compiled_binary_sha256,
        "platform_target": inputs.platform_target,
        "contract_version": inputs.contract_version,
    }
    blob = canonical_json_blob(payload)
    return sha256_hex(blob)


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------


def populate(
    cache_key: str,
    src_output_dir: Path,
    cache_root: Path,
    *,
    extractor_duration_seconds: float,
    cache_key_inputs: CacheKeyInputs,
) -> Path:
    """Atomically install ``src_output_dir``'s tree into the cache entry.

    Workflow:
      1. Copy src_output_dir → ``<cache_root>/<cache_key>.tmp-<pid>/``
      2. Write ``CACHE_MANIFEST.json`` into the staging dir
      3. Apply chmod-readonly recursively (files 0o444, dirs 0o555)
      4. ``os.rename(tmp, <cache_root>/<cache_key>)`` — POSIX-atomic
         (last-to-rename wins under concurrent same-key populate)

    PID-suffixed tmp staging prevents collisions when multiple workers
    race on the same cache_key. Extractor is deterministic under
    identical CacheKeyInputs, so all winners produce identical content.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    final_dir = cache_root / cache_key
    if final_dir.exists():
        # Another worker populated concurrently; skip (extractor is
        # deterministic — content is identical).
        return final_dir

    staging_dir = cache_root / f"{cache_key}{_TMP_SUFFIX}-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Copy full tree (mutable — will chmod-readonly after manifest write)
    shutil.copytree(src_output_dir, staging_dir)

    # Build + write CACHE_MANIFEST.json in staging BEFORE chmod
    files_manifest = _build_files_manifest(staging_dir)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "cache_key": cache_key,
        "created_at_utc": now_iso,
        "last_hit_at_utc": now_iso,  # self-hit at creation (LRU seed)
        "cache_key_inputs": {
            "extractor_git_sha": cache_key_inputs.extractor_git_sha,
            "extractor_cargo_lock_sha256": cache_key_inputs.extractor_cargo_lock_sha256,
            "reconstructor_git_sha": cache_key_inputs.reconstructor_git_sha,
            "hft_statistics_git_sha": cache_key_inputs.hft_statistics_git_sha,
            "raw_input_manifest_hash": cache_key_inputs.raw_input_manifest_hash,
            "compiled_binary_sha256": cache_key_inputs.compiled_binary_sha256,
            "platform_target": cache_key_inputs.platform_target,
            "contract_version": cache_key_inputs.contract_version,
        },
        "extractor_duration_seconds": extractor_duration_seconds,
        "files": files_manifest,
        "status": "active",
    }
    atomic_write_json(staging_dir / CACHE_MANIFEST_NAME, manifest)

    # Atomic finalize FIRST (while staging is still writable — chmod-readonly
    # BEFORE rename breaks ``os.rename`` on macOS/APFS because removing
    # write permission on the directory itself blocks entry-modification
    # even when the parent is writable).
    try:
        os.rename(staging_dir, final_dir)
    except OSError:
        # Lost the race to another worker (final_dir now exists); clean up
        # our staging (which is still writable at this point).
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        return final_dir

    # Apply chmod-readonly AFTER rename so the cache inode itself is locked
    # (manifest becomes RO too; ``_flip_status_poisoned`` temporarily unlocks
    # to flip status).
    _chmod_readonly_recursive(final_dir)

    return final_dir


def _build_files_manifest(staging_dir: Path) -> List[Dict[str, Any]]:
    """Walk the staging dir, compute per-file SHA-256 + size.

    Intentionally excludes ``CACHE_MANIFEST.json`` itself (written later,
    would create chicken-and-egg).
    """
    entries: List[Dict[str, Any]] = []
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == CACHE_MANIFEST_NAME:
            continue
        rel = path.relative_to(staging_dir).as_posix()
        data = path.read_bytes()
        entries.append({
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })
    return entries


def _chmod_readonly_recursive(root: Path) -> None:
    """Files → 0o444, dirs → 0o555. Applied post-populate before atomic rename."""
    for path in root.rglob("*"):
        if path.is_file():
            try:
                os.chmod(path, _READONLY_FILE_MODE)
            except OSError:
                # Non-fatal — symlinks on some platforms can't chmod. Log + continue.
                logger.debug("chmod(0o444) failed for %s", path)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                os.chmod(path, _READONLY_DIR_MODE)
            except OSError:
                logger.debug("chmod(0o555) failed for %s", path)
    # Root dir itself
    try:
        os.chmod(root, _READONLY_DIR_MODE)
    except OSError:
        logger.debug("chmod(0o555) failed for %s", root)


def _rm_readonly_tree(root: Path) -> None:
    """Helper: rmtree a chmod-readonly tree (used for cleanup + flip-poisoned)."""
    # Make everything writable first
    for path in root.rglob("*"):
        try:
            os.chmod(path, 0o777)
        except OSError:
            pass
    try:
        os.chmod(root, 0o777)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Resolve + link
# ---------------------------------------------------------------------------


def resolve_or_link(
    cache_key: str,
    output_dir: Path,
    cache_root: Path,
) -> CacheOutcome:
    """Check cache; on hit, replicate tree to output_dir atomically.

    Collision rule (Rev 3 Agent 2 SHIP-BLOCKER #4): if output_dir exists
    AND has no ``_CACHE_HIT.json`` sidecar, treat as legacy real data and
    return status=miss (caller keeps existing data, does not run extractor
    unless caller chose to).
    """
    cache_dir = cache_root / cache_key

    # Collision check: pre-existing output_dir WITHOUT sidecar → bypass
    if output_dir.exists() and not (output_dir / CACHE_SIDECAR_NAME).exists():
        # Legacy pre-cache run — don't clobber, don't cache-hit, don't re-link
        logger.info(
            "Cache bypass: %s contains legacy real data (no %s); "
            "preserving existing files.",
            output_dir,
            CACHE_SIDECAR_NAME,
        )
        return CacheOutcome(
            status="miss",
            cache_key=cache_key,
            cache_dir=None,
            output_dir=output_dir,
            linked_files=0,
            seconds_saved=0.0,
            link_type=None,
        )

    if not cache_dir.exists():
        return CacheOutcome(
            status="miss",
            cache_key=cache_key,
            cache_dir=None,
            output_dir=output_dir,
            linked_files=0,
            seconds_saved=0.0,
            link_type=None,
        )

    # Validate full SHA on the cache entry (memoized per-process)
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Cache %s has malformed %s — treating as miss. Error: %s",
            cache_key[:12],
            CACHE_MANIFEST_NAME,
            exc,
        )
        return CacheOutcome(
            status="miss",
            cache_key=cache_key,
            cache_dir=None,
            output_dir=output_dir,
            linked_files=0,
            seconds_saved=0.0,
            link_type=None,
        )

    if manifest.get("status") == "poisoned":
        return CacheOutcome(
            status="poisoned",
            cache_key=cache_key,
            cache_dir=cache_dir,
            output_dir=output_dir,
            linked_files=0,
            seconds_saved=0.0,
            link_type=None,
        )

    validation_ok = _validate_cache_entry(cache_key, cache_dir, manifest)
    if not validation_ok:
        _flip_status_poisoned(cache_dir, manifest)
        logger.warning(
            "Cache %s failed SHA validation — marked poisoned, falling "
            "through to extraction.",
            cache_key[:12],
        )
        return CacheOutcome(
            status="poisoned",
            cache_key=cache_key,
            cache_dir=cache_dir,
            output_dir=output_dir,
            linked_files=0,
            seconds_saved=0.0,
            link_type=None,
        )

    # LRU: touch cache_dir mtime (filesystem-atomic; no lock needed)
    try:
        os.utime(cache_dir, None)
    except OSError:
        # Read-only filesystem — acceptable; mtime stale but cache still works.
        pass

    # Best-effort CACHE_MANIFEST.json last_hit update (not load-bearing)
    _update_last_hit_opportunistic(cache_dir, manifest)

    # Replicate tree → output_dir atomically
    n_linked, link_type = replicate_tree(cache_dir, output_dir)

    # Write sidecar for debugging
    _write_sidecar(
        output_dir=output_dir,
        cache_key=cache_key,
        source_extractor_run_utc=manifest.get("created_at_utc", ""),
        link_type=link_type,
    )

    seconds_saved = float(manifest.get("extractor_duration_seconds", 0.0))
    return CacheOutcome(
        status="hit",
        cache_key=cache_key,
        cache_dir=cache_dir,
        output_dir=output_dir,
        linked_files=n_linked,
        seconds_saved=seconds_saved,
        link_type=link_type,
    )


def _validate_cache_entry(
    cache_key: str,
    cache_dir: Path,
    manifest: Dict[str, Any],
) -> bool:
    """Full-file SHA-256 validation, per-process memoized.

    Returns True if all files pass; False on any size/SHA mismatch.
    """
    memo_key = (str(cache_dir), cache_key, os.getpid())
    if memo_key in _SHA_VALIDATION_MEMO:
        return _SHA_VALIDATION_MEMO[memo_key]

    files = manifest.get("files", [])
    for entry in files:
        rel = entry["path"]
        expected_sha = entry["sha256"]
        expected_size = entry["size_bytes"]
        path = cache_dir / rel
        if not path.exists():
            logger.warning("Cache file missing: %s", path)
            _SHA_VALIDATION_MEMO[memo_key] = False
            return False
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            logger.warning(
                "Cache file size mismatch: %s (expected %d, got %d)",
                path,
                expected_size,
                actual_size,
            )
            _SHA_VALIDATION_MEMO[memo_key] = False
            return False
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            logger.warning(
                "Cache file SHA mismatch: %s (expected %s, got %s)",
                path,
                expected_sha[:12],
                actual_sha[:12],
            )
            _SHA_VALIDATION_MEMO[memo_key] = False
            return False

    _SHA_VALIDATION_MEMO[memo_key] = True
    return True


def _flip_status_poisoned(cache_dir: Path, manifest: Dict[str, Any]) -> None:
    """Mark CACHE_MANIFEST.json status=poisoned. Temporarily unlocks RO file."""
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    try:
        os.chmod(cache_dir, 0o755)
        os.chmod(manifest_path, 0o644)
        manifest["status"] = "poisoned"
        atomic_write_json(manifest_path, manifest)
    except OSError as exc:
        logger.warning("Could not flip cache %s to poisoned: %s", cache_dir.name, exc)
    finally:
        try:
            os.chmod(manifest_path, _READONLY_FILE_MODE)
            os.chmod(cache_dir, _READONLY_DIR_MODE)
        except OSError:
            pass


def _update_last_hit_opportunistic(cache_dir: Path, manifest: Dict[str, Any]) -> None:
    """Best-effort update of last_hit_at_utc. NOT load-bearing — mtime is
    authoritative for LRU. Quiet-fails on RO filesystem / permission issues.
    """
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    try:
        os.chmod(cache_dir, 0o755)
        os.chmod(manifest_path, 0o644)
        manifest["last_hit_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(manifest_path, manifest)
    except OSError:
        pass
    finally:
        try:
            os.chmod(manifest_path, _READONLY_FILE_MODE)
            os.chmod(cache_dir, _READONLY_DIR_MODE)
        except OSError:
            pass


def _write_sidecar(
    output_dir: Path,
    cache_key: str,
    source_extractor_run_utc: str,
    link_type: str,
) -> None:
    sidecar = {
        "cache_key": cache_key,
        "source_extractor_run_utc": source_extractor_run_utc,
        "linked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "link_type": link_type,
    }
    atomic_write_json(output_dir / CACHE_SIDECAR_NAME, sidecar)


# ---------------------------------------------------------------------------
# replicate_tree (platform-dispatch linking + atomic staging)
# ---------------------------------------------------------------------------


def replicate_tree(
    cache_entry_dir: Path,
    output_dir: Path,
) -> Tuple[int, str]:
    """Atomically replicate the cache entry tree to ``output_dir``.

    Strategy cascade:
      1. Try clonefile/reflink-style copy (platform-native, COW).
      2. Fall back to hardlink (same-fs, fastest; consumer-mutation
         blocked by cache-side chmod-readonly).
      3. Fall back to relative symlink on EXDEV (cross-filesystem).

    Returns (n_files_linked, link_type_str).

    Atomicity: stages to ``output_dir.tmp-<pid>/`` then renames. If the
    process dies before rename, no ``output_dir`` appears (gc sweeps
    orphan .tmp/ eventually).
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        # Pre-condition: caller ensured no collision (resolve_or_link check).
        # Remove to allow fresh replication.
        _rm_readonly_tree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f"{output_dir.name}{_TMP_SUFFIX}-{os.getpid()}"
    if staging_dir.exists():
        _rm_readonly_tree(staging_dir)

    link_type, n_linked = _replicate_tree_to_staging(cache_entry_dir, staging_dir)

    # Ensure the staging root is writable so ``os.rename`` succeeds on
    # APFS/Linux (clonefile + reflink preserve source's readonly mode bits).
    try:
        os.chmod(staging_dir, 0o755)
    except OSError:
        pass

    # Atomic finalize
    os.rename(staging_dir, output_dir)

    # For clonefile/reflink (COW, independent inodes), make the consumer
    # tree writable — consumer can legitimately add/modify files in their
    # output_dir without affecting the cache. Hardlinks + symlinks must
    # stay readonly-inherited so the cache-side chmod continues to protect.
    if link_type in ("clonefile", "reflink"):
        _chmod_writable_recursive(output_dir)

    return n_linked, link_type


def _chmod_writable_recursive(root: Path) -> None:
    """Consumer-side: COW copies get full write access (they own independent
    inodes). Only called when link_type implies inode independence.
    """
    try:
        os.chmod(root, 0o755)
    except OSError:
        pass
    for path in root.rglob("*"):
        try:
            if path.is_dir():
                os.chmod(path, 0o755)
            elif path.is_file():
                os.chmod(path, 0o644)
        except OSError:
            pass


def _replicate_tree_to_staging(
    cache_entry_dir: Path, staging_dir: Path,
) -> Tuple[str, int]:
    """Attempt clonefile → reflink → hardlink → symlink in order.

    The first strategy that succeeds on the cache root is used for the
    full tree (no mixing). Returns (link_type, n_files_linked).
    """
    # Strategy 1: clonefile (macOS APFS)
    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["cp", "-c", "-R", str(cache_entry_dir), str(staging_dir)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return "clonefile", _count_files(staging_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.debug("clonefile failed, falling through to hardlink: %s", exc)
            if staging_dir.exists():
                _rm_readonly_tree(staging_dir)

    # Strategy 2: reflink (Linux Btrfs/XFS)
    if sys.platform.startswith("linux"):
        try:
            subprocess.run(
                ["cp", "--reflink=always", "-R", str(cache_entry_dir), str(staging_dir)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return "reflink", _count_files(staging_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.debug("reflink failed, falling through to hardlink: %s", exc)
            if staging_dir.exists():
                _rm_readonly_tree(staging_dir)

    # Strategy 3: hardlink (same-fs; chmod-readonly inodes block mutation)
    try:
        n = _hardlink_tree(cache_entry_dir, staging_dir)
        return "hardlink_readonly", n
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        logger.info(
            "Cache %s on different filesystem from output_dir %s — "
            "falling back to relative symlink (supported).",
            cache_entry_dir,
            staging_dir,
        )
        if staging_dir.exists():
            _rm_readonly_tree(staging_dir)

    # Strategy 4: relative symlink (cross-fs fallback)
    n = _symlink_tree(cache_entry_dir, staging_dir)
    return "symlink_relative", n


def _hardlink_tree(src: Path, dst: Path) -> int:
    """Recursively hardlink files from src to dst. Dirs are recreated.
    Raises OSError(errno.EXDEV) on cross-fs — caller dispatches to symlink.
    """
    dst.mkdir(parents=True)
    n = 0
    for src_path in sorted(src.rglob("*")):
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        if src_path.is_dir():
            dst_path.mkdir(exist_ok=True)
        elif src_path.is_file():
            os.link(src_path, dst_path)
            n += 1
    return n


def _symlink_tree(src: Path, dst: Path) -> int:
    """Recursively symlink files (relative symlinks so symlink target is
    portable if the tree is later moved as a unit).
    """
    dst.mkdir(parents=True)
    n = 0
    for src_path in sorted(src.rglob("*")):
        rel = src_path.relative_to(src)
        dst_path = dst / rel
        if src_path.is_dir():
            dst_path.mkdir(exist_ok=True)
        elif src_path.is_file():
            target_abs = src_path.resolve()
            target_rel = os.path.relpath(target_abs, dst_path.parent)
            os.symlink(target_rel, dst_path)
            n += 1
    return n


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------


def gc_cache(
    cache_root: Path,
    *,
    older_than_days: Optional[int] = None,
    max_size_gb: Optional[float] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Evict cache entries under LRU-by-mtime + size-budget policy.

    Arguments combine with AND semantics (both filters must vote-delete).

    Returns the list of cache-entry paths that WERE evicted (or WOULD be
    under dry_run). Pinned entries (per ``_PINNED.json``) are never
    evicted.
    """
    if not cache_root.exists():
        return []

    pinned = _load_pinned(cache_root)

    # Collect all cache entries with (path, mtime, size_bytes)
    entries: List[Tuple[Path, float, int]] = []
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith(_TMP_SUFFIX) or "-" + _TMP_SUFFIX in child.name:
            # Orphan staging dir — sweep these separately; not a finalized entry.
            continue
        if child.name in pinned:
            continue
        # Validate cache-key naming (64 hex chars); skip non-cache dirs
        if len(child.name) != 64 or not all(c in "0123456789abcdef" for c in child.name):
            continue
        mtime = child.stat().st_mtime
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        entries.append((child, mtime, size))

    # Sort oldest-mtime first (LRU eviction order)
    entries.sort(key=lambda t: t[1])

    now = time.time()
    age_threshold = now - (older_than_days * 86400.0) if older_than_days is not None else None
    size_budget_bytes = int(max_size_gb * (1024 ** 3)) if max_size_gb is not None else None

    # Apply age filter: only entries older than threshold are candidates
    age_candidates = [e for e in entries if age_threshold is None or e[1] < age_threshold]

    # Apply size filter: evict oldest until total size ≤ budget
    if size_budget_bytes is not None:
        total_size = sum(e[2] for e in entries)
        to_evict: List[Path] = []
        if total_size > size_budget_bytes:
            shrunk = total_size
            for path, _mtime, size in age_candidates:
                if shrunk <= size_budget_bytes:
                    break
                to_evict.append(path)
                shrunk -= size
        evict_paths = to_evict
    else:
        # Age-only mode
        evict_paths = [e[0] for e in age_candidates]

    if not dry_run:
        for path in evict_paths:
            _rm_readonly_tree(path)

    return evict_paths


def _load_pinned(cache_root: Path) -> set:
    pin_file = cache_root / PINNED_FILE_NAME
    if not pin_file.exists():
        return set()
    try:
        data = json.loads(pin_file.read_text())
        return set(data.get("pinned_keys", []))
    except (OSError, json.JSONDecodeError):
        logger.warning("Malformed %s — treating as empty pin list.", pin_file)
        return set()


# ---------------------------------------------------------------------------
# Producer-side: gather the 9 cache-key inputs from environment
# ---------------------------------------------------------------------------


def prepare_cache_key_inputs(
    extractor_config_path: Path,
    *,
    extractor_dir: Path,
    reconstructor_dir: Path,
    hft_statistics_dir: Optional[Path] = None,
    contract_version: str,
    data_dir: Path,
) -> Optional[CacheKeyInputs]:
    """Gather all 9 cache-key inputs from the filesystem + git.

    Returns None (with a WARN log) if any input cannot be gathered — caller
    must fall back to non-cached extraction. Fail-closed design prevents
    unstable-env cache pollution.

    Arguments:
      extractor_config_path: Path to the extractor TOML config file
        (POST-sweep-override if this is a grid-point run).
      extractor_dir: feature-extractor-MBO-LOB/ for git + Cargo.lock + binary.
      reconstructor_dir: MBO-LOB-reconstructor/ for git.
      hft_statistics_dir: hft-statistics/ for git (patched via .cargo/config.toml).
        When None, attempt to locate via ``extractor_dir/.cargo/config.toml``.
      contract_version: hft_contracts.SCHEMA_VERSION.
      data_dir: pipeline_root/data/ — used to locate databento-ingest manifest.
    """
    import platform
    import sys

    # 1. extractor_config_resolved: load TOML as dict (captures post-sweep-
    # override content since sweeps write fully-resolved TOML files).
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover — py<3.11 fallback
        import tomli as tomllib  # type: ignore
    try:
        resolved_config = tomllib.loads(extractor_config_path.read_text())
    except (OSError, Exception) as exc:
        logger.warning("Cache disabled: cannot load extractor config %s: %s",
                       extractor_config_path, exc)
        return None

    # 2. extractor_git_sha
    extractor_git_sha = _git_rev_parse_head(extractor_dir)
    if not extractor_git_sha:
        logger.warning("Cache disabled: cannot read git SHA in %s", extractor_dir)
        return None

    # 3. extractor_cargo_lock_sha256 (captures transitive dep changes)
    cargo_lock = extractor_dir / "Cargo.lock"
    if not cargo_lock.exists():
        logger.warning("Cache disabled: Cargo.lock missing at %s", cargo_lock)
        return None
    cargo_lock_sha = hashlib.sha256(cargo_lock.read_bytes()).hexdigest()

    # 4. reconstructor_git_sha
    reconstructor_git_sha = _git_rev_parse_head(reconstructor_dir)
    if not reconstructor_git_sha:
        logger.warning("Cache disabled: cannot read git SHA in %s", reconstructor_dir)
        return None

    # 5. hft_statistics_git_sha (patched via .cargo/config.toml — SHIP-BLOCKER)
    hft_stats_sha = _resolve_hft_statistics_sha(
        hft_statistics_dir=hft_statistics_dir,
        extractor_dir=extractor_dir,
    )
    if not hft_stats_sha:
        logger.warning(
            "Cache disabled: cannot read hft-statistics git SHA "
            "(checked .cargo/config.toml patches + parent dir)"
        )
        return None

    # 6. raw_input_manifest_hash (databento-ingest manifest.json) —
    # SHIP-BLOCKER #6. Locate via extractor config's input_dir.
    raw_manifest_hash = _compute_raw_input_manifest_hash(
        resolved_config=resolved_config,
        data_dir=data_dir,
        extractor_dir=extractor_dir,
    )
    if not raw_manifest_hash:
        logger.warning(
            "Cache disabled: cannot locate/hash databento-ingest manifest "
            "(extractor config input_dir does not resolve to a data/raw "
            "directory with manifest.json)"
        )
        return None

    # 7. compiled_binary_sha256 — SHIP-BLOCKER (catches uncommitted
    # ``cargo update`` class where git SHA + Cargo.lock are stale but binary
    # was locally rebuilt).
    binary_path = extractor_dir / "target" / "release" / "export_dataset"
    if not binary_path.exists():
        logger.warning(
            "Cache disabled: compiled binary not found at %s — run "
            "`cargo build --release --bin export_dataset` first, or pass "
            "--no-cache-extraction to skip caching for this run.",
            binary_path,
        )
        return None
    binary_sha = hashlib.sha256(binary_path.read_bytes()).hexdigest()

    # 8. platform_target
    platform_target = f"{sys.platform}-{platform.machine()}"

    return CacheKeyInputs(
        extractor_config_resolved=resolved_config,
        extractor_git_sha=extractor_git_sha,
        extractor_cargo_lock_sha256=cargo_lock_sha,
        reconstructor_git_sha=reconstructor_git_sha,
        hft_statistics_git_sha=hft_stats_sha,
        raw_input_manifest_hash=raw_manifest_hash,
        compiled_binary_sha256=binary_sha,
        platform_target=platform_target,
        contract_version=contract_version,
    )


def _git_rev_parse_head(repo_dir: Path) -> Optional[str]:
    """Return ``git rev-parse HEAD`` in repo_dir, or None on failure."""
    if not (repo_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _resolve_hft_statistics_sha(
    hft_statistics_dir: Optional[Path],
    extractor_dir: Path,
) -> Optional[str]:
    """Locate hft-statistics repo + return its git HEAD.

    Resolution order:
      1. Explicit ``hft_statistics_dir`` argument (test / CLI override).
      2. Parse ``extractor_dir/.cargo/config.toml`` for a
         ``patch.crates-io.hft-statistics.path`` override, resolve relative
         to extractor_dir, rev-parse there.
      3. Fallback: sibling dir ``extractor_dir.parent / "hft-statistics"``.
    """
    if hft_statistics_dir is not None:
        return _git_rev_parse_head(hft_statistics_dir)

    cargo_config = extractor_dir / ".cargo" / "config.toml"
    if cargo_config.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        try:
            data = tomllib.loads(cargo_config.read_text())
        except Exception:  # pragma: no cover — corrupt toml
            data = {}
        # Navigate: [patch.crates-io.hft-statistics]\npath = "..."
        patch_section = data.get("patch", {}).get("crates-io", {})
        hft_stats = patch_section.get("hft-statistics", {})
        if isinstance(hft_stats, dict):
            patch_path = hft_stats.get("path")
            if patch_path:
                candidate = (extractor_dir / patch_path).resolve()
                sha = _git_rev_parse_head(candidate)
                if sha:
                    return sha

    # Fallback: sibling dir
    sibling = extractor_dir.parent / "hft-statistics"
    if sibling.exists():
        return _git_rev_parse_head(sibling)

    return None


def _compute_raw_input_manifest_hash(
    resolved_config: Dict[str, Any],
    data_dir: Path,
    extractor_dir: Path,
) -> Optional[str]:
    """Hash the databento-ingest manifest.json for the input data.

    Looks up ``[data].input_dir`` (relative to extractor_dir per TOML
    convention) and reads ``manifest.json`` there.
    """
    data_section = resolved_config.get("data", {})
    input_dir_rel = data_section.get("input_dir")
    if not input_dir_rel:
        return None
    # TOML paths are relative to the extractor_dir (that's where the TOML
    # is loaded from operationally). Fall back to pipeline data_dir if the
    # relative path escapes upward.
    candidate = (extractor_dir / input_dir_rel).resolve()
    manifest_path = candidate / "manifest.json"
    if not manifest_path.exists():
        # Try under pipeline data_dir directly (some configs use relative
        # paths that resolve there).
        alt = (data_dir / Path(input_dir_rel).name / "manifest.json")
        if alt.exists():
            manifest_path = alt
        else:
            return None
    try:
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError:
        return None
