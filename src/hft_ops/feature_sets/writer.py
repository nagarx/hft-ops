"""
Atomic FeatureSet writer.

Phase 4 introduces the first mutable registry inside ``contracts/`` —
``contracts/feature_sets/<name>.json``. Producer writes must be crash-
safe (no partial files on SIGKILL or disk-full) and collision-aware
(refuse to silently clobber a prior write under the same name).

Protocol (locked by ``tests/test_feature_set_writer.py``):

1. Resolve the target path: ``<feature_sets_dir>/<name>.json``.
2. If the path exists and ``force=False``:
   - Read existing content_hash.
   - If existing hash == new hash → idempotent no-op, return the
     existing path (do NOT rewrite — this avoids churning created_at
     + file mtime on re-runs with identical inputs).
   - If existing hash != new hash → raise ``FeatureSetExists`` (user
     must either bump the name suffix or pass ``--force``).
3. If the path exists and ``force=True``:
   - Overwrite (same atomic protocol).
4. Serialize the FeatureSet via ``to_dict`` → ``json.dumps`` with
   ``sort_keys=True, indent=2, default=str`` + trailing newline.
   Matches golden-fixture convention at
   ``lob-model-trainer/tests/fixtures/golden/generate_snapshots.py:100``.
5. Write to ``<target>.tmp.<pid>.<ns_time>`` → ``f.flush()`` →
   ``os.fsync(fd)`` → ``os.replace(tmp, target)``. ``os.replace`` is
   atomic on POSIX (same-filesystem rename is a single syscall) and
   Windows (``MoveFileEx`` with ``MOVEFILE_REPLACE_EXISTING``).

The writer integrity-verifies the FeatureSet BEFORE writing: callers
that construct a FeatureSet with a mismatched hash cannot poison the
registry. Verification uses ``FeatureSet.verify_integrity`` which
recomputes the hash from product fields.

This is the FIRST atomic-write helper in the monorepo. If a second
mutable-registry emerges (Phase 5+), extract to ``hft_utils``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from hft_ops.feature_sets.schema import (
    FeatureSet,
    FeatureSetValidationError,
    validate_feature_set_dict,
)


class FeatureSetExists(FileExistsError):
    """Raised when writing a FeatureSet whose target file already exists
    with DIFFERENT content (hash mismatch) and ``force=False``.

    Idempotent re-writes (same content hash) do NOT raise — they silently
    return the existing path.
    """


class AtomicWriteError(OSError):
    """An atomic write failed after tmp creation but before rename.

    The caller should inspect ``os.listdir(target.parent)`` for orphan
    ``.tmp.*`` files and delete them if the write is known to have
    failed permanently.
    """


def atomic_write_json(
    path: Path,
    obj: Mapping[str, Any],
    *,
    indent: int = 2,
) -> None:
    """Write a JSON-serializable object to ``path`` atomically.

    Crash-safe (SIGKILL between write and rename leaves at most a stale
    ``<path>.tmp.<pid>.<ns>`` file, never a partial ``<path>``).

    Serialization convention matches the monorepo (``sort_keys=True,
    default=str``) + trailing newline. Callers requiring NaN/Inf
    sanitization should pre-process the dict before calling this.

    Args:
        path: Target file path (will be created or replaced).
        obj: Mapping to serialize.
        indent: JSON indent spaces. Default 2 matches the golden-fixture
            convention.

    Raises:
        OSError: On disk errors (file system full, permission denied).
        AtomicWriteError: If rename fails after tmp creation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a pid+time_ns suffix so concurrent writers to the same name
    # do not collide on the tmp file. os.replace semantics then mean
    # whichever tmp gets renamed first wins (LIFO semantics — OK for
    # FeatureSet producers which should not actually run concurrently).
    tmp_path = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, indent=indent, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        # Best-effort cleanup; failure here is non-fatal (orphan tmp is
        # worse than a double-delete error).
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise AtomicWriteError(
            f"Atomic write failed for {path}: {exc}"
        ) from exc


def write_feature_set(
    path: Path,
    feature_set: FeatureSet,
    *,
    force: bool = False,
) -> Path:
    """Write a FeatureSet to the registry at ``path`` with protection.

    Ordering of guards:
    1. Integrity verification on the FeatureSet (defensive — callers
       shouldn't pass a tampered instance, but we refuse to persist one).
    2. Same-content idempotency: if a file already exists at ``path``
       with the same ``content_hash``, skip the rewrite and return
       ``path`` unchanged. This avoids churning ``created_at`` +
       filesystem mtime on re-runs that produce identical output.
    3. Different-content refuse-overwrite: if the existing file has a
       different content_hash, raise ``FeatureSetExists`` unless
       ``force=True``.
    4. Atomic write via ``atomic_write_json``.

    Args:
        path: Destination path (e.g.,
            ``<pipeline_root>/contracts/feature_sets/momentum_hft_v1.json``).
        feature_set: The FeatureSet to persist. Must pass
            ``verify_integrity`` (raised here if not).
        force: Overwrite behavior. When False (default), refuse to
            overwrite a different-content file at the same path; when
            True, allow overwrite.

    Returns:
        The resolved target path. Same as ``path`` unless the path was
        resolved (normalized); always the file that now contains the
        serialized FeatureSet.

    Raises:
        FeatureSetIntegrityError: If ``feature_set.verify_integrity()``
            fails (hash mismatch — the caller constructed a tampered
            instance).
        FeatureSetExists: If a different-content file already exists
            and ``force=False``.
        AtomicWriteError: On rename failure after tmp file creation.
    """
    feature_set.verify_integrity()

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_hash = existing.get("content_hash")
        except (OSError, json.JSONDecodeError):
            # Corrupt existing file — treat as different-content. If
            # force=True, we'll overwrite; if False, we refuse so the
            # operator can investigate.
            existing_hash = None

        if existing_hash == feature_set.content_hash:
            # Idempotent no-op — the on-disk file is already what we
            # would write. Skip the rewrite to preserve mtime + avoid
            # churning created_at.
            return path

        if not force:
            raise FeatureSetExists(
                f"FeatureSet '{feature_set.name}' already exists at {path} "
                f"with a DIFFERENT content_hash "
                f"(existing={existing_hash}, new={feature_set.content_hash}). "
                f"Either bump the name suffix (e.g., '{feature_set.name}' → "
                f"'{_bump_version_suffix(feature_set.name)}') to create a "
                f"new version, or pass force=True to overwrite."
            )

    # Pre-flight validate the serialized dict form. Catches writer-side
    # drift between dataclass construction and the schema validator.
    dict_form = feature_set.to_dict()
    try:
        validate_feature_set_dict(dict_form)
    except FeatureSetValidationError:
        raise  # Re-raise with original message; context already named.

    atomic_write_json(path, dict_form)
    return path


def _bump_version_suffix(name: str) -> str:
    """Suggest the next version number for a ``<base>_v<N>`` name.

    Used only inside error messages to give operators a concrete next
    step. Not load-bearing for correctness — failure to parse returns
    ``"<name>_next"`` rather than raising.
    """
    import re

    match = re.match(r"^(.+)_v(\d+)$", name)
    if not match:
        return f"{name}_next"
    base, n = match.group(1), int(match.group(2))
    return f"{base}_v{n + 1}"
