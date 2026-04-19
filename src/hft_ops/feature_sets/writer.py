"""
Atomic FeatureSet writer.

Phase 4 introduces the first mutable registry inside ``contracts/`` —
``contracts/feature_sets/<name>.json``. Producer writes must be crash-
safe (no partial files on SIGKILL or disk-full) and collision-aware
(refuse to silently clobber a prior write under the same name).

Phase 7 Stage 7.4 Round 5 (2026-04-20): ``atomic_write_json`` +
``AtomicWriteError`` moved to the canonical
``hft_contracts._atomic_io`` module (unified with
``ExperimentRecord.save`` and ``hft_ops.ledger.ledger._save_index``).
This module re-exports both names for back-compat with pre-Round-5
importers (Phase 4 consumers used ``from hft_ops.feature_sets.writer
import atomic_write_json, AtomicWriteError``).

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
4. Serialize the FeatureSet via ``to_dict`` → ``atomic_write_json``
   with canonical convention (``sort_keys=True`` + trailing newline;
   matches golden-fixture convention at
   ``lob-model-trainer/tests/fixtures/golden/generate_snapshots.py``).

The writer integrity-verifies the FeatureSet BEFORE writing: callers
that construct a FeatureSet with a mismatched hash cannot poison the
registry. Verification uses ``FeatureSet.verify_integrity`` which
recomputes the hash from product fields.
"""

from __future__ import annotations

import json
from pathlib import Path

# Phase 7 Stage 7.4 Round 5: re-export canonical atomic-write primitives
# from hft_contracts for back-compat. Older code imported these directly
# from this module; new code should import from hft_contracts._atomic_io.
from hft_contracts._atomic_io import AtomicWriteError, atomic_write_json

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
