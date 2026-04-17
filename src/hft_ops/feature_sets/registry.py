"""
FeatureSet registry: read-side queries over ``contracts/feature_sets/``.

Mirrors the read API of ``hft_ops.ledger.ledger.ExperimentLedger``
(``list``, ``get``, ``find_*``) but is simpler because the registry is
an append-only directory of immutable JSON files (no index file, no
fingerprint-based dedup — each file IS the record).

Typical use sites:
- CLI: ``hft-ops feature-sets list`` → ``FeatureSetRegistry.list_refs()``
- CLI: ``hft-ops feature-sets show <name>`` → ``FeatureSetRegistry.get(name)``
- Trainer resolver (Batch 4c): ``FeatureSetRegistry.get(name)`` →
  ``feature_indices`` → ``DataConfig._feature_indices_resolved``
- hft-ops fingerprint (Batch 4c): ``FeatureSetRegistry.get(name).feature_indices``
  fed into ``compute_fingerprint`` via the dedup.py resolver hook.
"""

from __future__ import annotations

import json
from pathlib import Path

from hft_ops.feature_sets.schema import (
    FeatureSet,
    FeatureSetRef,
    FeatureSetValidationError,
)


class FeatureSetNotFound(FileNotFoundError):
    """Raised when a FeatureSet name is not present in the registry."""


class FeatureSetRegistry:
    """Read-side queries over a ``contracts/feature_sets/`` directory.

    The registry is stateless — every call reads from disk. This is
    appropriate for a small registry (dozens-to-hundreds of JSON files
    at KB-size each); revisit with caching when the registry grows to
    ~500+ entries OR the read-rate justifies it.

    Attributes:
        root: The directory scanned for ``<name>.json`` files. Must
            exist at registry-construction time unless ``allow_missing=True``
            is passed (useful for first-run registries that are about
            to be created by the first writer).
    """

    def __init__(self, root: Path, *, allow_missing: bool = False) -> None:
        self.root = Path(root)
        if not self.root.exists():
            if not allow_missing:
                raise FileNotFoundError(
                    f"FeatureSet registry root does not exist: {self.root}. "
                    f"Pass allow_missing=True to defer to first-writer "
                    f"creation, or create the directory manually."
                )

    # -- Enumeration ---------------------------------------------------

    def list_refs(self) -> list[FeatureSetRef]:
        """Return lightweight references for every FeatureSet in the registry.

        Does NOT validate file schemas or integrity — this is a catalog
        listing, not a verification pass. For integrity-audit use cases
        call ``verify_all()`` or iterate ``get(name)`` per entry.

        Returns:
            Sorted list of ``FeatureSetRef(name, content_hash)``. Sort
            is by name for stable output; callers that need different
            orderings can sort the result themselves.
        """
        refs: list[FeatureSetRef] = []
        if not self.root.exists():
            return refs
        for json_path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # Skip unreadable/malformed files in listing mode. The
                # operator will see the issue when they try to `get`
                # that specific name.
                continue
            name = raw.get("name")
            content_hash = raw.get("content_hash")
            if not isinstance(name, str) or not isinstance(content_hash, str):
                continue
            refs.append(FeatureSetRef(name=name, content_hash=content_hash))
        return refs

    def names(self) -> list[str]:
        """Return the sorted list of FeatureSet names currently in the registry."""
        return [ref.name for ref in self.list_refs()]

    def exists(self, name: str) -> bool:
        """Check whether a FeatureSet named ``name`` exists without loading it."""
        return self._path_for(name).exists()

    # -- Retrieval -----------------------------------------------------

    def get(self, name: str, *, verify: bool = True) -> FeatureSet:
        """Load and validate a FeatureSet by name.

        Args:
            name: The FeatureSet identifier (the ``<name>.json`` basename).
            verify: When True (default), also call ``verify_integrity``
                to detect tampered files. Set False only for inspection
                workflows (e.g., a diff tool showing an edited file).

        Returns:
            The loaded FeatureSet.

        Raises:
            FeatureSetNotFound: If ``<name>.json`` is absent.
            FeatureSetValidationError: If the file is present but fails
                schema validation.
            FeatureSetIntegrityError: If ``verify=True`` and the stored
                content_hash disagrees with the recomputed hash.
            OSError / json.JSONDecodeError: On I/O or parse failures.
        """
        path = self._path_for(name)
        if not path.exists():
            raise FeatureSetNotFound(
                f"FeatureSet '{name}' not found in registry at {self.root}. "
                f"Available: {self.names()[:10]}{'...' if len(self.names()) > 10 else ''}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FeatureSetValidationError(
                f"FeatureSet '{name}' at {path} is not valid JSON: {exc}"
            ) from exc
        return FeatureSet.from_dict(raw, verify=verify)

    def path_for(self, name: str) -> Path:
        """Return the resolved on-disk path for a given FeatureSet name.

        Does not check existence; callers use ``exists(name)`` for that.
        Useful to name the target for a fresh ``write_feature_set`` call.
        """
        return self._path_for(name)

    def _path_for(self, name: str) -> Path:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(
                f"FeatureSet name must not contain path separators or "
                f"start with '.', got: {name!r}"
            )
        return self.root / f"{name}.json"
