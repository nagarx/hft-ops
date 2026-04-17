"""
FeatureSet schema + validation for the Phase 4 content-addressed registry.

A FeatureSet is an immutable-once-written JSON artifact pinning:
1. A feature selection (which feature indices to use),
2. The contract version those indices refer to,
3. The source feature width they were drawn from,
4. Provenance metadata (evaluator run, criteria, profile hash),
5. Applicability metadata (assets, horizons).

Identity is carried by ``content_hash`` — SHA-256 over the PRODUCT
fields only (indices + source_feature_count + contract_version). All
other fields are metadata: two producer runs with identical product
but different criteria hash identically.

Canonical on-disk form is JSON at ``contracts/feature_sets/<name>.json``,
written atomically via ``hft_ops.feature_sets.writer``. Readers must
call ``verify_integrity()`` before trusting the payload.

See:
- ``contracts/feature_sets/SCHEMA.md`` — human-readable schema reference.
- ``hft_ops.feature_sets.hashing`` — the canonical-form hash spec.
- Plan §4 + 6-agent validation (2026-04-15).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from hft_ops.feature_sets.hashing import compute_feature_set_hash


FEATURE_SET_SCHEMA_VERSION: str = "1.0"
"""Current FeatureSet JSON schema version.

Bumped on any change that alters required keys, field types, or the
hash scope. Additive fields that are optional for both readers and
writers do NOT require a version bump (they fall through ``from_dict``
via ``.get(...)`` defaults).
"""

_HEX_HASH_LEN: int = 64
"""SHA-256 produces 32 bytes → 64 lowercase hex characters."""


class FeatureSetValidationError(ValueError):
    """FeatureSet JSON failed schema validation."""


class FeatureSetIntegrityError(ValueError):
    """FeatureSet ``content_hash`` disagrees with recomputed hash.

    Raised when a file was edited without a matching hash update (either
    by hand or by a buggy writer). The correct remediation is to
    regenerate the file via ``hft-ops evaluate --save-feature-set <name>``
    or restore from git.
    """


@dataclass(frozen=True)
class FeatureSetRef:
    """Lightweight pointer to a FeatureSet instance.

    Used when one artifact (ExperimentRecord, signal_metadata.json)
    needs to reference a FeatureSet without inlining the full payload.
    Both ``name`` and ``content_hash`` are stored — name for human
    readability, hash for integrity confirmation.

    Attributes:
        name: The human-assigned identifier (e.g., ``"momentum_hft_v1"``).
            Matches the FeatureSet JSON filename: ``<name>.json``.
        content_hash: 64-char lowercase hex SHA-256 of the product fields.
    """

    name: str
    content_hash: str


@dataclass(frozen=True)
class FeatureSetAppliesTo:
    """Applicability metadata — which contexts a FeatureSet was built for.

    NOT included in the content hash (this is recipe metadata, not
    product metadata). Consumers MAY enforce at resolution time
    (e.g., refuse to use a FeatureSet with ``applies_to.assets=["NVDA"]``
    when training on MSFT), but the hash itself treats two FeatureSets
    with identical indices as identical regardless of this field.

    Attributes:
        assets: Tuple of ticker symbols this set was constructed for.
        horizons: Tuple of label horizons this set targets.
    """

    assets: tuple[str, ...]
    horizons: tuple[int, ...]


@dataclass(frozen=True)
class FeatureSetProducedBy:
    """Provenance metadata for the producer run.

    Not hashed — identical products with different producers share the
    same content_hash by design. Stored for traceability.

    Attributes:
        tool: Producer tool name (e.g., ``"hft-feature-evaluator"``).
        tool_version: Version string of the tool at production time.
        config_path: Path to the evaluator config YAML that drove the
            run (relative to pipeline root).
        config_hash: SHA-256 of the evaluator config (raw file bytes).
        source_profile_hash: Hash of the FeatureProfile snapshot that
            produced this selection (from ``EvaluationPipeline.last_profile_hash``).
        data_export: Path to the NPY export evaluated (relative to
            pipeline root).
        data_dir_hash: Manifest hash of the export directory (matches
            ``hft_ops.provenance.lineage.hash_directory_manifest``).
    """

    tool: str
    tool_version: str
    config_path: str
    config_hash: str
    source_profile_hash: str
    data_export: str
    data_dir_hash: str


@dataclass(frozen=True)
class FeatureSet:
    """Content-addressed feature selection artifact.

    Immutable-once-written JSON at ``contracts/feature_sets/<name>.json``.
    Constructed via ``FeatureSet.build`` (auto-computes hash) or via
    ``FeatureSet.from_dict`` (loads from parsed JSON + validates).
    Direct construction is allowed for tests but skips validation.

    The ``content_hash`` field is computed over PRODUCT fields only
    (``feature_indices``, ``source_feature_count``, ``contract_version``)
    per ``hft_ops.feature_sets.hashing.compute_feature_set_hash``.
    Metadata (description, notes, applies_to, produced_by, criteria,
    created_at, created_by) is stored for provenance but EXCLUDED from
    the hash.

    Attributes:
        schema_version: FeatureSet JSON schema version this record
            conforms to.
        name: Human-assigned identifier (e.g., ``"momentum_hft_v1"``).
            Used as the registry filename and the lookup key from
            trainer ``DataConfig.feature_set``.
        content_hash: 64-char lowercase hex SHA-256 of product fields.
        contract_version: Pipeline contract version at production time.
        source_feature_count: Source feature width (e.g., 98).
        applies_to: Applicability metadata (assets, horizons).
        feature_indices: Sorted-unique tuple of feature indices.
        feature_names: Parallel tuple of feature names (derived from
            indices + contract_version; stored for human readability,
            EXCLUDED from content_hash).
        produced_by: Producer provenance.
        criteria: SelectionCriteria as a dict (via ``dataclasses.asdict``).
        criteria_schema_version: Version string of the SelectionCriteria
            schema at production time.
        description: Free-text description (metadata, not hashed).
        notes: Free-text operator notes (metadata, not hashed).
        created_at: ISO 8601 UTC timestamp.
        created_by: User/agent identifier.
    """

    # Identity
    schema_version: str
    name: str
    content_hash: str

    # Contract + source
    contract_version: str
    source_feature_count: int

    # Applicability
    applies_to: FeatureSetAppliesTo

    # Product
    feature_indices: tuple[int, ...]
    feature_names: tuple[str, ...]

    # Provenance
    produced_by: FeatureSetProducedBy
    criteria: Mapping[str, Any]
    criteria_schema_version: str

    # Bookkeeping
    description: str
    notes: str
    created_at: str
    created_by: str

    @classmethod
    def build(
        cls,
        *,
        name: str,
        feature_indices: Iterable[int],
        feature_names: Iterable[str],
        source_feature_count: int,
        contract_version: str,
        applies_to: FeatureSetAppliesTo,
        produced_by: FeatureSetProducedBy,
        criteria: Mapping[str, Any],
        criteria_schema_version: str,
        description: str = "",
        notes: str = "",
        created_at: str = "",
        created_by: str = "",
    ) -> "FeatureSet":
        """Construct a FeatureSet with content_hash auto-computed.

        Arguments are keyword-only to guard against positional drift as
        the schema evolves.

        ``feature_indices`` is normalized to sorted-unique tuple before
        hashing. ``feature_names`` is stored as a tuple in the order
        provided (caller must ensure name[i] corresponds to indices[i]
        after sorting — typically the producer pairs them).

        ``created_at`` defaults to empty string; callers are expected to
        supply an ISO 8601 UTC timestamp (``datetime.now(timezone.utc).isoformat()``)
        at the producer boundary so the timestamp reflects the producer
        run, not the moment a specific line of code executed.
        """
        indices = tuple(sorted(set(int(i) for i in feature_indices)))
        content_hash = compute_feature_set_hash(
            feature_indices=indices,
            source_feature_count=source_feature_count,
            contract_version=contract_version,
        )
        return cls(
            schema_version=FEATURE_SET_SCHEMA_VERSION,
            name=name,
            content_hash=content_hash,
            contract_version=str(contract_version),
            source_feature_count=int(source_feature_count),
            applies_to=applies_to,
            feature_indices=indices,
            feature_names=tuple(feature_names),
            produced_by=produced_by,
            criteria=dict(criteria),
            criteria_schema_version=str(criteria_schema_version),
            description=description,
            notes=notes,
            created_at=created_at,
            created_by=created_by,
        )

    def to_dict(self) -> dict[str, Any]:
        """Produce the canonical dict representation for JSON serialization.

        Lists are emitted with explicit ``list(...)`` construction (not
        tuple) so the JSON output does not leak the tuple type through
        json.dumps's default handling. Order of keys is insertion order
        (stable across Python versions ≥ 3.7); writers MUST use
        ``sort_keys=True`` regardless.
        """
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "content_hash": self.content_hash,
            "contract_version": self.contract_version,
            "source_feature_count": self.source_feature_count,
            "applies_to": {
                "assets": list(self.applies_to.assets),
                "horizons": list(self.applies_to.horizons),
            },
            "feature_indices": list(self.feature_indices),
            "feature_names": list(self.feature_names),
            "produced_by": asdict(self.produced_by),
            "criteria": dict(self.criteria),
            "criteria_schema_version": self.criteria_schema_version,
            "description": self.description,
            "notes": self.notes,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, verify: bool = True) -> "FeatureSet":
        """Load a FeatureSet from a dict (e.g., parsed JSON).

        Runs ``validate_feature_set_dict`` first. When ``verify`` is
        True (default), also calls ``verify_integrity`` to recompute the
        content hash and reject tampered files.

        Callers needing to inspect a known-bad file (e.g., a diff tool
        showing tampered vs. expected) can pass ``verify=False`` to
        skip the integrity check.
        """
        validate_feature_set_dict(d)
        fs = cls(
            schema_version=d["schema_version"],
            name=d["name"],
            content_hash=d["content_hash"],
            contract_version=d["contract_version"],
            source_feature_count=int(d["source_feature_count"]),
            applies_to=FeatureSetAppliesTo(
                assets=tuple(d["applies_to"]["assets"]),
                horizons=tuple(int(h) for h in d["applies_to"]["horizons"]),
            ),
            feature_indices=tuple(int(i) for i in d["feature_indices"]),
            feature_names=tuple(d.get("feature_names", ())),
            produced_by=FeatureSetProducedBy(**d["produced_by"]),
            criteria=dict(d.get("criteria", {})),
            criteria_schema_version=str(d.get("criteria_schema_version", "1.0")),
            description=d.get("description", ""),
            notes=d.get("notes", ""),
            created_at=d["created_at"],
            created_by=d.get("created_by", ""),
        )
        if verify:
            fs.verify_integrity()
        return fs

    def verify_integrity(self) -> None:
        """Recompute content_hash and raise if it disagrees with stored value.

        Detects manual edits to ``feature_indices``, ``source_feature_count``,
        or ``contract_version`` without matching hash update. Metadata
        edits (description, notes, applies_to, produced_by, criteria)
        do NOT change the hash, so integrity checks will still pass
        after those — this is by design (product-only hash).

        Raises:
            FeatureSetIntegrityError: If stored and recomputed hashes
                disagree. The exception message names the file and
                suggests remediation.
        """
        expected = compute_feature_set_hash(
            feature_indices=self.feature_indices,
            source_feature_count=self.source_feature_count,
            contract_version=self.contract_version,
        )
        if expected != self.content_hash:
            raise FeatureSetIntegrityError(
                f"FeatureSet '{self.name}' integrity check failed. "
                f"Stored content_hash:    {self.content_hash}. "
                f"Recomputed from fields: {expected}. "
                f"This file was likely edited without a matching hash update. "
                f"Regenerate via `hft-ops evaluate --save-feature-set {self.name} ...` "
                f"or restore from git."
            )

    def ref(self) -> FeatureSetRef:
        """Return a lightweight reference (name + content_hash) to this set."""
        return FeatureSetRef(name=self.name, content_hash=self.content_hash)


def validate_feature_set_dict(d: Mapping[str, Any]) -> None:
    """Imperative schema validation for a FeatureSet dict.

    Matches the style of ``hft-contracts/src/hft_contracts/validation.py``
    (raise ``FeatureSetValidationError`` on hard violations, no warnings
    collected). Called at the boundary — loaders call this before
    constructing a FeatureSet.

    Checks (in order):
    1. All required top-level keys are present.
    2. ``schema_version`` matches the current package version.
    3. ``content_hash`` is a 64-char lowercase hex string.
    4. ``feature_indices`` is a non-empty list of unique non-negative
       ints, all < ``source_feature_count``.
    5. ``source_feature_count`` is a positive int.
    6. ``applies_to`` contains ``assets`` (list[str]) and ``horizons``
       (list[int]).
    7. ``produced_by`` is a dict (fine-grained validation done via
       ``FeatureSetProducedBy(**d["produced_by"])`` unpack).

    Raises:
        FeatureSetValidationError: With a descriptive message on any
            violation.
    """
    if not isinstance(d, Mapping):
        raise FeatureSetValidationError(
            f"FeatureSet payload must be a mapping, got {type(d).__name__}"
        )

    required_top = {
        "schema_version",
        "name",
        "content_hash",
        "contract_version",
        "source_feature_count",
        "applies_to",
        "feature_indices",
        "produced_by",
        "created_at",
    }
    missing = required_top - set(d.keys())
    if missing:
        raise FeatureSetValidationError(
            f"FeatureSet missing required keys: {sorted(missing)}. "
            f"Schema: contracts/feature_sets/SCHEMA.md"
        )

    schema_version = d["schema_version"]
    if schema_version != FEATURE_SET_SCHEMA_VERSION:
        raise FeatureSetValidationError(
            f"Unsupported FeatureSet schema_version: {schema_version!r} "
            f"(this package supports {FEATURE_SET_SCHEMA_VERSION!r}). "
            f"Upgrade the package or regenerate the FeatureSet."
        )

    content_hash = d["content_hash"]
    if (
        not isinstance(content_hash, str)
        or len(content_hash) != _HEX_HASH_LEN
        or not all(c in "0123456789abcdef" for c in content_hash)
    ):
        raise FeatureSetValidationError(
            f"content_hash must be {_HEX_HASH_LEN}-char lowercase hex "
            f"SHA-256 (no 'sha256:' prefix), got: {content_hash!r}"
        )

    indices = d["feature_indices"]
    if not isinstance(indices, list) or not indices:
        raise FeatureSetValidationError(
            "feature_indices must be a non-empty list"
        )
    if any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise FeatureSetValidationError(
            f"feature_indices must contain only ints (bool excluded), "
            f"got: {indices}"
        )
    if any(i < 0 for i in indices):
        raise FeatureSetValidationError(
            f"feature_indices must be non-negative, got min={min(indices)}"
        )
    if len(set(indices)) != len(indices):
        raise FeatureSetValidationError(
            "feature_indices must be unique (no duplicates)"
        )

    sfc = d["source_feature_count"]
    if not isinstance(sfc, int) or isinstance(sfc, bool) or sfc <= 0:
        raise FeatureSetValidationError(
            f"source_feature_count must be positive int, got: {sfc!r}"
        )
    if max(indices) >= sfc:
        raise FeatureSetValidationError(
            f"max(feature_indices)={max(indices)} must be < "
            f"source_feature_count={sfc}"
        )

    applies_to = d["applies_to"]
    if not isinstance(applies_to, Mapping):
        raise FeatureSetValidationError(
            f"applies_to must be a mapping, got {type(applies_to).__name__}"
        )
    for key, elem_type in (("assets", str), ("horizons", int)):
        if key not in applies_to:
            raise FeatureSetValidationError(
                f"applies_to missing required key: {key!r}"
            )
        seq = applies_to[key]
        if not isinstance(seq, list):
            raise FeatureSetValidationError(
                f"applies_to.{key} must be a list, got {type(seq).__name__}"
            )
        if any(not isinstance(v, elem_type) or isinstance(v, bool) for v in seq):
            raise FeatureSetValidationError(
                f"applies_to.{key} must contain only {elem_type.__name__}, "
                f"got: {seq}"
            )

    produced_by = d["produced_by"]
    if not isinstance(produced_by, Mapping):
        raise FeatureSetValidationError(
            f"produced_by must be a mapping, got {type(produced_by).__name__}"
        )
    required_produced_by = {
        "tool", "tool_version", "config_path", "config_hash",
        "source_profile_hash", "data_export", "data_dir_hash",
    }
    missing_pb = required_produced_by - set(produced_by.keys())
    if missing_pb:
        raise FeatureSetValidationError(
            f"produced_by missing required keys: {sorted(missing_pb)}"
        )
