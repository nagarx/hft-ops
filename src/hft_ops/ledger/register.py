"""Phase L Lifecycle Registration SSoT.

Lifts ledger registration from ``cli.py:_record_experiment`` (273 LOC, lines
419-691) into a 2-way build + register split. Used by:

- Orchestrator path: ``cli.py`` serial run + ``cli_parallel_sweep.py`` parallel
- Canonical-script path: ``lob-model-trainer/scripts/export_signals.py`` (Phase L
  wire-in via ``HFT_REGISTER_LEDGER=1`` env var, default-OFF)

Architecture per ``PHASE_L_DESIGN_REFINEMENTS_2026_05_06.md`` (monorepo root).

Phase L scope (this module):

- **Step 1 (THIS commit, 2026-05-06)**: ``RecordFields`` frozen dataclass —
  carrier between build phase and register phase.
- Step 2 (deferred): ``register_from_fields(fields, ledger) -> str`` with
  filelock + subsecond ID + ``dataclasses.replace`` for forward-compat with
  potential future frozen ``ExperimentRecord``.
- Step 3 (deferred): ``build_record_fields(manifest=None, artifact_dir=None)``
  2-way builder with synthetic-manifest fingerprint + divergence-mode WARN.

Architectural rationale:

The original ``_record_experiment`` body conflates 3 concerns: (a) extraction
from manifest+results+stage outputs, (b) ExperimentRecord construction, (c)
provenance composition + ledger persistence. Phase L's split:

1. **build_record_fields** (Step 3): a → produces ``RecordFields``
2. **register_from_fields** (Step 2): b + c → consumes ``RecordFields``,
   produces persisted ``ExperimentRecord``

The ``RecordFields`` frozen invariant prevents the build-phase output from
being silently mutated between phases — a class of bug Phase L was
specifically designed to eliminate.

See PHASE_L_DESIGN_REFINEMENTS_2026_05_06.md §3.1 "RecordFields data carrier"
for the full design rationale + 6 CRITICAL design issues this addresses.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_contracts.provenance import Provenance


__all__ = ["RecordFields"]


@dataclass(frozen=True)
class RecordFields:
    """Frozen carrier between ``build_record_fields()`` and ``register_from_fields()``.

    Mirrors ``ExperimentRecord`` constructor kwargs (post-Phase-Y, post-Phase-Q.6.5)
    EXCEPT for two register-time-only fields:

    1. ``experiment_provenance_hash``: composed in ``register_from_fields`` AFTER
       ExperimentRecord construction. Phase Y composition order requires
       ``training_config["model_config_hash"]`` already injected at build time
       (this is enforced — the build phase MUST set ``training_config`` with the
       harvested ``model_config_hash`` already present in the dict). The composer
       reads via ``record.training_config.get("model_config_hash")``; missing key
       → composer returns None → graceful degradation (matches existing
       ``compute_experiment_provenance_hash`` semantics at
       ``hft-contracts/.../experiment_record.py:748``).

    2. ``artifacts``: mutated by ``ExperimentLedger.persist_post_stage_artifacts(...)``
       during ``register_from_fields``. The mutation must happen BEFORE
       ``ledger.register(record)`` for the persisted JSON to include routed
       artifacts (Phase Q.6.5 invariant — verified by code-explorer Agent A3
       §B Hazard 4).

    Plus one EXTRA not on ``ExperimentRecord``:

    - ``training_output_dir``: passed to
      ``ExperimentLedger.persist_post_stage_artifacts(...)`` at register time.
      NOT serialized to the ledger record. Defaults to ``None`` for canonical-
      script path (no training stage results) — register_from_fields skips the
      persist_post_stage_artifacts call when None.

    **Frozen invariant**: build phase produces a finalized ``RecordFields``;
    register phase MUST NOT mutate. Use ``dataclasses.replace(...)`` if a
    field needs updating between phases (e.g., post-validation enrichment).

    **Field count**: 20 ExperimentRecord-mirrored fields + 1 extra = 21 total.

    Per ``PHASE_L_DESIGN_REFINEMENTS_2026_05_06.md`` §5.1 "RecordFields data
    carrier" + code-explorer Agent A3 §F Step 1 design.

    Example
    -------
    >>> fields = RecordFields(
    ...     experiment_id="exp_20260506T123456_abcd1234",
    ...     name="exp",
    ...     manifest_path="/tmp/exp.yaml",
    ...     fingerprint="a" * 64,
    ...     feature_set_ref={"name": "v1", "content_hash": "f" * 64},
    ...     compatibility_fingerprint="b" * 64,
    ...     signal_export_output_dir="/tmp/signals/test",
    ...     provenance=provenance,
    ...     contract_version="3.0",
    ...     training_config={"model_config_hash": "c" * 64},
    ...     training_metrics={"test_ic": 0.37},
    ...     gate_reports={},
    ...     cache_info={},
    ...     tags=["v3p0", "regression"],
    ...     hypothesis="",
    ...     description="",
    ...     created_at="2026-05-06T12:34:56+00:00",
    ...     duration_seconds=300.0,
    ...     status="completed",
    ...     stages_completed=["training", "signal_export"],
    ... )
    >>> kwargs = fields.to_record_kwargs()
    >>> from hft_contracts.experiment_record import ExperimentRecord
    >>> record = ExperimentRecord(**kwargs)  # smooth round-trip
    """

    # === Identity (4) ===
    experiment_id: str
    name: str
    manifest_path: str
    fingerprint: str

    # === Provenance + content fingerprints (5) ===
    feature_set_ref: Optional[Dict[str, str]]
    compatibility_fingerprint: Optional[str]
    signal_export_output_dir: Optional[str]
    provenance: Provenance
    contract_version: str

    # === Configuration + metrics (2) ===
    # NOTE: training_config MUST already contain "model_config_hash" key at
    # build-time (Phase Y composition order). The build phase reads the
    # harvested model_config_hash from signal_metadata.json's root field
    # and injects it into this dict BEFORE constructing RecordFields.
    training_config: Dict[str, Any]
    training_metrics: Dict[str, Any]

    # === Stage-derived observations (2) ===
    # gate_reports + cache_info are OBSERVATIONS, never enter compute_fingerprint
    # (locked by hft-ops/.../ledger/dedup.py invariant). Phase L preserves this.
    gate_reports: Dict[str, Dict[str, Any]]
    cache_info: Dict[str, Any]

    # === Operator-friendly metadata (3) ===
    tags: List[str]
    hypothesis: str
    description: str

    # === Lifecycle metadata (4) ===
    created_at: str
    duration_seconds: float
    status: str
    stages_completed: List[str]

    # === Register-time-only EXTRA (not on ExperimentRecord) ===
    # Used by register_from_fields to call ledger.persist_post_stage_artifacts(...)
    # before ledger.register(record). NOT serialized to ledger JSON.
    # None for canonical-script path (no training-stage output_dir).
    training_output_dir: Optional[Path] = None

    def to_record_kwargs(self) -> Dict[str, Any]:
        """Extract ``ExperimentRecord`` constructor kwargs.

        **Manual extraction** (NOT ``dataclasses.asdict``) to:

        - Preserve ``Provenance`` instance type. ``dataclasses.asdict`` would
          recursively flatten Provenance → dict, but ExperimentRecord wants
          a Provenance instance.
        - Exclude register-time-only fields (``training_output_dir``).
        - Make the kwarg shape explicit + auditable (any future field add to
          RecordFields must be threaded through here, surfacing in code review).

        **Mutable-container isolation contract** (Phase L Step 1 post-validation
        hardening, 2026-05-06; closes Agent 1 CRITICAL-1):

        Mutable container fields (``training_config``, ``training_metrics``,
        ``gate_reports``, ``cache_info``, ``feature_set_ref``, ``tags``,
        ``stages_completed``) are deep-copied so that downstream mutations of
        the returned kwargs (e.g., ``record.training_config["foo"] = "bar"``
        post-construction) do NOT retroactively mutate the frozen
        ``RecordFields`` instance via Python's reference-aliasing.

        Without this isolation, the frozen invariant on ``RecordFields`` is
        defeated: the build phase produces a "frozen" RecordFields, the
        register phase constructs ``ExperimentRecord(**kwargs)``, and any
        post-construction mutation of ``record.training_config`` would alias
        back to ``fields.training_config`` — exactly the silent-mutation
        class of bug Phase L was designed to eliminate.

        ``provenance`` is INTENTIONALLY NOT deep-copied — it's shared by
        reference (immutable-by-convention; Provenance is a leaf dataclass
        with no nested mutable containers besides ``config_hashes`` which is
        treated as write-once). This preserves ``record.provenance is
        fields.provenance`` as a contract surface (locked by
        ``test_round_trip_preserves_provenance_instance``).

        Returns 20 keys mirroring ``ExperimentRecord`` constructor positional
        args. Excludes:

        - ``experiment_provenance_hash`` (set by ``register_from_fields`` post-
          construction via ``dataclasses.replace``)
        - ``artifacts`` (mutated by ``ledger.persist_post_stage_artifacts``)
        - ``notes``, ``sweep_id``, ``axis_values``, ``record_type``,
          ``sub_records``, ``parent_experiment_id`` (default values on
          ExperimentRecord; not part of standard build-phase output)
        - ``extraction_config``, ``backtest_params``, ``backtest_metrics``,
          ``dataset_health``, ``sweep_failure_info`` (not populated by
          ``_record_experiment`` body — defaults to empty dict on
          ExperimentRecord)

        Returns
        -------
        Dict[str, Any]
            Keyword args ready for ``ExperimentRecord(**kwargs)`` construction.
            Mutable container fields are deep-copied (frozen invariant
            preserved); ``provenance`` is shared by reference.

        See PHASE_L_DESIGN_REFINEMENTS_2026_05_06.md §5.1 for the rationale on
        which fields are register-time-only vs build-time-only.
        """
        return {
            # Identity (4) — str values are immutable; share by reference
            "experiment_id": self.experiment_id,
            "name": self.name,
            "manifest_path": self.manifest_path,
            "fingerprint": self.fingerprint,
            # Provenance + content fingerprints (5)
            # feature_set_ref: Optional[Dict] — deep-copy if non-None to isolate
            "feature_set_ref": (
                _copy.deepcopy(self.feature_set_ref)
                if self.feature_set_ref is not None
                else None
            ),
            "compatibility_fingerprint": self.compatibility_fingerprint,  # Optional[str] — immutable
            "signal_export_output_dir": self.signal_export_output_dir,  # Optional[str] — immutable
            # provenance: SHARE BY REFERENCE (immutable-by-convention; locked by
            # test_round_trip_preserves_provenance_instance)
            "provenance": self.provenance,
            "contract_version": self.contract_version,
            # Configuration + metrics (2) — Dict[str, Any], deep-copy to isolate
            "training_config": _copy.deepcopy(self.training_config),
            "training_metrics": _copy.deepcopy(self.training_metrics),
            # Stage-derived observations (2) — Dict[str, Dict[str, Any]] /
            # Dict[str, Any], deep-copy
            "gate_reports": _copy.deepcopy(self.gate_reports),
            "cache_info": _copy.deepcopy(self.cache_info),
            # Operator metadata (3) — tags is List[str], shallow-copy via list()
            # since str elements are immutable
            "tags": list(self.tags),
            "hypothesis": self.hypothesis,
            "description": self.description,
            # Lifecycle metadata (4) — created_at/status are str (immutable),
            # duration_seconds is float (immutable), stages_completed is List[str]
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "stages_completed": list(self.stages_completed),
        }
