"""Tests for ``hft_ops.ledger.register`` module.

Phase L Step 1 (2026-05-06): RecordFields frozen dataclass tests.

Subsequent Phase L steps (Step 2 register_from_fields, Step 3 build_record_fields)
extend this file with their own test classes.

Test contract per PHASE_L_DESIGN_REFINEMENTS_2026_05_06.md §5.1:

1. RecordFields construction: 20 mirror-fields + 1 extra all accept their
   declared types
2. Frozen invariant: post-construction mutation raises FrozenInstanceError
3. ``to_record_kwargs()`` produces a 20-key dict (excludes training_output_dir
   + 8 register-only / default-only fields)
4. ``to_record_kwargs()`` preserves Provenance instance type (does NOT flatten
   to dict via dataclasses.asdict)
5. ``ExperimentRecord(**fields.to_record_kwargs())`` round-trips cleanly
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
from pathlib import Path
from typing import Any
import pytest

from hft_contracts.experiment_record import ExperimentRecord
from hft_contracts.provenance import GitInfo, Provenance

from hft_ops.ledger.register import RecordFields


# Golden key set — the EXACT 20 ExperimentRecord-mirrored fields produced by
# ``RecordFields.to_record_kwargs()``. Defends against silent rename of any of
# the 20 keys (per Agent 2 ground-truth verification recommendation).
_EXPECTED_TO_RECORD_KWARGS_KEYS = frozenset({
    # Identity (4)
    "experiment_id", "name", "manifest_path", "fingerprint",
    # Provenance + content fingerprints (5)
    "feature_set_ref", "compatibility_fingerprint", "signal_export_output_dir",
    "provenance", "contract_version",
    # Configuration + metrics (2)
    "training_config", "training_metrics",
    # Stage-derived observations (2)
    "gate_reports", "cache_info",
    # Operator metadata (3)
    "tags", "hypothesis", "description",
    # Lifecycle metadata (4)
    "created_at", "duration_seconds", "status", "stages_completed",
})

# All 19 non-experiment_id required fields — used by the parametric
# CRITICAL-2 closure to verify each is individually required at construction
# (defends against silent default-addition regression).
_NON_ID_REQUIRED_FIELDS = [
    "name", "manifest_path", "fingerprint",
    "feature_set_ref", "compatibility_fingerprint",
    "signal_export_output_dir", "provenance", "contract_version",
    "training_config", "training_metrics",
    "gate_reports", "cache_info",
    "tags", "hypothesis", "description",
    "created_at", "duration_seconds", "status", "stages_completed",
]


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


def _build_minimal_provenance() -> Provenance:
    """Construct a minimal Provenance instance for tests.

    All 7 Provenance fields explicitly populated to lock the GitInfo /
    Provenance schema contract at test time (defends against silent schema
    drift in hft-contracts).
    """
    return Provenance(
        git=GitInfo(
            commit_hash="0" * 40,
            branch="main",
            dirty=False,
            short_hash="00000000",
        ),
        config_hashes={},
        data_dir_hash="a" * 64,
        contract_version="3.0",
        timestamp_utc="2026-05-06T00:00:00+00:00",
        retroactive=False,
        schema_version="1.0",
    )


def _build_minimal_fields(**overrides: Any) -> RecordFields:
    """Construct a minimal RecordFields instance for tests.

    All 20 mirror-fields explicitly populated. ``training_output_dir`` defaults
    to None unless override-set. Override any field via kwargs.
    """
    base = dict(
        # Identity (4)
        experiment_id="test_exp_20260506T000000_abcd1234",
        name="test_exp",
        manifest_path="/tmp/test_exp.yaml",
        fingerprint="a" * 64,
        # Provenance + content fingerprints (5)
        feature_set_ref=None,
        compatibility_fingerprint=None,
        signal_export_output_dir=None,
        provenance=_build_minimal_provenance(),
        contract_version="3.0",
        # Configuration + metrics (2)
        training_config={},
        training_metrics={},
        # Stage-derived observations (2)
        gate_reports={},
        cache_info={},
        # Operator metadata (3)
        tags=[],
        hypothesis="",
        description="",
        # Lifecycle metadata (4)
        created_at="2026-05-06T00:00:00+00:00",
        duration_seconds=0.0,
        status="completed",
        stages_completed=["training"],
    )
    base.update(overrides)
    return RecordFields(**base)


# -----------------------------------------------------------------------------
# TestRecordFieldsConstruction — basic construction + frozen invariant
# -----------------------------------------------------------------------------


class TestRecordFieldsConstruction:
    """Construction + frozen invariants per PHASE_L_DESIGN_REFINEMENTS §5.1."""

    def test_can_construct_with_all_required_fields(self) -> None:
        fields = _build_minimal_fields()
        assert fields.experiment_id == "test_exp_20260506T000000_abcd1234"
        assert fields.name == "test_exp"
        assert isinstance(fields.provenance, Provenance)
        assert fields.contract_version == "3.0"
        assert fields.fingerprint == "a" * 64

    def test_frozen_raises_on_mutation(self) -> None:
        """Phase L C-4 closure: frozen invariant prevents silent build-phase mutation."""
        fields = _build_minimal_fields()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fields.name = "mutated"  # type: ignore[misc]

    def test_frozen_raises_on_provenance_attr_mutation(self) -> None:
        """Frozen guarantees apply to dataclass-level fields, not nested-dataclass mutability.

        ``provenance.config_hashes["foo"] = "bar"`` would still work because the
        outer-dataclass frozen invariant doesn't transitively freeze nested
        non-frozen dataclasses. This test documents that the dataclass-level
        frozen invariant is what we rely on (callers MUST treat RecordFields
        as immutable as a contract).
        """
        fields = _build_minimal_fields()
        # Outer frozen: this raises
        with pytest.raises(dataclasses.FrozenInstanceError):
            fields.provenance = _build_minimal_provenance()  # type: ignore[misc]

    def test_training_output_dir_defaults_to_None(self) -> None:
        fields = _build_minimal_fields()
        assert fields.training_output_dir is None

    def test_training_output_dir_can_be_provided(self) -> None:
        fields = _build_minimal_fields(training_output_dir=Path("/tmp/training"))
        assert fields.training_output_dir == Path("/tmp/training")

    def test_construction_requires_all_non_default_fields(self) -> None:
        """Verify that omitting a required field raises TypeError at construction.

        This locks the contract that build_record_fields MUST explicitly set
        every field (no implicit fallbacks via defaults — per hft-rules §5
        fail-fast). Note: this only proves >=1 field is required; the
        ``test_each_field_individually_required`` parametric extension below
        proves EVERY non-default field is required (CRITICAL-2 closure).
        """
        with pytest.raises(TypeError, match="missing"):
            RecordFields(  # type: ignore[call-arg]
                experiment_id="x",
                # Missing all other required fields
            )

    @pytest.mark.parametrize("omitted_field", _NON_ID_REQUIRED_FIELDS)
    def test_each_field_individually_required(self, omitted_field: str) -> None:
        """CRITICAL-2 closure (Agent 1 adversarial review, 2026-05-06).

        Locks each of the 19 non-experiment_id required fields individually:
        omitting ANY single one raises TypeError at construction. Defends
        against silent default-addition regressions (e.g., a future PR that
        adds ``tags: List[str] = field(default_factory=list)`` would defeat
        the explicit-field-set contract; this parametric test fails immediately).
        """
        complete_kwargs: dict[str, Any] = dict(
            # Identity (4)
            experiment_id="test_exp_20260506T000000_abcd1234",
            name="test_exp",
            manifest_path="/tmp/test_exp.yaml",
            fingerprint="a" * 64,
            # Provenance + content fingerprints (5)
            feature_set_ref=None,
            compatibility_fingerprint=None,
            signal_export_output_dir=None,
            provenance=_build_minimal_provenance(),
            contract_version="3.0",
            # Configuration + metrics (2)
            training_config={},
            training_metrics={},
            # Stage-derived observations (2)
            gate_reports={},
            cache_info={},
            # Operator metadata (3)
            tags=[],
            hypothesis="",
            description="",
            # Lifecycle metadata (4)
            created_at="2026-05-06T00:00:00+00:00",
            duration_seconds=0.0,
            status="completed",
            stages_completed=["training"],
        )
        complete_kwargs.pop(omitted_field)
        with pytest.raises(TypeError, match="missing"):
            RecordFields(**complete_kwargs)

    def test_unhashable_due_to_dict_fields(self) -> None:
        """HIGH-2 closure (Agent 1 adversarial review, 2026-05-06).

        ``@dataclass(frozen=True)`` auto-generates ``__hash__`` based on all
        field values. RecordFields contains ``Dict[str, Any]`` and
        ``List[str]`` fields, which are unhashable. ``hash(fields)`` raises
        TypeError. This test documents the contract; if a future schema
        switch (FrozenDict / Tuple / etc.) makes RecordFields hashable, this
        test fails — at which point a hashability test should replace it.
        """
        fields = _build_minimal_fields()
        with pytest.raises(TypeError, match="unhashable"):
            hash(fields)

    def test_pickle_round_trip(self) -> None:
        """HIGH-3a closure (Agent 1 adversarial review, 2026-05-06).

        ``RecordFields`` is passed across process boundaries when
        ``cli_parallel_sweep.py`` workers register experiments (per
        PHASE_L_DESIGN_REFINEMENTS §5.4). Pickling support is implicit in
        stdlib dataclasses but the test suite must verify it — a future
        field type that's not picklable (e.g., closure, file handle, lazy
        Path resolver) would silently raise inside the worker pool, where
        ``concurrent.futures.ProcessPoolExecutor`` is notoriously bad at
        surfacing pickling errors.
        """
        fields = _build_minimal_fields(
            feature_set_ref={"name": "v1", "content_hash": "f" * 64},
            training_config={"lr": 0.001, "batch_size": 64},
        )
        restored = pickle.loads(pickle.dumps(fields))
        assert restored == fields
        assert isinstance(restored.provenance, Provenance)
        assert restored.feature_set_ref == {"name": "v1", "content_hash": "f" * 64}
        assert restored.training_config == {"lr": 0.001, "batch_size": 64}

    def test_deepcopy_round_trip(self) -> None:
        """HIGH-3b closure (Agent 1 adversarial review, 2026-05-06).

        ``copy.deepcopy(fields)`` produces an independent copy with the
        same field values. Verifies that deepcopy actually copies (no
        aliasing) — protects against future regression where a field type
        accidentally implements ``__deepcopy__`` to share state.
        """
        fields = _build_minimal_fields(
            training_config={"lr": 0.001},
            tags=["v3p0", "regression"],
        )
        restored = copy.deepcopy(fields)
        assert restored == fields
        # Verify deepcopy actually copies (no aliasing)
        assert restored.training_config is not fields.training_config
        assert restored.tags is not fields.tags

    def test_dataclasses_replace_round_trip(self) -> None:
        """Phase L Step 2 forward-compat (Agent 2 + Agent 6 closure).

        Step 2's ``register_from_fields`` will use ``dataclasses.replace``
        to set ``experiment_provenance_hash`` post-construction (forward-
        compat with potential Phase X.5 frozen Pydantic ExperimentRecord).
        Lock the pattern works on RecordFields itself.
        """
        fields = _build_minimal_fields()
        fields_v2 = dataclasses.replace(fields, name="renamed")
        assert fields_v2.name == "renamed"
        # Original unchanged (frozen invariant preserved)
        assert fields.name != "renamed"
        # Other fields preserved
        assert fields_v2.experiment_id == fields.experiment_id
        assert fields_v2.provenance is fields.provenance  # share by ref


# -----------------------------------------------------------------------------
# TestRecordFieldsToRecordKwargs — to_record_kwargs() output shape
# -----------------------------------------------------------------------------


class TestRecordFieldsToRecordKwargs:
    """``to_record_kwargs()`` produces ExperimentRecord-compatible dict."""

    def test_returns_dict_with_20_mirror_keys(self) -> None:
        """to_record_kwargs returns the 20 ExperimentRecord-mirrored fields."""
        fields = _build_minimal_fields()
        kwargs = fields.to_record_kwargs()
        assert len(kwargs) == 20

    def test_returns_exact_20_key_set_locked(self) -> None:
        """Golden-key-set lock (Agent 2 ground-truth verification recommendation).

        Asserts the EXACT 20 keys returned. Defends against silent rename of
        any key (e.g., ``provenance`` → ``prov``). If a future edit changes a
        key, this test fails AND the corresponding edit must be made in
        consumers (ExperimentRecord constructor kwargs).
        """
        fields = _build_minimal_fields()
        kwargs = fields.to_record_kwargs()
        assert frozenset(kwargs.keys()) == _EXPECTED_TO_RECORD_KWARGS_KEYS

    def test_excludes_training_output_dir(self) -> None:
        """Phase L design: training_output_dir is register-time-only, NOT serialized."""
        fields = _build_minimal_fields(training_output_dir=Path("/tmp/training"))
        kwargs = fields.to_record_kwargs()
        assert "training_output_dir" not in kwargs

    def test_excludes_register_only_fields(self) -> None:
        """experiment_provenance_hash + artifacts are set by register_from_fields, not by build phase."""
        fields = _build_minimal_fields()
        kwargs = fields.to_record_kwargs()
        assert "experiment_provenance_hash" not in kwargs
        assert "artifacts" not in kwargs

    def test_excludes_default_only_fields(self) -> None:
        """ExperimentRecord fields not populated by _record_experiment body
        default to their dataclass defaults; to_record_kwargs does not include them.

        This avoids accidentally overriding user-set fields if RecordFields is
        ever extended without updating to_record_kwargs.
        """
        fields = _build_minimal_fields()
        kwargs = fields.to_record_kwargs()
        # Default-only on ExperimentRecord (set post-construction by sweep loop OR
        # never set by _record_experiment body):
        for excluded in [
            "notes",
            "sweep_id",
            "axis_values",
            "record_type",
            "sub_records",
            "parent_experiment_id",
            "extraction_config",
            "backtest_params",
            "backtest_metrics",
            "dataset_health",
            "sweep_failure_info",
        ]:
            assert excluded not in kwargs, f"{excluded} should NOT be in to_record_kwargs output"

    def test_preserves_provenance_instance_type(self) -> None:
        """Provenance is NOT flattened to dict (would happen with dataclasses.asdict).

        ExperimentRecord constructor expects a Provenance instance. Manual
        extraction in to_record_kwargs preserves the type.
        """
        fields = _build_minimal_fields()
        kwargs = fields.to_record_kwargs()
        assert isinstance(kwargs["provenance"], Provenance)
        assert isinstance(kwargs["provenance"].git, GitInfo)

    def test_constructs_valid_experiment_record(self) -> None:
        """RecordFields → to_record_kwargs() → ExperimentRecord(**kwargs) round-trip.

        This is the critical contract: the build-phase RecordFields output must
        cleanly construct a valid ExperimentRecord without any post-hoc
        adjustment. Locks the kwarg-shape invariant.
        """
        fields = _build_minimal_fields(
            feature_set_ref={"name": "test_v1", "content_hash": "f" * 64},
            compatibility_fingerprint="b" * 64,
            training_config={"model_config_hash": "c" * 64},
            tags=["v3p0", "regression"],
        )
        record = ExperimentRecord(**fields.to_record_kwargs())
        # Identity
        assert record.experiment_id == fields.experiment_id
        assert record.name == fields.name
        assert record.fingerprint == fields.fingerprint
        # Provenance + content
        assert record.feature_set_ref == fields.feature_set_ref
        assert record.compatibility_fingerprint == fields.compatibility_fingerprint
        assert record.contract_version == fields.contract_version
        # Configuration
        assert record.training_config == fields.training_config
        # Lifecycle
        assert record.status == fields.status
        # Defaults preserved on register-only fields
        assert record.experiment_provenance_hash is None
        assert record.artifacts == []
        assert record.notes == ""

    def test_round_trip_preserves_provenance_instance(self) -> None:
        """ExperimentRecord(**kwargs) doesn't accidentally clone or mutate Provenance.

        Provenance is INTENTIONALLY shared by reference (immutable-by-
        convention; CRITICAL-1 deep-copy isolation in to_record_kwargs
        explicitly excludes provenance). This is the contract.
        """
        fields = _build_minimal_fields()
        record = ExperimentRecord(**fields.to_record_kwargs())
        assert record.provenance is fields.provenance  # same instance

    def test_to_record_kwargs_isolates_mutable_containers(self) -> None:
        """CRITICAL-1 closure (Agent 1 adversarial review, 2026-05-06).

        Mutable container fields (training_config, training_metrics,
        gate_reports, cache_info, feature_set_ref, tags, stages_completed)
        are deep-copied so post-extraction mutation does NOT alias back to
        the frozen RecordFields instance. This preserves the frozen
        invariant — a load-bearing contract for Phase L Step 2's
        ``register_from_fields`` which constructs ExperimentRecord and then
        mutates ``record.training_config`` / ``record.artifacts``.

        Without this isolation: build phase produces "frozen" RecordFields,
        register phase calls ``to_record_kwargs()`` → ``ExperimentRecord(**kwargs)``,
        any post-construction mutation of ``record.training_config["foo"]``
        retroactively mutates ``fields.training_config`` via Python reference
        aliasing — defeating the entire purpose of the frozen carrier.
        """
        fields = _build_minimal_fields(
            feature_set_ref={"name": "v1", "content_hash": "f" * 64},
            training_config={"existing_key": "existing_value"},
            training_metrics={"test_ic": 0.37},
            gate_reports={"validation": {"status": "pass"}},
            cache_info={"cache_hit": True},
            tags=["v3p0"],
            stages_completed=["training"],
        )
        kwargs = fields.to_record_kwargs()
        # Mutate every mutable container in the kwargs
        kwargs["training_config"]["NEW_KEY"] = "mutated"
        kwargs["training_metrics"]["NEW_METRIC"] = 0.99
        kwargs["gate_reports"]["NEW_GATE"] = {"status": "fail"}
        kwargs["cache_info"]["NEW_CACHE"] = "mutated"
        kwargs["feature_set_ref"]["NEW_FIELD"] = "mutated"
        kwargs["tags"].append("MUTATED_TAG")
        kwargs["stages_completed"].append("MUTATED_STAGE")
        # Frozen invariant preserved — fields unchanged
        assert "NEW_KEY" not in fields.training_config
        assert "NEW_METRIC" not in fields.training_metrics
        assert "NEW_GATE" not in fields.gate_reports
        assert "NEW_CACHE" not in fields.cache_info
        assert fields.feature_set_ref is not None  # type narrowing
        assert "NEW_FIELD" not in fields.feature_set_ref
        assert "MUTATED_TAG" not in fields.tags
        assert "MUTATED_STAGE" not in fields.stages_completed

    def test_excluded_fields_land_at_canonical_defaults(self) -> None:
        """HIGH-4 closure (Agent 1 adversarial review, 2026-05-06).

        ExperimentRecord has 33 declared fields. ``to_record_kwargs`` returns
        20. The 13 not-set fields land at their dataclass defaults. This
        test locks those defaults — defends against a future ExperimentRecord
        schema bump that accidentally non-defaults one of these fields,
        which would silently leak stale values from a previous record (the
        kind of cross-build leakage that contaminated R9-R14's compat_fps).
        """
        fields = _build_minimal_fields()
        record = ExperimentRecord(**fields.to_record_kwargs())
        # Register-only fields (set later in register_from_fields)
        assert record.experiment_provenance_hash is None
        assert record.artifacts == []
        # Default-only on ExperimentRecord (never set by build phase)
        assert record.notes == ""
        assert record.extraction_config == {}
        assert record.backtest_params == {}
        assert record.backtest_metrics == {}
        assert record.dataset_health == {}
        assert record.sweep_failure_info == {}
        assert record.sweep_id == ""
        assert record.axis_values == {}
        # record_type defaults to literal "training" — verify it equals
        # RecordType.TRAINING.value (would surface a future enum rename).
        from hft_contracts.experiment_record import RecordType
        assert record.record_type == RecordType.TRAINING.value
        assert record.sub_records == []
        assert record.parent_experiment_id == ""


# -----------------------------------------------------------------------------
# TestRecordFieldsFieldShapes — field type validation
# -----------------------------------------------------------------------------


class TestRecordFieldsFieldShapes:
    """Optional / dict / list shapes accept their declared types."""

    def test_optional_str_fields_accept_None(self) -> None:
        fields = _build_minimal_fields(
            compatibility_fingerprint=None,
            signal_export_output_dir=None,
        )
        assert fields.compatibility_fingerprint is None
        assert fields.signal_export_output_dir is None

    def test_optional_str_fields_accept_str(self) -> None:
        fields = _build_minimal_fields(
            compatibility_fingerprint="b" * 64,
            signal_export_output_dir="/tmp/signals/test",
        )
        assert fields.compatibility_fingerprint == "b" * 64
        assert fields.signal_export_output_dir == "/tmp/signals/test"

    def test_optional_dict_field_accepts_None(self) -> None:
        fields = _build_minimal_fields(feature_set_ref=None)
        assert fields.feature_set_ref is None

    def test_optional_dict_field_accepts_dict(self) -> None:
        fields = _build_minimal_fields(
            feature_set_ref={"name": "test_v1", "content_hash": "f" * 64},
        )
        assert fields.feature_set_ref == {"name": "test_v1", "content_hash": "f" * 64}

    def test_required_dict_fields_accept_empty(self) -> None:
        fields = _build_minimal_fields(
            training_config={},
            training_metrics={},
            gate_reports={},
            cache_info={},
        )
        assert fields.training_config == {}
        assert fields.training_metrics == {}
        assert fields.gate_reports == {}
        assert fields.cache_info == {}

    def test_required_dict_fields_accept_populated(self) -> None:
        fields = _build_minimal_fields(
            training_config={"model_config_hash": "c" * 64, "lr": 1e-3},
            training_metrics={"test_ic": 0.37, "test_da": 0.64},
        )
        assert fields.training_config == {"model_config_hash": "c" * 64, "lr": 1e-3}
        assert fields.training_metrics == {"test_ic": 0.37, "test_da": 0.64}

    def test_required_list_fields_accept_empty(self) -> None:
        fields = _build_minimal_fields(
            tags=[],
            stages_completed=[],
        )
        assert fields.tags == []
        assert fields.stages_completed == []

    def test_required_list_fields_accept_populated(self) -> None:
        fields = _build_minimal_fields(
            tags=["v3p0", "regression", "tlob"],
            stages_completed=["training", "signal_export"],
        )
        assert fields.tags == ["v3p0", "regression", "tlob"]
        assert fields.stages_completed == ["training", "signal_export"]
