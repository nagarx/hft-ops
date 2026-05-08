"""Unit tests for `hft_ops.manifest.slot_taxonomy`.

Phase α-1 / #PY-78 (2026-05-10) — locks the slot path-base classification
(SSoT for the ${...} substitution rebase logic in
`hft_ops.manifest.loader._maybe_rebase_path`).

When a NEW slot is added that holds a path, this test must be extended in
parametric coverage. Per hft-rules §0 reuse-first: do NOT re-implement
slot-classification logic in consumer code; always import + call
``detect_slot_path_base(...)`` from the SSoT.
"""

import pytest

from hft_ops.manifest.slot_taxonomy import PathBase, detect_slot_path_base


class TestPathBaseEnum:
    """Sanity checks on the PathBase enum surface (frozen contract)."""

    def test_three_values_exist(self):
        assert PathBase.PIPELINE_ROOT.value == "pipeline_root"
        assert PathBase.TRAINER_CWD.value == "trainer_cwd"
        assert PathBase.NONE.value == "none"

    def test_enum_membership(self):
        assert PathBase("pipeline_root") == PathBase.PIPELINE_ROOT
        assert PathBase("trainer_cwd") == PathBase.TRAINER_CWD
        assert PathBase("none") == PathBase.NONE


class TestDetectSlotPathBaseTrainerCwd:
    """Trainer-cwd-relative slots — consumed bare by trainer subprocess
    (cwd = $pipeline_root/lob-model-trainer/). When a pipeline-root-relative
    source slot substitutes INTO one of these, the resolver MUST rebase via
    paths.resolve() + os.path.relpath() to produce trainer-cwd-relative form.
    """

    @pytest.mark.parametrize(
        "key",
        [
            "stages.training.overrides.data.data_dir",
            "stages.training.trainer_config.data.data_dir",
            "stages.training.overrides.data.feature_sets_dir",
            "stages.training.trainer_config.data.feature_sets_dir",
        ],
    )
    def test_trainer_cwd_slots(self, key: str):
        assert detect_slot_path_base(key) == PathBase.TRAINER_CWD


class TestDetectSlotPathBasePipelineRoot:
    """Pipeline-root-relative slots — consumed via config.paths.resolve().
    No rebasing needed when source slot is also pipeline-root-relative.
    """

    @pytest.mark.parametrize(
        "key",
        [
            # Stage output_dir slots (orchestrator-consumed)
            "stages.extraction.output_dir",
            "stages.extraction.config",
            "stages.raw_analysis.output_dir",
            "stages.raw_analysis.data_dir",
            "stages.dataset_analysis.output_dir",
            "stages.dataset_analysis.data_dir",
            "stages.validation.output_dir",
            "stages.training.output_dir",
            "stages.training.script",
            "stages.post_training_gate.output_dir",
            "stages.signal_export.output_dir",
            "stages.signal_export.checkpoint",
            "stages.signal_export.script",
            "stages.signal_export.config",
            "stages.backtesting.data_dir",
            "stages.backtesting.signals_dir",
            "stages.backtesting.model_checkpoint",
            "stages.backtesting.script",
            "stages.backtesting.params_file",
        ],
    )
    def test_pipeline_root_slots(self, key: str):
        assert detect_slot_path_base(key) == PathBase.PIPELINE_ROOT


class TestDetectSlotPathBaseNone:
    """Non-path slots — neither pipeline-root nor trainer-cwd. No rebasing."""

    @pytest.mark.parametrize(
        "key",
        [
            "experiment.name",
            "experiment.description",
            "experiment.contract_version",
            "stages.training.trainer_config.train.epochs",
            "stages.training.trainer_config.train.seed",
            "stages.training.trainer_config.model.model_type",
            "stages.training.horizon_value",
            "stages.backtesting.params.spread_bps",
            "sweep.name",
            "sweep.strategy",
            "",  # empty key (top-level)
        ],
    )
    def test_non_path_slots(self, key: str):
        assert detect_slot_path_base(key) == PathBase.NONE

    def test_unknown_slot_returns_none(self):
        # Unmatched dotted-path keys default to NONE (presumed non-path slot).
        assert detect_slot_path_base("stages.fake.totally_made_up") == PathBase.NONE
        assert detect_slot_path_base("nonexistent.slot.path") == PathBase.NONE


class TestDetectSlotPathBaseRegression:
    """Regression tests for the parametric loop matching order (first match wins)."""

    def test_overrides_data_dir_does_not_match_pipeline_root_pattern(self):
        # Critical: ``stages.training.overrides.data.data_dir`` must NOT be
        # classified as PIPELINE_ROOT (which would make the rebase a no-op
        # and silently re-introduce the #PY-78 bug). The TRAINER_CWD pattern
        # must match FIRST.
        result = detect_slot_path_base("stages.training.overrides.data.data_dir")
        assert result == PathBase.TRAINER_CWD
        assert result != PathBase.PIPELINE_ROOT

    def test_anchored_match_does_not_overshoot(self):
        # Patterns are anchored with ^...$. So a key with extra suffix should
        # NOT match. (E.g., ``stages.extraction.output_dir.subkey`` is not the
        # output_dir slot itself.)
        assert detect_slot_path_base("stages.extraction.output_dir.x") == PathBase.NONE
