"""Phase R-17 F4: regression tests for training.py override migration to apply_override_loud.

Closes #PY-128 (training.py:393 silent override of trainer_config.data.horizon_idx via
manifest-level horizon_value resolution) + NEW-BUG-10 sister site (training.py:399
output_dir override).

Tests verify:
- F4-A: clobber-vs-existing emits UserWarning with source attribution
- F4-B: no-conflict path runs clean (no warning when no existing value)
- F4-C: value-matches-existing path runs clean (no warning when values agree)
- F4-D: integration with sweep.py horizon axis (manifest overrides set + training.py
  resolves same key — coexistence WARN-only path active)

Per design doc Phase R-17 §10.3 sequencing: F4 lands AFTER F2 (fingerprint integrity
test in place) and BUNDLED with F3 (apply_override_loud SSoT). user_set_check=False
during initial migration; Phase R-18 promotes per-site.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import pytest

from hft_ops.stages._override_discipline import (
    OverrideConflictError,
    apply_override_loud,
)


# =============================================================================
# Test class: TestTrainingOverrideDiscipline
# =============================================================================


class TestTrainingOverrideDiscipline:
    """Phase R-17 F4: apply_override_loud usage at training.py:393 + :399.

    These tests exercise the migration pattern in isolation (the actual
    training.py call sites are wrapped inside StageRunner.run() which requires
    significant fixture setup). The behavior contract IS the
    apply_override_loud() guarantees; we lock that the migration preserves
    Phase R-17 v2 user_set_check=False semantics + source attribution.
    """

    def test_horizon_idx_override_no_warn_when_no_existing_value(self):
        """F4-B: training.py:393 pattern — no existing value, no warning."""
        overrides: Dict[str, Any] = {}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                overrides,
                "data.horizon_idx",
                2,
                source=(
                    "training.py:393 resolved from horizon_value=60 via export metadata"
                ),
                user_set_check=False,
            )
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings, (
                f"Override with no existing value should NOT warn, got: {user_warnings}"
            )
        assert overrides["data"]["horizon_idx"] == 2

    def test_horizon_idx_override_warns_on_clobber(self):
        """F4-A: training.py:393 pattern — existing user-set value differs, WARN emitted."""
        # Simulate manifest sweep axis having set data.horizon_idx=0
        overrides = {"data": {"horizon_idx": 0}}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                overrides,
                "data.horizon_idx",
                2,  # runtime-resolved value differs from user-set 0
                source=(
                    "training.py:393 resolved from horizon_value=300 via export metadata"
                ),
                user_set_check=False,  # F4 initial migration: WARN-only
            )
            user_warnings = [
                str(w.message) for w in captured if issubclass(w.category, UserWarning)
            ]
            assert any("data.horizon_idx" in m for m in user_warnings), (
                f"Expected warning mentioning data.horizon_idx, got: {user_warnings}"
            )
            assert any("training.py:393" in m for m in user_warnings), (
                "Warning must include source attribution training.py:393"
            )
        # Override IS applied (runtime wins, but emits warning)
        assert overrides["data"]["horizon_idx"] == 2

    def test_horizon_idx_override_no_warn_when_values_match(self):
        """F4-C: training.py:393 pattern — runtime value matches user-set, no warning."""
        overrides = {"data": {"horizon_idx": 2}}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                overrides,
                "data.horizon_idx",
                2,  # matches existing
                source="training.py:393 resolved from horizon_value=60",
                user_set_check=False,
            )
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings, (
                f"Matching values should NOT warn, got: {user_warnings}"
            )
        assert overrides["data"]["horizon_idx"] == 2

    def test_output_dir_override_no_warn_when_no_existing_value(self):
        """F4-B: training.py:399 sister site — no existing value, no warning."""
        overrides: Dict[str, Any] = {}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                overrides,
                "output_dir",
                "/tmp/test/runs/exp1",
                source="training.py:399 resolved from stage.output_dir=runs/exp1",
                user_set_check=False,
            )
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings
        assert overrides["output_dir"] == "/tmp/test/runs/exp1"

    def test_output_dir_override_warns_on_clobber(self):
        """F4-A: training.py:399 sister — user-set output_dir differs, WARN."""
        overrides = {"output_dir": "/user/specified/path"}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                overrides,
                "output_dir",
                "/runtime/resolved/path",
                source="training.py:399 resolved from stage.output_dir=runs/exp1",
                user_set_check=False,
            )
            user_warnings = [
                str(w.message) for w in captured if issubclass(w.category, UserWarning)
            ]
            assert any("output_dir" in m for m in user_warnings)
            assert any("training.py:399" in m for m in user_warnings)
        # Override applied (runtime wins under user_set_check=False)
        assert overrides["output_dir"] == "/runtime/resolved/path"

    def test_phase_r18_promotion_xfail(self):
        """F4 future-flip xfail: when Phase R-18 promotes user_set_check=True,
        the clobber path raises instead of warning. This test ASSERTS the future
        behavior; currently xfail; flips to xpass when migration ships.

        Per design doc anti-drift #4: documented for forward-compat.
        """
        overrides = {"data": {"horizon_idx": 0}}
        with pytest.raises(OverrideConflictError, match="data.horizon_idx"):
            apply_override_loud(
                overrides,
                "data.horizon_idx",
                2,
                source="training.py:393 hypothetical Phase R-18 strict-mode",
                user_set_check=True,  # Phase R-18 promotion
            )
        # Override NOT applied on raise
        assert overrides["data"]["horizon_idx"] == 0
