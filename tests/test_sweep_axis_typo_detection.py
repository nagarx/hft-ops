"""Phase R-17 F5: tests for #PY-131 / NEW-BUG-11/16 typo detection.

Tests `validate_trainer_override_prefixes` + `_apply_overrides_to_dict(validate_prefixes=True)`
+ production manifest pre-audit lock.

Per Step 0 audit (2026-05-11): 51 manifests scanned, ZERO typos in current state.
This test suite locks the invariant: NO production manifest contains an override
with unknown top-level prefix.

References:
- Phase R-17 v2 design: POST_R16A_DESIGN_PHASE_2026_05_11.md §10.1 (F5)
- PHASE_P_BACKLOG.md #PY-131 + #PY-137 (sweep producer sister)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hft_ops.stages._override_discipline import (
    KNOWN_TRAINER_PREFIXES,
    UnknownOverrideKeyError,
    validate_trainer_override_prefixes,
)
from hft_ops.stages.training import _apply_overrides_to_dict


PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Test class: TestKnownTrainerPrefixes
# =============================================================================


class TestKnownTrainerPrefixesFrozenBaseline:
    """Phase R-17 F5: lock the 4 trainer prefixes at 2026-05-11 baseline.

    Adding a new trainer config top-level section REQUIRES updating both this
    test AND the frozenset in `_override_discipline.py`.
    """

    EXPECTED_PREFIXES = frozenset({"data", "model", "train", "cv"})

    def test_baseline_count_is_4(self):
        assert len(KNOWN_TRAINER_PREFIXES) == 4, (
            f"Baseline trainer prefix count drift: {sorted(KNOWN_TRAINER_PREFIXES)}. "
            f"If you intentionally added a new prefix, update this test."
        )

    def test_baseline_membership_locked(self):
        assert KNOWN_TRAINER_PREFIXES == self.EXPECTED_PREFIXES, (
            f"Trainer prefix membership drift: got {sorted(KNOWN_TRAINER_PREFIXES)}, "
            f"expected {sorted(self.EXPECTED_PREFIXES)}"
        )


# =============================================================================
# Test class: TestValidateTrainerOverridePrefixes
# =============================================================================


class TestValidateTrainerOverridePrefixes:
    """F5: typo detection via first-segment validation."""

    def test_all_known_prefixes_pass(self):
        """data.X, model.X, train.X, cv.X all pass validation."""
        overrides = {
            "data.data_dir": "/path",
            "data.labels.return_type": "point_return",
            "model.model_type": "tlob",
            "model.params": {},
            "train.seed": 42,
            "cv.n_splits": 5,
        }
        # No raise
        validate_trainer_override_prefixes(overrides, source="test")

    def test_typo_axis_seedd_at_top_level_raises(self):
        """F5a: typo'd top-level prefix raises UnknownOverrideKeyError."""
        overrides = {"trian.seed": 42}  # typo: trian vs train
        with pytest.raises(UnknownOverrideKeyError, match="trian"):
            validate_trainer_override_prefixes(overrides, source="test")

    def test_typo_mode_dot_dropout_raises(self):
        """F5b: typo `mode.dropout` (vs model.dropout) raises."""
        overrides = {"mode.dropout": 0.5}
        with pytest.raises(UnknownOverrideKeyError, match="mode"):
            validate_trainer_override_prefixes(overrides, source="test")

    def test_close_match_in_error_message(self):
        """F5d: error message includes difflib close-match suggestion."""
        overrides = {"trian.seed": 42}
        try:
            validate_trainer_override_prefixes(overrides, source="test")
        except UnknownOverrideKeyError as e:
            assert "train" in str(e), (
                f"Error must suggest 'train' as close match for 'trian', got: {e}"
            )

    def test_bare_key_not_validated(self):
        """Bare (no-dot) keys are NOT validated (handled by sweep.py's separate path)."""
        overrides = {"horizon_value": 60, "some_bare_key": "value"}
        # No raise — bare keys pass through
        validate_trainer_override_prefixes(overrides, source="test")

    def test_mixed_bare_and_dotted_only_validates_dotted(self):
        """Mixed: dotted keys validated, bare ones passed through."""
        overrides = {
            "horizon_value": 60,  # bare, skipped
            "data.feature_count": 98,  # dotted, validated (pass)
        }
        # No raise
        validate_trainer_override_prefixes(overrides, source="test")

    def test_dotted_key_with_unknown_prefix_AND_known_prefix(self):
        """When BOTH valid and invalid prefixes present, raises on the invalid one."""
        overrides = {
            "data.feature_count": 98,  # valid
            "mode.dropout": 0.5,  # typo
        }
        with pytest.raises(UnknownOverrideKeyError, match="mode"):
            validate_trainer_override_prefixes(overrides, source="test")


# =============================================================================
# Test class: TestApplyOverridesToDict — validate_prefixes parameter
# =============================================================================


class TestApplyOverridesToDictValidatesPrefixes:
    """F5: _apply_overrides_to_dict optional prefix validation."""

    def test_legacy_default_no_validation(self):
        """Pre-F5 callers (default validate_prefixes=False) accept ANY prefix."""
        cfg: Dict[str, Any] = {}
        # Even bogus prefix passes when validation off
        _apply_overrides_to_dict(cfg, {"bogus.key": "value"})
        assert cfg == {"bogus": {"key": "value"}}

    def test_strict_known_prefix_passes(self):
        """validate_prefixes=True with known prefix — pass."""
        cfg: Dict[str, Any] = {}
        _apply_overrides_to_dict(
            cfg, {"data.horizon_idx": 2},
            validate_prefixes=True,
            source="test",
        )
        assert cfg["data"]["horizon_idx"] == 2

    def test_strict_unknown_prefix_raises(self):
        """validate_prefixes=True with unknown prefix raises BEFORE mutation."""
        cfg: Dict[str, Any] = {}
        with pytest.raises(UnknownOverrideKeyError, match="mode"):
            _apply_overrides_to_dict(
                cfg, {"mode.dropout": 0.5},
                validate_prefixes=True,
                source="test",
            )
        # Cfg NOT mutated (fail-fast before mutation)
        assert cfg == {}, (
            "Partial mutation occurred despite validation failure — fail-fast violated"
        )


# =============================================================================
# Test class: TestProductionManifestAuditLock
# =============================================================================


class TestProductionManifestAuditLock:
    """F5 pre-audit lock: NO production manifest contains override with unknown prefix.

    Per Step 0 audit (2026-05-11): 51 manifests scanned, 13 dotted keys + 11 sweep
    axes — ZERO typos. This test locks that invariant going forward; any new
    manifest with a typo'd prefix fails immediately.
    """

    def test_all_production_manifests_have_valid_override_prefixes(self):
        """Glob hft-ops/experiments/**/*.yaml; collect all dotted override keys;
        assert each has first-segment in KNOWN_TRAINER_PREFIXES ∪ {known stage-level prefixes}.
        """
        manifests_dir = PIPELINE_ROOT / "hft-ops" / "experiments"
        if not manifests_dir.exists():
            pytest.skip(f"Experiments dir not found: {manifests_dir}")

        # Trainer-config prefixes + recognized stage-level prefixes from Step 0 audit
        VALID_PREFIXES = KNOWN_TRAINER_PREFIXES | frozenset({
            "backtesting",  # backtesting.params.*
            "extraction",  # extraction.config
            "signal_export",  # signal_export.X
            "validation",  # validation.X
            "stages",  # stages.X.Y
            "experiment",  # experiment.X (metadata, not override target)
        })

        violations: list[str] = []
        for path in manifests_dir.rglob("*.yaml"):
            try:
                with path.open() as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError:
                continue  # broken YAML; skip
            if data is None or not isinstance(data, dict):
                continue
            # Recursively collect all dotted keys at any "overrides:" block
            self._check_overrides(data, VALID_PREFIXES, violations, str(path.relative_to(PIPELINE_ROOT)))

        assert not violations, (
            f"Production manifests contain overrides with unknown top-level prefix "
            f"(#PY-131 typo class active): {violations}"
        )

    def _check_overrides(
        self, obj: Any, valid_prefixes: frozenset, violations: list, manifest_name: str
    ) -> None:
        """Recursively walk a manifest dict; flag any 'overrides:' dict with unknown-prefix keys."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "overrides" and isinstance(value, dict):
                    for override_key in value.keys():
                        if "." not in str(override_key):
                            continue  # bare key, skip (validated separately)
                        first_seg = str(override_key).split(".", 1)[0]
                        if first_seg not in valid_prefixes:
                            violations.append(
                                f"{manifest_name}: override key '{override_key}' "
                                f"has unknown prefix '{first_seg}'"
                            )
                self._check_overrides(value, valid_prefixes, violations, manifest_name)
        elif isinstance(obj, list):
            for item in obj:
                self._check_overrides(item, valid_prefixes, violations, manifest_name)
