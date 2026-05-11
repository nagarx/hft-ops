"""Phase R-17 F3: tests for hft_ops.stages._override_discipline SSoT.

Tests apply_override_loud() public surface + private helper functions.
Closes #PY-128 + NEW-BUG-9/10/11/16 cluster regression coverage.

Per G4 agent edge-case identification:
- Empty key string
- Key with None value (vs missing)
- Empty known_keys frozenset
- Clobber + unknown key combo (order: unknown_key check first)
- Intermediate scalar → dict collision
- Equality semantics (1 == 1.0 no-conflict)
- Unicode keys
"""

from __future__ import annotations

import warnings

import pytest

from hft_ops.stages._override_discipline import (
    OverrideConflictError,
    UnknownOverrideKeyError,
    _get_nested,
    _set_nested,
    _suggest_close_match,
    apply_override_loud,
)


# =============================================================================
# Test class: TestApplyOverrideLoud — public API
# =============================================================================


class TestApplyOverrideLoud:
    """Phase R-17 F3: apply_override_loud() core behavior."""

    def test_adds_new_key(self):
        """Basic: adds new key without conflict."""
        target = {}
        apply_override_loud(target, "data.horizon_idx", 5, source="test")
        assert target == {"data": {"horizon_idx": 5}}

    def test_warns_on_clobber_when_user_set_check_false(self):
        """WARN path: existing value differs + user_set_check=False emits warning."""
        target = {"data": {"horizon_idx": 0}}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                target, "data.horizon_idx", 5,
                source="test source",
                user_set_check=False,
            )
            warning_messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
            assert any("data.horizon_idx" in m for m in warning_messages), (
                f"Expected UserWarning mentioning 'data.horizon_idx', got: {warning_messages}"
            )
            assert any("test source" in m for m in warning_messages), (
                "Warning must include source attribution"
            )
        # Override IS applied despite warning
        assert target["data"]["horizon_idx"] == 5

    def test_raises_on_conflict_when_user_set_check_true(self):
        """RAISE path: existing value differs + user_set_check=True (default) raises."""
        target = {"data": {"horizon_idx": 0}}
        with pytest.raises(OverrideConflictError, match="data.horizon_idx"):
            apply_override_loud(
                target, "data.horizon_idx", 5,
                source="test",
                # user_set_check=True is default
            )
        # Override NOT applied on raise
        assert target["data"]["horizon_idx"] == 0

    def test_passes_when_value_matches_existing(self):
        """No-op path: existing value == new value, no conflict/warning."""
        target = {"data": {"horizon_idx": 5}}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(target, "data.horizon_idx", 5, source="test")
            # No UserWarning when values match
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings, f"Unexpected warning when values match: {user_warnings}"
        assert target["data"]["horizon_idx"] == 5

    def test_raises_on_unknown_key_with_close_match(self):
        """Typo detection: key not in known_keys raises UnknownOverrideKeyError."""
        target = {}
        known = frozenset({"data.horizon_idx", "data.feature_count", "train.seed"})
        with pytest.raises(UnknownOverrideKeyError, match="horzion_idx"):  # typo'd
            apply_override_loud(
                target, "data.horzion_idx", 5,  # typo: horzion vs horizon
                source="test",
                known_keys=known,
            )

    def test_close_match_suggestion_in_error_message(self):
        """Typo error message includes close-match suggestion."""
        target = {}
        known = frozenset({"data.horizon_idx", "train.seed"})
        try:
            apply_override_loud(
                target, "data.horzion_idx", 5,
                source="test",
                known_keys=known,
            )
        except UnknownOverrideKeyError as e:
            # Should suggest the close match
            assert "horizon_idx" in str(e), (
                f"Error should suggest close match, got: {e}"
            )

    def test_handles_nested_dotted_keys(self):
        """Multi-level dotted keys resolve through nested dicts."""
        target = {"data": {"labels": {"primary_horizon_idx": 0}}}
        apply_override_loud(
            target, "data.labels.primary_horizon_idx", 2,
            source="test",
            user_set_check=False,
        )
        assert target["data"]["labels"]["primary_horizon_idx"] == 2

    def test_handles_missing_intermediate_dicts(self):
        """Auto-create intermediate dicts when nested path doesn't exist."""
        target = {}
        apply_override_loud(target, "deep.nested.key", "value", source="test")
        assert target == {"deep": {"nested": {"key": "value"}}}

    def test_empty_key_string_raises(self):
        """Empty key string raises ValueError per hft-rules §5."""
        with pytest.raises(ValueError, match="empty or invalid key"):
            apply_override_loud({}, "", 5, source="test")

    def test_value_of_none_explicit_no_conflict(self):
        """Setting None explicitly (not just-missing) is preserved + no conflict."""
        target = {"data": {"horizon_idx": None}}
        # When existing is None and new value is None, no-op semantics
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(target, "data.horizon_idx", None, source="test")
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings
        assert target["data"]["horizon_idx"] is None

    def test_clobber_with_unknown_key_raises_unknown_first(self):
        """Order invariant: when both unknown_key + clobber apply, unknown wins."""
        target = {"data": {"horzion_idx": 0}}  # typo'd existing
        known = frozenset({"data.horizon_idx"})  # only correct key in known
        # Should raise UnknownOverrideKeyError (typo check fires first), NOT OverrideConflictError
        with pytest.raises(UnknownOverrideKeyError):
            apply_override_loud(
                target, "data.horzion_idx", 5,  # same typo'd key
                source="test",
                known_keys=known,
            )

    def test_intermediate_scalar_collision_raises(self):
        """When intermediate path key is scalar (not dict), raises ValueError."""
        target = {"data": 42}  # scalar at "data" level
        with pytest.raises(ValueError, match="scalar"):
            apply_override_loud(
                target, "data.horizon_idx", 5,
                source="test",
            )

    def test_python_eq_semantics_no_conflict_for_int_vs_float(self):
        """Equality uses Python ==: 1 == 1.0 is True → no conflict."""
        target = {"data": {"horizon_idx": 1}}  # int
        # 1.0 == 1 is True; should NOT trigger conflict
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            apply_override_loud(
                target, "data.horizon_idx", 1.0,
                source="test",
                user_set_check=True,  # would raise if conflict detected
            )
            # No warning, no exception
            user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
            assert not user_warnings


# =============================================================================
# Test class: TestPrivateHelpers — _get_nested, _set_nested, _suggest_close_match
# =============================================================================


class TestGetNested:
    """Phase R-17 F3: _get_nested helper coverage."""

    def test_top_level_key(self):
        assert _get_nested({"x": 1}, "x") == 1

    def test_nested_key(self):
        assert _get_nested({"a": {"b": {"c": "deep"}}}, "a.b.c") == "deep"

    def test_missing_top_level(self):
        assert _get_nested({}, "x") is None

    def test_missing_nested(self):
        assert _get_nested({"a": {}}, "a.b.c") is None

    def test_scalar_at_intermediate(self):
        """Returns None when intermediate path is scalar (not dict)."""
        assert _get_nested({"a": 42}, "a.b.c") is None


class TestSetNested:
    """Phase R-17 F3: _set_nested helper coverage."""

    def test_creates_intermediate_dicts(self):
        target = {}
        _set_nested(target, "a.b.c", "value")
        assert target == {"a": {"b": {"c": "value"}}}

    def test_overwrites_existing_value(self):
        target = {"a": {"b": 0}}
        _set_nested(target, "a.b", 5)
        assert target == {"a": {"b": 5}}

    def test_raises_on_intermediate_scalar(self):
        target = {"a": 42}
        with pytest.raises(ValueError, match="scalar"):
            _set_nested(target, "a.b", "value")

    def test_preserves_sibling_keys(self):
        """Adding a new nested key doesn't clobber siblings."""
        target = {"data": {"horizon_idx": 0}}
        _set_nested(target, "data.feature_count", 98)
        assert target == {"data": {"horizon_idx": 0, "feature_count": 98}}


class TestSuggestCloseMatch:
    """Phase R-17 F3: _suggest_close_match helper coverage."""

    def test_close_match_returned(self):
        known = frozenset({"data.horizon_idx", "data.feature_count"})
        assert _suggest_close_match("data.horzion_idx", known) == "data.horizon_idx"

    def test_no_close_match_fallback(self):
        known = frozenset({"data.horizon_idx", "data.feature_count"})
        result = _suggest_close_match("totally_unrelated_zzzzzz", known)
        assert result == "<no close match>"

    def test_empty_known_keys(self):
        result = _suggest_close_match("any_key", frozenset())
        assert result == "<no close match>"
