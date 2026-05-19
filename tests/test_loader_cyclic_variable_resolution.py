"""F-1 (audit §4.1, 2026-05-19) regression tests — manifest variable resolver
fail-loud on unresolved ``${...}`` literals post-fixed-point.

Pre-impl gate APPROVE-WITH-MICRO-FIX 2026-05-19: collects ALL unresolved refs
(not just first); excludes ``_DEFERRED_PREFIXES`` (legitimately resolved later
at stage-runner time, not manifest load time).

Tests against the `_resolve_variables` function in `hft_ops.manifest.loader`.
The pre-F-1 fixed-point loop silently terminated on cyclic ``${A.x} ↔ ${B.x}``
or unresolvable dotted-paths, letting ``${...}`` literals propagate downstream
as silent broken state (hft-rules §8 violation). Post-F-1, every unresolved
reference left over after the fixed-point loop raises ValueError with
actionable diagnostics listing every offending (dotted_path, key) pair.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hft_ops.manifest.loader import _resolve_variables


@pytest.fixture
def paths():
    """`paths=None` is supported by _resolve_variables; cyclic-detection
    logic doesn't need PipelinePaths (only _maybe_rebase_path side path does)."""
    return None


@pytest.fixture
def now():
    """Frozen UTC timestamp for deterministic resolver behavior."""
    return datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


class TestF1CyclicVariableResolution:
    """Pre-F-1 silently ate cyclic refs; post-F-1 raises ValueError listing all."""

    def test_cyclic_AB_BA_raises_with_both_paths(self, paths, now):
        """A.x → B.x → A.x cycle: must raise listing BOTH unresolved refs."""
        raw = {"A": {"x": "${B.x}"}, "B": {"x": "${A.x}"}}
        with pytest.raises(ValueError, match=r"unresolved reference"):
            _resolve_variables(raw, now=now, paths=paths)
        # Verify message lists BOTH (not just first)
        try:
            _resolve_variables(raw, now=now, paths=paths)
        except ValueError as e:
            assert "B.x" in str(e), f"Cyclic ref B.x missing from error: {e}"
            assert "A.x" in str(e), f"Cyclic ref A.x missing from error: {e}"

    def test_self_reference_raises(self, paths, now):
        """A.x → A.x self-reference: must raise."""
        raw = {"A": {"x": "${A.x}"}}
        with pytest.raises(ValueError, match=r"unresolved reference"):
            _resolve_variables(raw, now=now, paths=paths)

    def test_deferred_resolved_prefix_does_not_raise(self, paths, now):
        """${resolved.horizon_idx} is deferred; must NOT raise (legitimate)."""
        raw = {
            "stages": {
                "backtesting": {"extra_args": ["--primary-horizon-idx", "${resolved.horizon_idx}"]}
            }
        }
        # No raise — deferred prefix is excluded from the fail-loud gate
        result = _resolve_variables(raw, now=now, paths=paths)
        # Literal ${resolved.horizon_idx} should be preserved
        assert "${resolved.horizon_idx}" in str(result), (
            "Deferred ${resolved.*} literal should be preserved post-resolution"
        )

    def test_unresolved_dotted_path_raises(self, paths, now):
        """${nonexistent.key} (no source): must raise."""
        raw = {"experiment": {"name": "test"}, "stages": {"foo": "${nonexistent.key}"}}
        with pytest.raises(ValueError, match=r"unresolved reference"):
            _resolve_variables(raw, now=now, paths=paths)

    def test_resolved_variables_do_not_raise(self, paths, now):
        """Happy path: all refs resolve, no raise."""
        raw = {
            "experiment": {"name": "test_exp"},
            "stages": {"foo": "${experiment.name}_suffix"},
        }
        result = _resolve_variables(raw, now=now, paths=paths)
        assert result["stages"]["foo"] == "test_exp_suffix"

    def test_partially_resolved_with_one_unresolved_raises_listing_one(self, paths, now):
        """Mixed: 1 resolves, 1 unresolved → error lists ONLY the unresolved."""
        raw = {
            "experiment": {"name": "test_exp"},
            "stages": {
                "ok": "${experiment.name}",  # resolves
                "bad": "${nonexistent.path}",  # doesn't
            },
        }
        with pytest.raises(ValueError) as exc_info:
            _resolve_variables(raw, now=now, paths=paths)
        msg = str(exc_info.value)
        assert "nonexistent.path" in msg
        # Resolved ref should NOT appear in error message
        assert "experiment.name" not in msg, (
            f"Resolved ref erroneously in error message: {msg}"
        )

    def test_unresolved_inside_list_raises(self, paths, now):
        """${...} inside a list value must also be detected (list descent)."""
        raw = {
            "stages": {
                "backtesting": {"extra_args": ["--foo", "${nonexistent.bar}"]}
            }
        }
        with pytest.raises(ValueError, match=r"unresolved reference"):
            _resolve_variables(raw, now=now, paths=paths)

    def test_unresolved_inside_tuple_raises(self, paths, now):
        """Wave 2Y D1 (2026-05-19): ${...} inside a tuple value must also be detected.

        YAML loader emits dict/list only, but Python callers (test fixtures,
        sweep expansion, future Pydantic models) may pre-populate the tree
        with tuples — silent skip would let ${...} literals propagate as
        silent broken state (hft-rules §8 violation). Defense-in-depth.
        """
        raw = {
            "stages": {
                # Tuple wrapped manually to simulate Python caller pre-pop
                "backtesting": {"extra_args": ("--foo", "${nonexistent.bar}")}
            }
        }
        with pytest.raises(ValueError, match=r"unresolved reference"):
            _resolve_variables(raw, now=now, paths=paths)
