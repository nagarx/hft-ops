"""F-2 (audit §4.1, 2026-05-19) regression tests — paths.resolve() traversal guard.

Pre-F-2 `paths.resolve("../../../etc/passwd")` returned a literal cross-tree path
because `Path.absolute()` does NOT normalize `..` segments (stdlib pathlib quirk).
47+ resolve callsites accepted this as legitimate, exposing path-traversal hazard
for untrusted manifest inputs (hft-rules §8 violation).

Pre-impl gate APPROVE-WITH-MICRO-FIX 2026-05-19: SIMPLE design (no env-var
allowlist needed); single `os.path.normpath` + `.relative_to()` probe.
The `canonical()` escape hatch is UNAFFECTED (consumers are trusted-internal:
cache keys, lineage hashes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_ops.paths import PipelinePaths


@pytest.fixture
def paths(tmp_path):
    """Production-like pipeline_root rooted at tmp_path for isolation."""
    return PipelinePaths(pipeline_root=tmp_path)


class TestF2PathTraversalGuard:
    """Pre-F-2 `paths.resolve()` accepted `..` traversal silently; post-F-2 raises."""

    def test_resolve_rejects_dotdot_traversal(self, paths, tmp_path):
        """`paths.resolve("../../../etc/passwd")` must raise ValueError per hft-rules §8."""
        with pytest.raises(ValueError, match=r"escapes pipeline_root"):
            paths.resolve("../../../etc/passwd")

    def test_resolve_rejects_absolute_outside_pipeline_root(self, paths):
        """`paths.resolve("/etc/passwd")` (absolute outside): must raise."""
        with pytest.raises(ValueError, match=r"escapes pipeline_root"):
            paths.resolve("/etc/passwd")

    def test_resolve_accepts_legitimate_relative(self, paths, tmp_path):
        """`paths.resolve("data/exports/foo")` must NOT raise for legitimate use."""
        result = paths.resolve("data/exports/foo")
        assert result == (tmp_path / "data/exports/foo").absolute()

    def test_resolve_accepts_nested_legitimate(self, paths, tmp_path):
        """Deeper nested paths must NOT raise."""
        result = paths.resolve("lob-backtester/scripts/run_regression_backtest.py")
        assert str(result).startswith(str(tmp_path))

    def test_resolve_accepts_absolute_under_pipeline_root(self, paths, tmp_path):
        """Absolute path that IS under pipeline_root must pass."""
        # Under pipeline_root prefix — legitimate
        inside_path = str(tmp_path / "foo/bar")
        result = paths.resolve(inside_path)
        assert str(result) == inside_path

    def test_canonical_unaffected_by_traversal_guard(self, paths, tmp_path):
        """`canonical()` is the documented escape hatch; must NOT have the
        traversal guard (its consumers are trusted-internal: cache keys, lineage)."""
        # canonical() dereferences symlinks; doesn't apply F-2 guard
        # (test verifies traversal returns the resolved path, doesn't raise)
        result = paths.canonical("../foo")  # legitimate per canonical's contract
        # canonical's behavior is .resolve() which DOES normalize; we just verify
        # it doesn't raise the F-2 ValueError specifically
        assert "escapes pipeline_root" not in str(result)

    def test_error_message_actionable(self, paths):
        """Error message must cite both the input path AND pipeline_root + canonical escape hatch hint."""
        with pytest.raises(ValueError) as exc_info:
            paths.resolve("../escape/attempt")
        msg = str(exc_info.value)
        assert "../escape/attempt" in msg, f"Input path not in error: {msg}"
        assert "pipeline_root" in msg, f"pipeline_root not cited: {msg}"
        assert "canonical()" in msg or "escape hatch" in msg, (
            f"canonical() escape hatch hint missing: {msg}"
        )

    def test_resolve_normalizes_dotdot_within_pipeline_root(self, paths, tmp_path):
        """`paths.resolve("foo/../bar")` resolves to `bar` (under pipeline_root) — must NOT raise.

        Wave 2Y D3 (2026-05-19): post-fix return value is NORMALIZED (no
        literal `..` segments) — pre-fix return-value carried `foo/../bar`
        while the guard-check compared the normalized form `bar`, breaking
        Path equality / hashing / string-comparison-based dedup semantics.
        """
        # foo/../bar normalizes to bar — still under pipeline_root
        result = paths.resolve("foo/../bar")
        # Post-D3: return is normalized; consumers see canonical `bar`
        assert result == (tmp_path / "bar").absolute()
        # Defensive: ensure NO literal `..` segment leaks into the returned Path
        assert ".." not in result.parts, (
            f"Path traversal leak: returned path contains `..`: {result}"
        )

    def test_resolve_rejects_mixed_segment_traversal(self, paths):
        """Wave 2Y test gap closure: `data/../../../etc/passwd` mixed-segment must raise.

        Empirically: `os.path.normpath('data/../../../etc/passwd')` =
        `'../../etc/passwd'` which escapes pipeline_root. Verifies the
        guard catches mixed legitimate-prefix + traversal combinations.
        """
        with pytest.raises(ValueError, match=r"escapes pipeline_root"):
            paths.resolve("data/../../../etc/passwd")
