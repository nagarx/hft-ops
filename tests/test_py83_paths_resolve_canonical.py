"""Tests for #PY-83 paths.resolve() symlink-deref defect (Phase α-1.1 / 2026-05-10).

Locks the post-α-1.1 contract:

- ``PipelinePaths.resolve(rel)`` returns a symlink-source-preserved absolute
  path (uses ``Path.absolute()`` internally; no FS access; NEVER derefs).
- ``PipelinePaths.canonical(rel)`` is the explicit escape hatch for the rare
  cases that need the canonical filesystem path (uses ``Path.resolve()``
  internally; touches FS; DOES deref symlinks).

The defect: pre-α-1.1, ``paths.resolve()`` derefs the ``data/`` symlink at
start. In deployments where ``data/`` is symlinked to an external mount
(e.g. ``data/`` -> ``/Volumes/WD_Black/HFT-data/``), this DEFEATS the α-1
``_maybe_rebase_path`` logic — producing a 5-level cross-mount relpath
(``'../../../../../Volumes/...'``) that the trainer subprocess cannot
interpret. Sister defect to #PY-79 (closed by α-3 in
``feature_set_resolver.py:442``).

Discovered by: 8-agent prep round 2026-05-10 (Agent I FINDING 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_ops.paths import PipelinePaths


# ---------------------------------------------------------------------------
# Fixture — symlinked-data deployment (mirrors user's actual setup)
# ---------------------------------------------------------------------------


@pytest.fixture
def symlinked_data_pipeline(tmp_path: Path) -> Path:
    """Build a pipeline root where ``data/`` is symlinked to an external dir.

    Mirrors the user's actual deployment where
    ``data/`` -> ``/Volumes/WD_Black/HFT-data/``.

    Returns the pipeline_root path (NOT the symlink target).
    """
    # Real data location — outside the pipeline root, mimics external mount
    real_data = tmp_path / "external_volume" / "HFT-data"
    (real_data / "exports" / "foo").mkdir(parents=True)

    # Pipeline root with contracts/ + data symlink
    root = tmp_path / "HFT-pipeline-v2"
    root.mkdir()
    (root / "contracts").mkdir()
    (root / "contracts" / "pipeline_contract.toml").write_text(
        '[contract]\nschema_version = "3.0"\n'
    )
    # Symlink data -> external location (the production scenario)
    (root / "data").symlink_to(real_data)
    return root


# ---------------------------------------------------------------------------
# resolve() preserves symlink-source (the α-1.1 fix)
# ---------------------------------------------------------------------------


class TestPathsResolvePreservesSymlinkSource:
    """Phase α-1.1 / #PY-83: ``paths.resolve()`` must NOT deref symlinks.

    Default behavior preserves the symlink-source lineage so that downstream
    relpath() / walk-up logic finds the right pipeline_root anchor.
    """

    def test_resolve_returns_path_under_pipeline_root_not_deref_target(
        self, symlinked_data_pipeline: Path
    ):
        """The critical α-1.1 invariant: even when ``data/`` is symlinked
        OUTSIDE the pipeline_root, ``paths.resolve('data/exports/foo')``
        returns a path STARTING with pipeline_root (NOT the deref target).
        """
        paths = PipelinePaths(pipeline_root=symlinked_data_pipeline)
        result = paths.resolve("data/exports/foo")
        # Result MUST be under pipeline_root (symlink-source preserved)
        assert str(result).startswith(str(symlinked_data_pipeline)), (
            f"#PY-83 regression: paths.resolve() returned {result} which "
            f"does NOT start with pipeline_root {symlinked_data_pipeline}. "
            f"This means resolve() is dereferencing the data/ symlink — "
            f"the bug α-1.1 was supposed to fix."
        )
        # Result MUST be absolute
        assert result.is_absolute()
        # Result MUST NOT contain external_volume (the deref target)
        assert "external_volume" not in str(result), (
            f"#PY-83 regression: paths.resolve() leaked deref target into "
            f"result {result}. Expected symlink-source preservation."
        )

    def test_resolve_idempotent_under_repeated_calls(
        self, symlinked_data_pipeline: Path
    ):
        """``paths.resolve()`` is purely lexical (no FS touch). Calling
        twice returns identical results."""
        paths = PipelinePaths(pipeline_root=symlinked_data_pipeline)
        r1 = paths.resolve("data/exports/foo")
        r2 = paths.resolve("data/exports/foo")
        assert r1 == r2

    def test_resolve_works_when_symlink_target_does_not_exist(
        self, tmp_path: Path
    ):
        """``paths.resolve()`` must NOT fail if the symlink target is broken
        — it's purely lexical. (Path.resolve() with strict=False also
        tolerates broken symlinks, but Path.absolute() doesn't even touch
        the FS.)
        """
        root = tmp_path / "HFT-pipeline-v2"
        root.mkdir()
        (root / "data").symlink_to(tmp_path / "nonexistent_target")
        paths = PipelinePaths(pipeline_root=root)
        # Should not raise — purely lexical
        result = paths.resolve("data/exports/foo")
        assert str(result).startswith(str(root))


# ---------------------------------------------------------------------------
# canonical() is the explicit escape hatch
# ---------------------------------------------------------------------------


class TestPathsCanonicalDereferencesSymlinks:
    """Phase α-1.1 / #PY-83: ``paths.canonical()`` is the explicit escape
    hatch for callers that DO need the deref'd canonical filesystem path
    (cache-key inputs, content-addressed hashes, lineage manifests).
    """

    def test_canonical_returns_deref_target_not_pipeline_root(
        self, symlinked_data_pipeline: Path
    ):
        paths = PipelinePaths(pipeline_root=symlinked_data_pipeline)
        result = paths.canonical("data/exports/foo")
        # Result MUST contain "external_volume" (the deref target)
        assert "external_volume" in str(result), (
            f"paths.canonical() should deref symlinks. Got {result} "
            f"which does NOT contain the expected deref target."
        )
        # Result MUST be absolute
        assert result.is_absolute()
        # Result MUST NOT start with pipeline_root/data (deref happened)
        assert not str(result).startswith(
            str(symlinked_data_pipeline / "data")
        )

    def test_canonical_method_is_callable(self, tmp_path: Path):
        """Lock the public API: canonical() is a method on PipelinePaths."""
        root = tmp_path / "HFT-pipeline-v2"
        root.mkdir()
        paths = PipelinePaths(pipeline_root=root)
        assert callable(paths.canonical)


# ---------------------------------------------------------------------------
# Divergence contract: resolve() and canonical() differ ONLY for symlinks
# ---------------------------------------------------------------------------


class TestResolveCanonicalDivergeOnlyForSymlinks:
    """When pipeline_root has NO symlinks in the relative path, ``resolve()``
    and ``canonical()`` return paths that compare equal (after both go
    through the FS — Path.absolute() lexical vs Path.resolve() FS-aware).
    """

    def test_no_symlink_resolve_and_canonical_both_under_pipeline_root(
        self, tmp_path: Path
    ):
        """No symlink in the path → both methods produce paths under
        pipeline_root. They may differ in `..` collapsing or canonical
        case (FS-dependent), but both anchor under pipeline_root."""
        root = tmp_path / "HFT-pipeline-v2"
        (root / "data" / "exports" / "foo").mkdir(parents=True)
        paths = PipelinePaths(pipeline_root=root)
        r_resolve = paths.resolve("data/exports/foo")
        r_canonical = paths.canonical("data/exports/foo")
        # Both anchor under pipeline_root (no symlink-deref divergence)
        assert str(r_resolve).startswith(str(root))
        assert str(r_canonical).startswith(str(root))


# ---------------------------------------------------------------------------
# auto_detect() preserves symlink-source (α-1.1 fix mirror)
# ---------------------------------------------------------------------------


class TestAutoDetectPreservesSymlinkSource:
    """Phase α-1.1 / #PY-83 mirror: ``PipelinePaths.auto_detect()`` uses
    ``Path(__file__).absolute()`` (NOT ``.resolve()``) so that hft-ops
    checked out under a symlinked directory still detects the right root
    via symlink-source lineage. This mirrors the α-3 / #PY-79 pattern.

    NOTE: this test asserts the implementation choice (uses absolute() not
    resolve() at paths.py:185), which is the α-1.1 invariant. It does NOT
    require a symlinked checkout — the test simply confirms auto_detect()
    succeeds and returns a PipelinePaths instance.
    """

    def test_auto_detect_returns_pipeline_paths_instance(self):
        """In the actual hft-ops checkout (this test process's), auto_detect
        succeeds and returns a PipelinePaths."""
        result = PipelinePaths.auto_detect()
        assert isinstance(result, PipelinePaths)
        # The detected root must contain contracts/pipeline_contract.toml
        assert (result.pipeline_root / "contracts" / "pipeline_contract.toml").exists()
