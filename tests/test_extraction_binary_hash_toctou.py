"""The extraction cache key must describe the binary that ACTUALLY RUNS.

WHAT THIS LOCKS
---------------
``compiled_binary_sha256`` is one of the 9 canonical ``CacheKeyInputs`` and is
read from ``feature-extractor-MBO-LOB/target/release/export_dataset``. Until
2026-08-11 it was hashed at ``stages/extraction.py`` line ~117, while the
extraction itself ran ``cargo run --release`` at line ~174 — and ``cargo run``
REBUILDS whenever a source changed.

Those are two different artifacts whenever a source edit is pending, and the
window is precisely the one that matters. ``MBO-LOB-reconstructor`` is consumed
through a path ``[patch]`` in ``feature-extractor-MBO-LOB/.cargo/config.toml``
(gitignored), so an uncommitted edit there reaches the extractor build with no
tag moving and no git SHA changing. The failure sequence:

    1. edit MBO-LOB-reconstructor (uncommitted; e.g. the T/F decoder fix)
    2. do NOT rebuild by hand
    3. run the extraction stage
    4. the cache key is computed from the STALE binary          -> key K_old
    5. cache MISS on K_old
    6. `cargo run` REBUILDS with the edit and extracts
    7. the corrected output is populated under K_old

...which names the pre-edit binary. The reverse direction is equally wrong: a
pre-fix cached extraction stays reachable until something rebuilds the binary.

WHY NOT A NEW KEY FIELD
-----------------------
The obvious-looking fix — add a ``reconstructor_dirty`` input — is worse on
every axis and is deliberately NOT what this locks:

  * ``CacheKeyInputs``' own docstring requires a MAJOR
    ``CACHE_MANIFEST_SCHEMA_VERSION`` bump for any new field, which invalidates
    every existing cache entry.
  * git-dirtiness is a PROXY. The compiled binary hash is the ground truth: a
    source edit can only change the produced data by changing the binary.
  * dirtiness false-invalidates the cache on a docs-only reconstructor edit.

The fix is instead to BUILD BEFORE HASHING, so the hashed artifact is the
artifact that runs. No schema change; no cache invalidation.

REVERT BEHAVIOUR
----------------
Delete the pre-hash ``cargo build`` block from ``ExtractionRunner.run`` and
``test_build_precedes_cache_key_computation`` fails: the recorded call order no
longer has a build before ``prepare_cache_key_inputs``.
"""

from __future__ import annotations

import types
from pathlib import Path

from hft_ops.config import OpsConfig
from hft_ops.manifest.schema import (
    ExperimentHeader,
    ExperimentManifest,
    ExtractionStage,
    Stages,
)
import hft_ops.stages.extraction as extraction_mod
from hft_ops.stages.extraction import ExtractionRunner

# Resolved lazily, NOT imported by name. A module-level
# `from ... import _EXTRACTOR_CARGO_FEATURES` would make the whole file fail to
# COLLECT against a pre-fix tree, and a collection error is not a demonstration
# that these assertions detect anything. Resolving it per-test lets the ordering
# test below run against the unfixed code and fail on its own assertion, which
# is the only evidence that it has detection power.
_FEATURES_FALLBACK = "parallel"


def _pipeline_root(tmp_path: Path) -> Path:
    for rel in (
        "feature-extractor-MBO-LOB/configs",
        "MBO-LOB-reconstructor",
        "data/exports",
        "hft-ops/ledger",
        "contracts",
        "lob-model-trainer",
        "lob-backtester",
    ):
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    (tmp_path / "feature-extractor-MBO-LOB/configs/x.toml").write_text("")
    (tmp_path / "contracts/pipeline_contract.toml").write_text("")
    return tmp_path


def _manifest(name: str) -> ExperimentManifest:
    return ExperimentManifest(
        experiment=ExperimentHeader(name=name),
        stages=Stages(
            extraction=ExtractionStage(
                config="feature-extractor-MBO-LOB/configs/x.toml",
                output_dir=f"data/exports/{name}",
            ),
        ),
    )


def _install_spies(monkeypatch, trace: list, *, export_dir: Path):
    """Record the ORDER of the two events the defect is about."""

    def _run_subprocess(cmd, cwd=None, verbose=False, env=None):
        trace.append(("subprocess", list(cmd)))
        if "run" in cmd:  # the extraction itself — write a valid export
            split = export_dir / "train"
            split.mkdir(parents=True, exist_ok=True)
            (split / "20250203_metadata.json").write_text("{}")
            (split / "20250203_sequences.npy").write_bytes(b"")
            (split / "20250203_regression_labels.npy").write_bytes(b"")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def _prepare(**kw):
        # This is the call that reads target/release/export_dataset and folds
        # its sha256 into the key. Its position in `trace` is the whole point.
        trace.append(("hash_binary", None))
        return types.SimpleNamespace()

    monkeypatch.setattr("hft_ops.stages.extraction.run_subprocess", _run_subprocess)
    monkeypatch.setattr("hft_ops.stages.extraction.prepare_cache_key_inputs", _prepare)
    monkeypatch.setattr(
        "hft_ops.stages.extraction.compute_cache_key", lambda inputs: "k" * 64
    )
    monkeypatch.setattr(
        "hft_ops.stages.extraction.resolve_or_link",
        lambda key, out, root: types.SimpleNamespace(
            status="miss", seconds_saved=0.0, linked_files=0, link_type=""
        ),
    )
    monkeypatch.setattr("hft_ops.stages.extraction.populate", lambda *a, **kw: None)


class TestBinaryHashDescribesTheBinaryThatRuns:
    def test_build_precedes_cache_key_computation(self, tmp_path, monkeypatch):
        """A `cargo build` must occur BEFORE the binary is hashed.

        Without it, the hashed binary and the executed binary can differ, and
        the extraction is cached under a key that names the wrong artifact.
        """
        root = _pipeline_root(tmp_path)
        name = "toctou_order"
        export_dir = root / "data" / "exports" / name
        export_dir.mkdir(parents=True, exist_ok=True)
        trace: list = []
        _install_spies(monkeypatch, trace, export_dir=export_dir)

        ExtractionRunner().run(
            _manifest(name), OpsConfig.from_pipeline_root(pipeline_root=root)
        )

        kinds = [k for k, _ in trace]
        assert "hash_binary" in kinds, (
            "The cache key was never computed — this test cannot observe the "
            "ordering it exists to lock. Fixture is wrong, not the code."
        )
        build_positions = [
            i
            for i, (k, cmd) in enumerate(trace)
            if k == "subprocess" and "build" in cmd
        ]
        hash_position = kinds.index("hash_binary")

        assert build_positions, (
            "No `cargo build` ran before the extraction. The cache key is "
            "therefore computed from whatever binary happens to be on disk, "
            "while `cargo run` below may rebuild — so a cache entry can be "
            "written under a key naming a binary that never produced it."
        )
        assert build_positions[0] < hash_position, (
            "`cargo build` ran but AFTER the binary was hashed "
            f"(build at {build_positions[0]}, hash at {hash_position}). The "
            "build must precede the hash, or the hash still describes the "
            "stale artifact."
        )

    def test_build_and_run_use_the_same_feature_set(self, tmp_path, monkeypatch):
        """A feature mismatch would make cargo rebuild between hash and run,
        re-opening the same window through a different door."""
        root = _pipeline_root(tmp_path)
        name = "toctou_features"
        export_dir = root / "data" / "exports" / name
        export_dir.mkdir(parents=True, exist_ok=True)
        trace: list = []
        _install_spies(monkeypatch, trace, export_dir=export_dir)

        ExtractionRunner().run(
            _manifest(name), OpsConfig.from_pipeline_root(pipeline_root=root)
        )

        def _features(cmd):
            return cmd[cmd.index("--features") + 1]

        cmds = [cmd for k, cmd in trace if k == "subprocess"]
        build = next(c for c in cmds if "build" in c)
        run = next(c for c in cmds if "run" in c)

        expected = getattr(
            extraction_mod, "_EXTRACTOR_CARGO_FEATURES", _FEATURES_FALLBACK
        )
        assert _features(build) == _features(run) == expected, (
            "The pre-hash build and the extraction run must request identical "
            f"cargo features; got build={_features(build)!r} "
            f"run={_features(run)!r} constant={expected!r}. "
            "Differing features produce different binaries, so cargo would "
            "rebuild between the hash and the run."
        )

    def test_dry_run_does_not_build(self, tmp_path, monkeypatch):
        """Control arm: --dry-run must stay side-effect-free.

        Without this, the fix above would silently turn `--dry-run` into a
        compile, which is exactly what a dry run is for avoiding.
        """
        root = _pipeline_root(tmp_path)
        name = "toctou_dry"
        export_dir = root / "data" / "exports" / name
        export_dir.mkdir(parents=True, exist_ok=True)
        trace: list = []
        _install_spies(monkeypatch, trace, export_dir=export_dir)

        ExtractionRunner().run(
            _manifest(name),
            OpsConfig.from_pipeline_root(pipeline_root=root, dry_run=True),
        )

        builds = [cmd for k, cmd in trace if k == "subprocess" and "build" in cmd]
        assert builds == [], (
            f"--dry-run compiled the extractor: {builds}. A dry run must not "
            "have side effects on the build tree."
        )

    def test_no_build_when_caching_disabled(self, tmp_path, monkeypatch):
        """Control arm: with caching off there is no key to poison, so the
        pre-hash build is unnecessary work and must not run.

        This also stops the test above from passing for the wrong reason —
        the build is conditional on the cache path, not unconditional.
        """
        root = _pipeline_root(tmp_path)
        name = "toctou_nocache"
        export_dir = root / "data" / "exports" / name
        export_dir.mkdir(parents=True, exist_ok=True)
        trace: list = []
        _install_spies(monkeypatch, trace, export_dir=export_dir)

        ExtractionRunner().run(
            _manifest(name),
            OpsConfig.from_pipeline_root(pipeline_root=root, cache_extraction=False),
        )

        builds = [cmd for k, cmd in trace if k == "subprocess" and "build" in cmd]
        assert builds == [], (
            f"A pre-hash build ran with caching disabled: {builds}. There is "
            "no cache key to describe, so the build is pure overhead."
        )
