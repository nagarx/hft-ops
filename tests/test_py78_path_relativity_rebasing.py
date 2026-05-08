"""Tests for #PY-78 path-relativity orchestrator fix (Phase α-1 / 2026-05-10).

Locks the path-base aware ``${...}`` substitution behavior:
- Source slot ``stages.extraction.output_dir`` (pipeline-root-relative) →
  target slot ``stages.training.overrides.data.data_dir`` or
  ``stages.training.trainer_config.data.data_dir`` (trainer-cwd-relative)
  MUST rebase via ``paths.resolve() + os.path.relpath(..., trainer_dir)``.
- Source PIPELINE_ROOT → target PIPELINE_ROOT must NOT rebase (e.g.,
  ``backtesting.data_dir``, ``backtesting.model_checkpoint``).
- ``paths=None`` (legacy / test fixtures lacking pipeline_root) MUST behave
  exactly like before this fix landed (substitution-only, no rebasing).

Locks the behavior empirically validated by the Cycle 5 multi-arm sweep
manifest (which would FAIL on launch as-is per 9-agent V3 verdict pre-fix).
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hft_ops.manifest.loader import _resolve_variables
from hft_ops.paths import PipelinePaths


# ---------------------------------------------------------------------------
# Core path-rebase semantics (empirical match to Cycle 5 launch scenario)
# ---------------------------------------------------------------------------


class TestPathRebaseAcrossCoordinates:
    """When source PIPELINE_ROOT substitutes into target TRAINER_CWD, value
    must be rebased. This is the critical #PY-78 behavior."""

    def test_overrides_data_dir_gets_rebased(self, tmp_pipeline: Path):
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # tmp_pipeline/data/exports/foo → relpath(tmp_pipeline/lob-model-trainer)
        # = ../data/exports/foo
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "../data/exports/foo"
        )

    def test_trainer_config_data_dir_gets_rebased(self, tmp_pipeline: Path):
        # The Cycle 5 launch scenario: nested data.data_dir inside trainer_config.
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/e5_timebased_60s_v3p0"},
                "training": {
                    "trainer_config": {
                        "data": {
                            "data_dir": "${stages.extraction.output_dir}",
                        },
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        assert (
            resolved["stages"]["training"]["trainer_config"]["data"]["data_dir"]
            == "../data/exports/e5_timebased_60s_v3p0"
        )

    def test_feature_sets_dir_also_rebased(self, tmp_pipeline: Path):
        # feature_sets_dir is also TRAINER_CWD per slot_taxonomy.
        raw = {
            "stages": {
                "extraction": {"output_dir": "contracts/feature_sets"},
                "training": {
                    "trainer_config": {
                        "data": {
                            "feature_sets_dir": "${stages.extraction.output_dir}",
                        },
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        assert (
            resolved["stages"]["training"]["trainer_config"]["data"]["feature_sets_dir"]
            == "../contracts/feature_sets"
        )


# ---------------------------------------------------------------------------
# Negative cases: same-coordinate substitutions must NOT rebase
# ---------------------------------------------------------------------------


class TestNoRebaseWhenSameCoordinates:
    """When source and target are both PIPELINE_ROOT, no rebasing happens."""

    def test_backtesting_data_dir_not_rebased(self, tmp_pipeline: Path):
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "backtesting": {
                    "data_dir": "${stages.extraction.output_dir}",
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # Both PIPELINE_ROOT → no rebase, value is the source verbatim.
        assert resolved["stages"]["backtesting"]["data_dir"] == "data/exports/foo"

    def test_signal_export_checkpoint_not_rebased(self, tmp_pipeline: Path):
        raw = {
            "stages": {
                "training": {"output_dir": "outputs/experiments/foo"},
                "signal_export": {
                    "checkpoint": "${stages.training.output_dir}/checkpoints/best.pt",
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        assert (
            resolved["stages"]["signal_export"]["checkpoint"]
            == "outputs/experiments/foo/checkpoints/best.pt"
        )


# ---------------------------------------------------------------------------
# Backward compatibility: paths=None preserves legacy behavior
# ---------------------------------------------------------------------------


class TestPathsNoneLegacyBehavior:
    """When paths is None (legacy callers), behavior is exactly as before
    Phase α-1 — substitution happens, no rebasing."""

    def test_paths_none_does_not_rebase_trainer_cwd_slot(self):
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=None)
        # Without paths, no rebasing — substitution returns source value verbatim.
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "data/exports/foo"
        )

    def test_paths_omitted_kwarg_legacy(self):
        # Don't even pass the paths kwarg — verifies the default value
        # produces legacy behavior (no rebasing).
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now)  # no paths kwarg
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "data/exports/foo"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRebaseAbsolutePathShortCircuit:
    """HIGH-1 (mid-impl review): absolute source paths are cwd-independent.
    The rebase short-circuits and returns the absolute path verbatim, avoiding
    fragile cross-mount `../../../...` relative paths."""

    def test_absolute_source_value_skipped(self, tmp_pipeline: Path):
        # Source value is absolute. Rebase MUST NOT produce a cross-mount
        # relative path; instead, return the absolute literal as-is.
        raw = {
            "stages": {
                "extraction": {"output_dir": "/Volumes/WD_Black/HFT-data/exports/foo"},
                "training": {
                    "trainer_config": {
                        "data": {
                            "data_dir": "${stages.extraction.output_dir}",
                        },
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        assert (
            resolved["stages"]["training"]["trainer_config"]["data"]["data_dir"]
            == "/Volumes/WD_Black/HFT-data/exports/foo"
        )

    def test_relative_source_value_still_rebased(self, tmp_pipeline: Path):
        # Negative regression: a RELATIVE source value (the common case) must
        # still rebase. Locks the short-circuit to absolute-only.
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "trainer_config": {
                        "data": {
                            "data_dir": "${stages.extraction.output_dir}",
                        },
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        assert (
            resolved["stages"]["training"]["trainer_config"]["data"]["data_dir"]
            == "../data/exports/foo"
        )


class TestRebaseFailLoud:
    """HIGH-2 (mid-impl review): when paths.resolve() fails (e.g., broken
    symlink, permission error), the error MUST surface with actionable
    diagnostic citing slot keys + source value. Per hft-rules §8."""

    def test_resolve_error_raises_with_actionable_message(
        self, tmp_pipeline: Path, monkeypatch
    ):
        # Force paths.resolve to raise OSError to simulate filesystem issues.
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        def _raise_oserror(self, _relative_path):
            raise OSError("simulated filesystem error")

        monkeypatch.setattr(PipelinePaths, "resolve", _raise_oserror)
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Cannot rebase manifest path"):
            _resolve_variables(raw, now=now, paths=paths)

    def test_resolve_error_message_includes_slot_keys(
        self, tmp_pipeline: Path, monkeypatch
    ):
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        def _raise_oserror(self, _relative_path):
            raise OSError("network volume unreachable")

        monkeypatch.setattr(PipelinePaths, "resolve", _raise_oserror)
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        with pytest.raises(ValueError) as exc_info:
            _resolve_variables(raw, now=now, paths=paths)
        msg = str(exc_info.value)
        # Must cite the source slot, target slot, source value, and underlying error.
        assert "stages.extraction.output_dir" in msg
        assert "stages.training.overrides.data.data_dir" in msg
        assert "data/exports/foo" in msg
        assert "network volume unreachable" in msg


class TestRebaseEdgeCases:
    """Edge cases: deferred prefixes, builtin vars, extra_vars, lists."""

    def test_deferred_resolved_prefix_not_substituted(self, tmp_pipeline: Path):
        raw = {
            "stages": {
                "training": {
                    "overrides": {
                        "data.data_dir": "${resolved.something}",
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # Deferred prefix preserved verbatim — no substitution, no rebasing.
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "${resolved.something}"
        )

    def test_builtin_timestamp_not_rebased(self, tmp_pipeline: Path):
        raw = {
            "stages": {
                "training": {
                    "overrides": {
                        "data.data_dir": "${timestamp}",
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # Builtin var is not classifiable by slot_taxonomy as PIPELINE_ROOT,
        # so no rebasing happens. Substitution returns timestamp string as-is.
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "20260510T120000"
        )

    def test_lists_pass_through_path_unchanged(self, tmp_pipeline: Path):
        # Lists don't extend the path. Substitutions inside list elements
        # use the LIST'S slot, not [idx] paths.
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "tags": ["${stages.extraction.output_dir}", "tag2"],
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # tags slot is not a path slot (NONE), so no rebasing.
        assert resolved["stages"]["training"]["tags"][0] == "data/exports/foo"
        assert resolved["stages"]["training"]["tags"][1] == "tag2"


# ---------------------------------------------------------------------------
# Multi-pass + transitive references (max_passes=5 fixed-point)
# ---------------------------------------------------------------------------


class TestMultiPassRebase:
    """Transitive substitutions resolve correctly across multiple passes."""

    def test_transitive_substitution_with_rebase(self, tmp_pipeline: Path):
        # A → B → C, where A is a PIPELINE_ROOT slot and C is TRAINER_CWD.
        # Multi-pass should still rebase at the final step.
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "output_dir": "${stages.extraction.output_dir}/training",
                    "overrides": {
                        # Transitive: substituting through training.output_dir
                        # doesn't make this rebase, since training.output_dir
                        # is also PIPELINE_ROOT. End result is rebased once.
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        resolved = _resolve_variables(raw, now=now, paths=paths)
        # training.output_dir = "data/exports/foo/training" (no rebase, both PIPELINE_ROOT)
        assert (
            resolved["stages"]["training"]["output_dir"]
            == "data/exports/foo/training"
        )
        # overrides.data.data_dir = "../data/exports/foo" (rebased)
        assert (
            resolved["stages"]["training"]["overrides"]["data.data_dir"]
            == "../data/exports/foo"
        )

    def test_idempotent_repeated_resolution(self, tmp_pipeline: Path):
        # Resolving twice produces the same result (max_passes=5 reaches
        # fixed point after first pass).
        raw = {
            "stages": {
                "extraction": {"output_dir": "data/exports/foo"},
                "training": {
                    "overrides": {
                        "data.data_dir": "${stages.extraction.output_dir}",
                    },
                },
            },
        }
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        now = datetime(2026, 5, 10, tzinfo=timezone.utc)
        once = _resolve_variables(raw, now=now, paths=paths)
        twice = _resolve_variables(once, now=now, paths=paths)
        assert once == twice
