"""Path-base coordinate system for ${...} substitution across manifest slots.

Phase α-1 / #PY-78 (2026-05-10) — closes path-relativity contract mismatch.

Each manifest slot has a fixed PATH-BASE (pipeline-root or trainer-cwd).
When a substitution crosses path-bases (source slot is pipeline-root-relative,
target slot is trainer-cwd-relative), the resolver converts via paths.resolve()
+ os.path.relpath() to restore consumer-expected form.

Without this, a manifest like:

    stages:
      extraction:
        output_dir: "data/exports/foo"            # pipeline-root-relative
      training:
        trainer_config:
          data:
            data_dir: "${stages.extraction.output_dir}"  # trainer-cwd-relative SLOT

silently passes "data/exports/foo" (pipeline-root-relative) into the trainer's
data.data_dir slot which interprets it as cwd-relative (= ``$pipeline_root /
lob-model-trainer/data/exports/foo``) → file-not-found.

This module is the SSoT for slot path-base classification. Per hft-rules §0,
when a NEW slot gets added that holds a path, it MUST be classified here AND
the parametric test coverage extended (see test_slot_taxonomy.py).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple


class PathBase(Enum):
    """Path-base coordinate system for a manifest slot."""

    PIPELINE_ROOT = "pipeline_root"
    """Slot is consumed via ``config.paths.resolve()`` (i.e., interpreted
    relative to pipeline_root). Default for orchestrator-side slots."""

    TRAINER_CWD = "trainer_cwd"
    """Slot is consumed bare by the trainer subprocess (cwd =
    ``<pipeline_root>/lob-model-trainer/``). Path values must be expressed
    relative to that cwd (e.g., ``../data/exports/...``)."""

    NONE = "none"
    """Not a path slot (string fragment, int, bool, name, etc.). Default for
    unmatched dotted-path keys."""


# Slot taxonomy. Each entry is a (compiled regex, PathBase) pair.
# Order: more-specific patterns BEFORE more-general so the first match wins.
#
# To add a new slot: append a regex pattern + PathBase classification here AND
# extend test_slot_taxonomy.py parametric coverage. Per hft-rules §0 reuse-first:
# do NOT re-implement detection logic in consumer code; always import + call
# `detect_slot_path_base(...)` from this SSoT.
_SLOT_PATH_BASE_PATTERNS: List[Tuple[re.Pattern, PathBase]] = [
    # ---------- TRAINER_CWD slots (consumed by trainer subprocess) ----------
    # The trainer's data.data_dir field is consumed relative to cwd=trainer_dir.
    # Both `overrides.data.data_dir` (orchestrator override path) and
    # `trainer_config.data.data_dir` (inline path) flow into the same trainer
    # consumer.
    (re.compile(r"^stages\.training\.overrides\.data\.data_dir$"), PathBase.TRAINER_CWD),
    (re.compile(r"^stages\.training\.trainer_config\.data\.data_dir$"), PathBase.TRAINER_CWD),
    # The trainer's data.feature_sets_dir field is also consumed relative to
    # cwd=trainer_dir (default auto-detect walks up from data_dir).
    (re.compile(r"^stages\.training\.overrides\.data\.feature_sets_dir$"), PathBase.TRAINER_CWD),
    (re.compile(r"^stages\.training\.trainer_config\.data\.feature_sets_dir$"), PathBase.TRAINER_CWD),

    # ---------- PIPELINE_ROOT slots (consumed via paths.resolve()) ----------
    # Stage-level output_dir + script + data_dir slots. All consumed by
    # orchestrator via `config.paths.resolve(stage.output_dir)` etc., so they
    # must be pipeline-root-relative literals (or absolute paths).
    (re.compile(r"^stages\.extraction\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.extraction\.config$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.raw_analysis\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.raw_analysis\.data_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.dataset_analysis\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.dataset_analysis\.data_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.validation\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.training\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.training\.script$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.post_training_gate\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.signal_export\.output_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.signal_export\.checkpoint$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.signal_export\.script$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.signal_export\.config$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.backtesting\.data_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.backtesting\.signals_dir$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.backtesting\.model_checkpoint$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.backtesting\.script$"), PathBase.PIPELINE_ROOT),
    (re.compile(r"^stages\.backtesting\.params_file$"), PathBase.PIPELINE_ROOT),
]


def detect_slot_path_base(slot_dotted_key: str) -> PathBase:
    """Classify a manifest slot's path-base coordinate system.

    Args:
        slot_dotted_key: Dotted-path key for the slot being inspected, e.g.
            ``"stages.extraction.output_dir"`` or
            ``"stages.training.trainer_config.data.data_dir"``.

    Returns:
        The slot's :class:`PathBase` classification. Returns ``PathBase.NONE``
        for unmatched keys (presumed to be non-path slots like
        ``experiment.name`` or ``stages.training.trainer_config.train.epochs``).

    Examples:
        >>> detect_slot_path_base("stages.extraction.output_dir")
        <PathBase.PIPELINE_ROOT: 'pipeline_root'>
        >>> detect_slot_path_base("stages.training.overrides.data.data_dir")
        <PathBase.TRAINER_CWD: 'trainer_cwd'>
        >>> detect_slot_path_base("experiment.name")
        <PathBase.NONE: 'none'>
    """
    for pattern, base in _SLOT_PATH_BASE_PATTERNS:
        if pattern.match(slot_dotted_key):
            return base
    return PathBase.NONE
