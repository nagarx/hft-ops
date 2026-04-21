"""
InputContract pre-flight validation (Phase V.A.8 MVP, 2026-04-21).

Catches misconfigured YAMLs BEFORE the trainer subprocess is launched —
saving GPU-hours and surfacing actionable errors at manifest-validation
time rather than at model-initialization time inside the trainer.

Typical caught miss-config:
  * ``model.model_type=deeplob`` with ``data.feature_count=128`` (DeepLOB
    expects 40 <= F <= 98 per its InputContract).
  * ``model.model_type=deeplob`` with ``data.sequence.window_size=10``
    (DeepLOB requires T >= 20).
  * ``model.model_type=tlob`` with ``data.sequence.window_size=1``
    (TLOB requires T >= 4).
  * Any model with non-positive feature_count or window_size.
  * Unknown ``model_type`` string → WARN (observation; trainer will
    surface its own error, we just flag early).

Architectural status (MVP, Phase V.A.8):
    This module uses a HARDCODED per-model constraint table
    (``_INPUT_CONTRACTS``) synced manually from ``lobmodels.ModelRegistry``
    live-queried values (2026-04-21). Drift risk: if a model's
    ``InputContract`` is tightened/loosened in lob-models, this table
    must be updated in the same release cycle. Phase VI (task #135 /
    task #109) replaces this hardcoded table with automated consumption
    of a committed ``lob-models/src/lobmodels/_registry_snapshot.json``
    produced by ``lob-models/tools/generate_registry_snapshot.py`` +
    hatch build hook + CI drift check.

Contract surface:
    * ``validate_input_contract(model_name, feature_count, window_size,
      data_source) -> None`` — pure function. Raises ``ValueError`` on
      violation; returns None (via ``pass``) when no constraint applies.
    * ``preflight_trainer_config(trainer_config_path) -> None`` —
      convenience loader: parses trainer YAML + extracts the 3 fields +
      delegates to ``validate_input_contract``. Consumed by
      ``TrainingRunner.run`` via try/except → ``StageResult(FAILED)``.

Fail-loud per hft-rules §8: callers must handle ``ValueError`` — the
``TrainingRunner`` wrap converts to ``StageResult(FAILED)`` with a
``gate_report`` dict conforming to ``hft_contracts.gate_report.GateReportDict``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


# =============================================================================
# Hardcoded constraint table (Phase V.A.8 MVP)
# =============================================================================
#
# Synced manually from `lobmodels.ModelRegistry.list_models()` live query
# on 2026-04-21. If lob-models ships a new model OR tightens/loosens a
# constraint, update this table in the same coordinated commit series.
#
# Phase VI will replace this with auto-generated
# `lob-models/_registry_snapshot.json` consumed via `importlib.resources` —
# zero-drift, single-source-of-truth.
#
# Dict shape mirrors `lobmodels.registry.protocols.InputContract`:
#   min_features: int         (default 1 if None)
#   max_features: Optional[int] (None = no upper bound)
#   min_sequence_length: int  (default 1)
#   compatible_sources: List[str]  (['any'] = permissive)
#
# Absent models fall through to the "unknown" WARN path (no constraint
# applied; trainer surfaces its own error downstream).

_INPUT_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "deeplob": {
        "min_features": 40,
        "max_features": 98,
        "min_sequence_length": 20,
        "compatible_sources": ["any"],
    },
    "gru": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "hmhp": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "hmhp_regressor": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "logistic_lob": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "lstm": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "mlplob": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 4,
        "compatible_sources": ["any"],
    },
    "temporal_gradboost": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "temporal_ridge": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 1,
        "compatible_sources": ["any"],
    },
    "tlob": {
        "min_features": 1,
        "max_features": None,
        "min_sequence_length": 4,
        "compatible_sources": ["any"],
    },
}


# =============================================================================
# Public API
# =============================================================================


def validate_input_contract(
    model_name: str,
    feature_count: int,
    window_size: int,
    data_source: str = "mbo_lob",
) -> None:
    """Raise ``ValueError`` if a manifest's (model, features, window) tuple
    violates the model's declared ``InputContract``.

    Args:
        model_name: Lowercase registry name (e.g. ``"tlob"``, ``"deeplob"``).
            Unknown models log a WARNING and return without raising —
            lob-models' trainer loader will surface its own error
            downstream.
        feature_count: Manifest's ``data.feature_count``. Must be positive.
        window_size: Manifest's ``data.sequence.window_size`` (or trainer
            YAML's equivalent key). Must be positive.
        data_source: ``"mbo_lob"`` (default) or ``"off_exchange"`` —
            reserved for future InputContract.compatible_sources checks;
            currently unused because all registered models accept ``"any"``.

    Raises:
        ValueError: on any of (1) non-positive feature_count or
            window_size; (2) feature_count outside
            ``[min_features, max_features]``; (3) window_size <
            ``min_sequence_length``; (4) data_source not in
            ``compatible_sources`` (when compatible_sources != ["any"]).
    """
    if feature_count <= 0:
        raise ValueError(
            f"Input contract violation: feature_count={feature_count} must be "
            f"positive. Check data.feature_count in the trainer YAML."
        )
    if window_size <= 0:
        raise ValueError(
            f"Input contract violation: window_size={window_size} must be "
            f"positive. Check data.sequence.window_size (or train.window_size) "
            f"in the trainer YAML."
        )

    contract = _INPUT_CONTRACTS.get(model_name)
    if contract is None:
        # Unknown model — WARN-but-proceed path (observation tier).
        # Trainer-side model factory will surface its own KeyError /
        # registry-lookup failure if the model_type truly doesn't exist.
        logger.warning(
            "InputContract pre-flight: unknown model_type %r. Skipping "
            "constraint check; trainer subprocess will surface the error "
            "at initialization time. Add the model to _INPUT_CONTRACTS in "
            "hft_ops.stages.contract_preflight after verifying its "
            "live InputContract in lobmodels.ModelRegistry.",
            model_name,
        )
        return

    min_f = contract["min_features"]
    max_f = contract["max_features"]
    min_seq = contract["min_sequence_length"]
    compat_sources = contract["compatible_sources"]

    if feature_count < min_f:
        raise ValueError(
            f"Input contract violation: model {model_name!r} requires "
            f"min_features={min_f}, but YAML specifies feature_count="
            f"{feature_count}. Either switch to a model with smaller "
            f"min_features or extract a wider feature layout."
        )
    if max_f is not None and feature_count > max_f:
        raise ValueError(
            f"Input contract violation: model {model_name!r} requires "
            f"max_features={max_f}, but YAML specifies feature_count="
            f"{feature_count}. Either switch to a model with larger "
            f"max_features or narrow the feature layout (e.g., use a "
            f"FeatureSet to select a subset)."
        )
    if window_size < min_seq:
        raise ValueError(
            f"Input contract violation: model {model_name!r} requires "
            f"min_sequence_length={min_seq}, but YAML specifies "
            f"window_size={window_size}. Either switch to a model with "
            f"smaller min_sequence_length or increase the sequence window."
        )
    if compat_sources != ["any"] and data_source not in compat_sources:
        raise ValueError(
            f"Input contract violation: model {model_name!r} accepts "
            f"data_sources={compat_sources}, but YAML specifies "
            f"data_source={data_source!r}."
        )


def preflight_trainer_config(trainer_config_path: Path) -> None:
    """Load a resolved trainer YAML + extract (model, features, window) +
    delegate to ``validate_input_contract``.

    Convenience wrapper for ``TrainingRunner.run`` — keeps the YAML-parsing
    + field-plumbing out of the stage runner so the stage-runner wrap stays
    focused on the try/except → ``StageResult(FAILED)`` contract.

    Extracts:
        * ``model.model_type`` (required; raises if missing).
        * ``data.feature_count`` (required; raises if missing).
        * ``data.sequence.window_size`` (optional, falls back to default
          100 — matches DataConfig default).
        * ``data.data_source`` (optional, defaults to ``"mbo_lob"``).

    Args:
        trainer_config_path: Absolute path to the materialized trainer YAML
            (post ``_base:`` resolution + override application — i.e.,
            what ``TrainingRunner.run`` calls ``effective_config``).

    Raises:
        ValueError: forwarded from ``validate_input_contract`` OR raised
            directly when required YAML keys are missing (structural
            violation — trainer would crash too, but we catch here with
            a clearer message).
    """
    if not trainer_config_path.exists():
        raise ValueError(
            f"Input contract pre-flight: trainer config not found at "
            f"{trainer_config_path!r}."
        )

    with open(trainer_config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Input contract pre-flight: trainer config at {trainer_config_path!r} "
            f"is not a mapping (got {type(cfg).__name__})."
        )

    model_block = cfg.get("model") or {}
    data_block = cfg.get("data") or {}

    model_name = model_block.get("model_type")
    if not model_name or not isinstance(model_name, str):
        raise ValueError(
            f"Input contract pre-flight: trainer config missing "
            f"model.model_type (got {model_name!r})."
        )

    feature_count = data_block.get("feature_count")
    if feature_count is None or not isinstance(feature_count, int):
        raise ValueError(
            f"Input contract pre-flight: trainer config missing or "
            f"non-integer data.feature_count (got {feature_count!r})."
        )

    sequence_block = data_block.get("sequence") or {}
    window_size = sequence_block.get("window_size")
    if window_size is None:
        # Fall back to DataConfig default. Matches the trainer's own
        # default-fill behavior (dataconfig.DataConfig.sequence.window_size
        # default = 100).
        window_size = 100

    data_source = data_block.get("data_source") or "mbo_lob"

    validate_input_contract(
        model_name=model_name.lower(),
        feature_count=int(feature_count),
        window_size=int(window_size),
        data_source=str(data_source),
    )
