"""Phase R-17 F3: SSoT for orchestrator state-mutation discipline.

This module provides the canonical pattern for ALL orchestrator override sites
that previously used direct `overrides[X] = Y` mutation. Closes the bug class
underlying #PY-128 (training.py:393 silent override of trainer_config.data.horizon_idx)
+ NEW-BUG-9/10 (training.py:399 sister site) + #PY-131 / NEW-BUG-11/16
(sweep axis + override typos silently land).

**Class A SSoT primitive** (per CLAUDE.md "Shared coordination surface" list).
Cross-module blast radius: any new orchestrator state-mutation site MUST use
``apply_override_loud()`` instead of direct dict mutation. See Change-Coordination
Checklist row "Add a new orchestrator state-mutation site" in root CLAUDE.md.

Design discipline:
- **WARN on clobber** (default): when ``user_set_check=False``, override clobber of
  existing key emits ``UserWarning`` with source attribution — preserves Phase R-17
  back-compat for legacy sites whose existing override pattern is grandfather-allowed.
- **RAISE on user-set conflict** (default ``user_set_check=True``): when the existing
  value was explicitly set by user (not just-defaulted), conflict raises
  ``OverrideConflictError`` per hft-rules §5 fail-fast.
- **Typo detection** (when ``known_keys`` provided): unknown key raises
  ``UnknownOverrideKeyError`` with difflib close-match suggestion.
- **Equality semantics**: Python ``==`` (numeric coercion accepted, e.g.,
  ``1 == 1.0`` no-conflict). For bit-exact-type checks, callers should compare
  pre-call.

Phase R-17 migration discipline (per design doc §2 F4):
- Initial migration uses ``user_set_check=False`` (WARN-only) for back-compat.
- Phase R-18 promotes per-site to ``user_set_check=True`` after audit confirms
  no production manifest relies on silent override.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)


class OverrideConflictError(ValueError):
    """Raised when a runtime-derived value conflicts with a user-set value.

    Phase R-17 fail-fast per hft-rules §5: when the orchestrator computes a
    value (e.g., resolved horizon_idx from manifest export metadata) that
    differs from a value the user explicitly set in their manifest's
    ``overrides:`` block, this signals a configuration ambiguity that the
    user must resolve — silent override would corrupt experiment provenance.
    """


class UnknownOverrideKeyError(KeyError):
    """Raised when an override key doesn't match any known schema field (typo detection).

    Phase R-17 typo-detection per hft-rules §5: prevents typo'd manifest keys
    from silently landing in trainer config dicts where they're never read.
    Closes #PY-131 sister cluster (sweep axis typos + manifest override typos).
    """


def apply_override_loud(
    target: Dict[str, Any],
    key: str,
    value: Any,
    *,
    source: str,
    known_keys: Optional[FrozenSet[str]] = None,
    user_set_check: bool = True,
) -> None:
    """Apply override with fail-loud discipline.

    Phase R-17 SSoT: replaces direct ``target[key] = value`` mutation across all
    orchestrator override sites. Provides four guarantees:

    1. **Typo detection**: when ``known_keys`` provided, unknown ``key`` raises
       ``UnknownOverrideKeyError`` with difflib close-match suggestion.
    2. **Conflict detection** (when ``user_set_check=True``): if ``target[key]``
       differs from ``value`` and was explicitly user-set, raises
       ``OverrideConflictError``.
    3. **WARN on clobber** (when ``user_set_check=False``): if ``target[key]``
       differs from ``value``, emits ``UserWarning`` with source attribution.
       Used for Phase R-17 initial migration to preserve back-compat.
    4. **Source attribution**: ``source`` argument required for traceability;
       included in all warning/error messages.

    Args:
        target: dict to mutate (may be deeply nested; dotted-key resolved via
            `_get_nested` + `_set_nested` helpers).
        key: dotted-path key (e.g., ``"data.horizon_idx"``) or bare key.
        value: value to write.
        source: human-readable description of the override source (e.g.,
            ``"training.py:393 horizon_value=10 → horizon_idx=0 via export metadata"``).
            REQUIRED for traceability; not optional.
        known_keys: optional ``FrozenSet[str]`` of allowed dotted-path keys for
            schema validation. When provided, ``key`` not in set raises
            ``UnknownOverrideKeyError``. Closes #PY-131 typo class.
        user_set_check: when ``True`` (default), conflict with existing value
            raises ``OverrideConflictError``. When ``False``, conflict emits
            ``UserWarning`` instead — used for Phase R-17 initial migration
            (back-compat WARN-only; promote per-site in Phase R-18 after audit).

    Raises:
        UnknownOverrideKeyError: ``key`` not in ``known_keys`` (typo detected).
        OverrideConflictError: when ``user_set_check=True`` AND
            ``target[key] != value`` AND existing value present (was user-set).
        ValueError: when ``key`` is empty string or contains only dots.

    Side effects:
        - Mutates ``target`` to apply the override.
        - May emit ``UserWarning`` (when ``user_set_check=False`` AND clobber).
        - DEBUG-logs every applied override with source attribution.
    """
    # Sanity check on key (empty / all-dots is operator error)
    if not key or all(part == "" for part in key.split(".")):
        raise ValueError(
            f"apply_override_loud: empty or invalid key {key!r} from {source}. "
            f"Override keys must be non-empty dotted-paths (e.g., 'data.horizon_idx')."
        )

    # Typo detection via schema validation (when known_keys provided)
    if known_keys is not None and key not in known_keys:
        raise UnknownOverrideKeyError(
            f"Unknown override key '{key}' from {source}. "
            f"Did you mean: {_suggest_close_match(key, known_keys)}? "
            f"Known keys (first 10 alphabetical): "
            f"{sorted(known_keys)[:10]}..."
        )

    # Conflict detection: read existing value via nested-dotted-path resolver
    existing = _get_nested(target, key)
    if existing is not None and existing != value:
        if user_set_check:
            raise OverrideConflictError(
                f"Override conflict at '{key}': user-set value {existing!r} "
                f"conflicts with runtime-derived value {value!r} from {source}. "
                f"Resolve by either (a) removing one of the two settings, or "
                f"(b) explicitly aligning the user value to match the runtime "
                f"resolution. Silent override would corrupt experiment provenance."
            )
        else:
            warnings.warn(
                f"Override at '{key}': clobbering existing value {existing!r} "
                f"with {value!r} from {source}. To silence this warning, pass "
                f"user_set_check=True (raises on conflict) or remove the existing "
                f"value from the manifest.",
                UserWarning,
                stacklevel=2,
            )

    # Apply override
    _set_nested(target, key, value)
    logger.debug(
        "apply_override_loud: '%s' = %r from %s "
        "(prev=%r, user_set_check=%s)",
        key, value, source, existing, user_set_check,
    )


def _get_nested(target: Dict[str, Any], dotted_key: str) -> Any:
    """Read dotted-path key from nested dict; return None if missing.

    Examples:
        >>> _get_nested({"data": {"horizon_idx": 5}}, "data.horizon_idx")
        5
        >>> _get_nested({}, "data.horizon_idx")  # returns None (missing)
        >>> _get_nested({"data": 42}, "data.horizon_idx")  # scalar at intermediate
        None
    """
    parts = dotted_key.split(".")
    cur: Any = target
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write dotted-path key into nested dict; create intermediate dicts as needed.

    When an intermediate key exists but is NOT a dict (e.g., ``target["data"] = 42``),
    raises ``ValueError`` per hft-rules §5 — overwriting a scalar with a dict
    would silently lose user-set data.

    Examples:
        >>> t = {}
        >>> _set_nested(t, "data.horizon_idx", 5)
        >>> t
        {'data': {'horizon_idx': 5}}

        >>> t = {"data": {"existing": "preserved"}}
        >>> _set_nested(t, "data.horizon_idx", 5)
        >>> t
        {'data': {'existing': 'preserved', 'horizon_idx': 5}}
    """
    parts = dotted_key.split(".")
    cur: Dict[str, Any] = target
    for part in parts[:-1]:
        if part not in cur:
            cur[part] = {}
        elif not isinstance(cur[part], dict):
            raise ValueError(
                f"_set_nested: intermediate key '{part}' in dotted-path "
                f"'{dotted_key}' is a scalar {cur[part]!r}, not a dict. "
                f"Overwriting would silently lose user-set data. "
                f"Resolve by either (a) removing the conflicting scalar, or "
                f"(b) using a different override path."
            )
        cur = cur[part]
    cur[parts[-1]] = value


def _suggest_close_match(key: str, known_keys: FrozenSet[str]) -> str:
    """difflib-based close match for typo error messages.

    Returns the closest match (cutoff=0.7) or '<no close match>' fallback.
    """
    import difflib
    matches = difflib.get_close_matches(key, list(known_keys), n=1, cutoff=0.7)
    return matches[0] if matches else "<no close match>"


# =============================================================================
# Phase R-17 F5: known prefixes + first-segment validation
# =============================================================================
#
# Locked at 2026-05-11 from Step 0 manifest audit (51 manifests / 13 dotted
# override keys + 11 sweep axes / ZERO typos detected) + SafeBaseModel registry
# (9 trainer config classes per `lobtrainer.config.base.SafeBaseModel._registry`
# verified via Step 0 introspection).
#
# Minimum-viable scope: first-segment validation only. Catches gross typos
# like `mode.X` → `model.X`. Full-key (per sub-field) validation deferred to
# Phase R-18 because:
# (a) Step 0 audit found ZERO production typos (immediate risk is dotted-prefix
#     class, not sub-key drift)
# (b) Sub-key validation requires hardcoded list per SafeBaseModel class (~80
#     fields total) which is brittle to maintain
# (c) Drift-detection test would import lobtrainer → trigger torch (breaks
#     hft-ops AST torch-free invariant per Cycle C1 regression lock)
# Phase R-18 candidate: add `_PYDANTIC_CONFIG_CLASSES` snapshot pattern (Phase
# VI architecture) to enable sub-key validation without runtime torch import.

KNOWN_TRAINER_PREFIXES: FrozenSet[str] = frozenset({
    # Top-level trainer config sections (per ExperimentConfig SafeBaseModel
    # fields: data, model, train, cv — see lobtrainer.config.schema)
    "data", "model", "train", "cv",
})


def validate_trainer_override_prefixes(
    overrides: Dict[str, Any],
    *,
    source: str,
    known_prefixes: FrozenSet[str] = KNOWN_TRAINER_PREFIXES,
) -> None:
    """Validate top-level prefixes of dotted override keys against known set.

    Phase R-17 F5: closes #PY-131 / NEW-BUG-11/16 typo detection class.
    Catches gross typos at the prefix level (e.g., `mode.X` → `model.X`,
    `tra1n.X` → `train.X`, `dta.X` → `data.X`).

    Bare keys (no dot) are NOT validated by this function — they flow through
    sweep.py's separate `_MANIFEST_LEVEL_TRAINING_FIELDS` validation path.

    Args:
        overrides: dict of dotted-key → value (trainer config overrides).
        source: human-readable source for error messages (e.g.,
            "sweep.py:445 dotted-prefix passthrough").
        known_prefixes: optional override set; defaults to ``KNOWN_TRAINER_PREFIXES``.

    Raises:
        UnknownOverrideKeyError: when any dotted key has a first segment not
            in ``known_prefixes``. Error message includes close-match suggestion.
    """
    for key in overrides:
        if "." not in key:
            continue  # bare keys validated separately
        first_seg = key.split(".", 1)[0]
        if first_seg not in known_prefixes:
            raise UnknownOverrideKeyError(
                f"Unknown override prefix '{first_seg}' in key '{key}' from {source}. "
                f"Did you mean: {_suggest_close_match(first_seg, known_prefixes)}? "
                f"Known top-level prefixes: {sorted(known_prefixes)}"
            )
