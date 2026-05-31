"""Single source of truth for pipeline-stage structural introspection.

Derives the canonical stage-name set, the stage-name → stage-dataclass map, and
the known-key set per stage DIRECTLY from the :class:`~hft_ops.manifest.schema.Stages`
container dataclass — so consumers NEVER hand-maintain a parallel copy of
"which stages exist" or "which keys a stage accepts". Every drift between a
manifest and the schema is then caught at a boundary that DERIVES from this
module (loader unknown-stage RAISE, unknown-key WARN, sweep-axis routing,
validator enabled-stages), instead of each consumer carrying its own list that
silently rots out of sync (the "hand-maintained mirror" class — see
VALIDATION_AND_DESIGN_2026_05_30.md §3 A3 / §7 / §12).

Module-internal (underscore-prefixed): do NOT import across module boundaries.
Torch-free + import-light (only ``dataclasses`` / ``functools`` / ``typing`` +
``manifest.schema``), preserving the hft-ops torch-free invariant (root
CLAUDE.md §Module Technical Map). Locked by a torch-free AST + sys.modules
sentinel test (``tests/test_field_introspection.py``).

**Type-resolution contract — the load-bearing detail.** Stage dataclass types
are resolved via :func:`typing.get_type_hints`, NEVER
:attr:`dataclasses.Field.type`. ``schema.py`` declares ``from __future__ import
annotations``, so a ``Field``'s ``.type`` is the raw *string* annotation (e.g.
``"ExtractionStage"``). An ``isinstance(field.type, type)`` check is therefore
always ``False`` and silently skips every field — the exact vacuous-test trap
this module exists to prevent (``test_manifest_schema.py::TestStagesLoaderParity``
was false-green for precisely that reason). ``get_type_hints`` evaluates the
string annotations against ``schema``'s module namespace and returns the real
classes.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 1 (2026-05-31).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, get_type_hints

from hft_ops.manifest.schema import Stages


@lru_cache(maxsize=1)
def _stage_type_map() -> Dict[str, type]:
    """Resolved ``{stage_name: stage_dataclass}`` for every field on ``Stages``.

    Resolved once via :func:`typing.get_type_hints` (NOT ``Field.type`` — see
    the module docstring). Each resolved type is asserted to be a dataclass so
    that a future *non*-dataclass field on ``Stages`` fails LOUD here rather
    than silently corrupting a consumer's known-key derivation.

    Returns:
        Insertion-ordered (= schema field order) mapping. Treat as read-only;
        the result is cached and shared.

    Raises:
        TypeError: if any ``Stages`` field resolves to a non-dataclass type.
    """
    hints = get_type_hints(Stages)
    resolved: Dict[str, type] = {}
    for f in fields(Stages):
        stage_type = hints[f.name]
        if not is_dataclass(stage_type):
            raise TypeError(
                f"Stages.{f.name} resolves to {stage_type!r}, which is not a "
                f"dataclass. _field_introspection derives each stage's known "
                f"keys from its dataclass fields; a non-dataclass stage field "
                f"breaks that contract. If this is intentional, update "
                f"_field_introspection to handle the new shape explicitly."
            )
        resolved[f.name] = stage_type
    return resolved


@lru_cache(maxsize=1)
def stage_names() -> FrozenSet[str]:
    """The canonical set of pipeline-stage names (the fields of ``Stages``)."""
    return frozenset(_stage_type_map())


def stage_dataclass(stage_name: str) -> type:
    """Return the stage-config dataclass for ``stage_name``.

    Args:
        stage_name: One of :func:`stage_names`.

    Returns:
        The dataclass type (e.g. ``TrainingStage`` for ``"training"``).

    Raises:
        KeyError: if ``stage_name`` is not a known stage. Callers that expect a
            miss should pre-check membership with :func:`stage_names`; this
            fail-loud guards programming errors / typos at the boundary.
    """
    try:
        return _stage_type_map()[stage_name]
    except KeyError:
        raise KeyError(
            f"unknown stage {stage_name!r}; known stages: {sorted(stage_names())}"
        ) from None


@lru_cache(maxsize=None)
def known_keys_for_stage(stage_name: str) -> FrozenSet[str]:
    """The full set of dataclass field names a stage accepts.

    This is the FULL field set (including "special" fields such as
    ``training.overrides``). Consumers that need a narrower projection (e.g.
    sweep-axis routing, which excludes ``overrides``) subtract from this set at
    their own call site — there is exactly one source here.

    Args:
        stage_name: One of :func:`stage_names`.

    Raises:
        KeyError: via :func:`stage_dataclass` if ``stage_name`` is unknown.
    """
    return frozenset(f.name for f in fields(stage_dataclass(stage_name)))
