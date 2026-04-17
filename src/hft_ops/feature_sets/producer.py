"""
FeatureSet producer: orchestrates ``hft-feature-evaluator`` → FeatureSet.

The producer is the only module in ``hft_ops.feature_sets`` that depends
on ``hft-feature-evaluator`` being installed. All evaluator imports are
LAZY (inside ``produce_feature_set``), so the rest of the registry
(writer, reader, hashing, schema) works in any hft-ops venv regardless
of whether the evaluator is pip-installed.

Workflow (executed inside ``produce_feature_set``):

1. Load ``EvaluationConfig`` from the evaluator config YAML.
2. Load ``SelectionCriteria`` from the criteria YAML (Phase 4 Batch 4a).
3. Instantiate ``EvaluationPipeline`` and call ``run_v2()``.
   - Produces ``dict[feature_name -> FeatureProfile]``
   - Populates ``pipeline.last_profile_hash`` (Phase 4 Batch 4a).
4. Apply ``select_features(profiles, criteria)`` → sorted list of names.
5. Map names → indices via ``profiles[name].feature_index``.
6. Compute provenance hashes: evaluator-config file hash + data-export
   directory manifest hash.
7. Assemble a ``FeatureSet`` via ``FeatureSet.build`` (auto-computes
   content_hash from PRODUCT fields).

Writer is NOT invoked here — the caller (typically the CLI) decides
where to persist and whether to use ``force``. This separation allows
the same producer to be reused from non-CLI contexts (e.g., notebooks,
sweep orchestration) without duplicating the assembly code.

Error handling:
- Evaluator not installed → raise ``EvaluatorNotInstalled`` with a
  clear install hint (subclasses ``ImportError`` for dual ergonomics).
- Evaluator run fails → the original evaluator exception propagates.
- Criteria YAML malformed → ``ValueError`` from ``SelectionCriteria.from_yaml``.
- No features selected → raise ``NoFeaturesSelectedError`` rather than
  producing an empty FeatureSet (the hashing layer would reject an
  empty index list anyway; we reject earlier with a clearer message).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from hft_ops.feature_sets.schema import (
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetProducedBy,
)
from hft_ops.paths import PipelinePaths
from hft_ops.provenance.lineage import hash_directory_manifest, hash_file


class EvaluatorNotInstalled(ImportError):
    """Raised when the producer is invoked but ``hft-feature-evaluator``
    is not installed in the active venv.

    Subclasses ``ImportError`` for compatibility with callers that
    catch the generic import family. The custom message provides the
    install command explicitly.
    """


class NoFeaturesSelectedError(ValueError):
    """Raised when ``select_features(profiles, criteria)`` returns an
    empty list — which would result in an invalid FeatureSet (the
    hashing layer rejects empty index lists)."""


def produce_feature_set(
    *,
    evaluator_config_path: Path,
    criteria_yaml_path: Path,
    name: str,
    applies_to_assets: Sequence[str],
    applies_to_horizons: Sequence[int],
    pipeline_paths: PipelinePaths,
    description: str = "",
    notes: str = "",
    created_by: str = "",
) -> FeatureSet:
    """Run the evaluator and assemble a FeatureSet from its output.

    This function does NOT persist the FeatureSet — the caller should
    pass the result to ``write_feature_set`` if disk persistence is
    needed. Returning the in-memory FeatureSet enables notebook /
    sweep orchestration workflows that want to inspect before writing.

    Args:
        evaluator_config_path: Path to the evaluator YAML config.
            Consumed by ``EvaluationConfig.from_yaml``.
        criteria_yaml_path: Path to a YAML file describing the
            ``SelectionCriteria`` (either flat or wrapped under a
            ``criteria:`` key — Phase 4 Batch 4a).
        name: Identifier for the FeatureSet. Typically ends with ``_vN``
            for human-incremented versioning (e.g., ``"momentum_hft_v1"``).
            This becomes the JSON filename at the writer boundary.
        applies_to_assets: Tuple of ticker symbols this set was built
            for (metadata only; does not affect content hash).
        applies_to_horizons: Tuple of label horizons (same: metadata).
        pipeline_paths: Already-resolved ``PipelinePaths`` used to
            anchor relative paths for provenance (``config_path``,
            ``data_export``).
        description: Free-text description. Metadata only.
        notes: Free-text operator notes. Metadata only.
        created_by: User/agent identifier. Metadata only.

    Returns:
        The assembled ``FeatureSet`` (not yet written to disk).

    Raises:
        EvaluatorNotInstalled: If ``hft-feature-evaluator`` is missing
            from the active venv.
        FileNotFoundError: If either config path does not exist.
        NoFeaturesSelectedError: If the criteria match no features.
        ValueError: Propagated from config parsers on malformed YAML.
        Exception: Propagated from ``EvaluationPipeline.run_v2()`` on
            evaluation failures (numerical errors, missing data, etc.).
    """
    # ------------------------------------------------------------------
    # Lazy import — the only module-scope evaluator dependency.
    # ------------------------------------------------------------------
    try:
        from hft_evaluator.config import EvaluationConfig
        from hft_evaluator.criteria import SelectionCriteria, select_features
        from hft_evaluator.pipeline import EvaluationPipeline
    except ImportError as exc:
        raise EvaluatorNotInstalled(
            "hft-feature-evaluator is not installed in this venv. "
            "Install with: "
            "`pip install -e /Users/knight/code_local/HFT-pipeline-v2/hft-feature-evaluator` "
            "(or adjust the path for your checkout). "
            f"Underlying ImportError: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 1 + 2. Parse configs.
    # ------------------------------------------------------------------
    evaluator_config_path = Path(evaluator_config_path)
    criteria_yaml_path = Path(criteria_yaml_path)
    if not evaluator_config_path.exists():
        raise FileNotFoundError(
            f"Evaluator config not found: {evaluator_config_path}"
        )
    if not criteria_yaml_path.exists():
        raise FileNotFoundError(
            f"Criteria YAML not found: {criteria_yaml_path}"
        )

    evaluator_config = EvaluationConfig.from_yaml(str(evaluator_config_path))
    criteria = SelectionCriteria.from_yaml(criteria_yaml_path)

    # ------------------------------------------------------------------
    # 3. Run the evaluator.
    # ------------------------------------------------------------------
    pipeline = EvaluationPipeline(evaluator_config)
    profiles = pipeline.run_v2()
    source_profile_hash = pipeline.last_profile_hash
    if source_profile_hash is None:
        # Should never happen — run_v2 populates the hash on success.
        # Defensive guard so a producer bug never produces an unattributed
        # FeatureSet.
        raise RuntimeError(
            "EvaluationPipeline.run_v2 returned without populating "
            "last_profile_hash. This indicates a bug in the evaluator "
            "version; verify hft-feature-evaluator >= Phase 4 Batch 4a."
        )

    # ------------------------------------------------------------------
    # 4 + 5. Apply criteria + map names → indices.
    # ------------------------------------------------------------------
    selected_names = select_features(profiles, criteria)
    if not selected_names:
        raise NoFeaturesSelectedError(
            f"Criteria '{criteria.name}' matched zero features in the "
            f"profiles produced by {evaluator_config_path.name}. "
            f"Loosen the criteria or verify the evaluator ran correctly."
        )

    # select_features returns names sorted alphabetically; we further
    # sort-by-index for canonical index ordering (the hashing layer
    # sorts regardless, but matching order here makes feature_names
    # parallel-indexed with feature_indices on the FeatureSet).
    name_index_pairs = sorted(
        ((n, profiles[n].feature_index) for n in selected_names),
        key=lambda p: p[1],
    )
    indices = [idx for _, idx in name_index_pairs]
    names_in_index_order = [n for n, _ in name_index_pairs]

    # ------------------------------------------------------------------
    # 6. Compute provenance hashes.
    # ------------------------------------------------------------------
    try:
        config_hash = hash_file(evaluator_config_path)
    except OSError:
        # Extremely unlikely (we just verified existence) but if the
        # file was deleted between check and hash, record as empty so
        # the FeatureSet is still constructible. Logged at INFO level
        # would be ideal; keeping this minimal for now.
        config_hash = ""

    export_dir = Path(evaluator_config.export_dir)
    if not export_dir.is_absolute():
        # Resolve relative to the pipeline root (matches evaluator
        # convention — configs reference `data/exports/...`).
        export_dir = (pipeline_paths.pipeline_root / export_dir).resolve()

    try:
        data_dir_hash = hash_directory_manifest(export_dir)
    except OSError:
        data_dir_hash = ""

    # Relativize the paths for the JSON manifest (stable across checkouts).
    try:
        config_path_rel = str(
            evaluator_config_path.resolve().relative_to(pipeline_paths.pipeline_root)
        )
    except ValueError:
        config_path_rel = str(evaluator_config_path.resolve())

    try:
        data_export_rel = str(export_dir.relative_to(pipeline_paths.pipeline_root))
    except ValueError:
        data_export_rel = str(export_dir)

    produced_by = FeatureSetProducedBy(
        tool="hft-feature-evaluator",
        tool_version=_evaluator_version(),
        config_path=config_path_rel,
        config_hash=config_hash,
        source_profile_hash=source_profile_hash,
        data_export=data_export_rel,
        data_dir_hash=data_dir_hash,
    )

    # ------------------------------------------------------------------
    # 7. Assemble and return.
    # ------------------------------------------------------------------
    from dataclasses import asdict
    criteria_dict = asdict(criteria)

    return FeatureSet.build(
        name=name,
        feature_indices=indices,
        feature_names=names_in_index_order,
        source_feature_count=_resolve_source_feature_count(pipeline),
        contract_version=_resolve_contract_version(pipeline),
        applies_to=FeatureSetAppliesTo(
            assets=tuple(applies_to_assets),
            horizons=tuple(applies_to_horizons),
        ),
        produced_by=produced_by,
        criteria=criteria_dict,
        criteria_schema_version=criteria.criteria_schema_version,
        description=description,
        notes=notes,
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by=created_by,
    )


def _evaluator_version() -> str:
    """Return the installed hft-feature-evaluator version, or ``"unknown"``.

    Tries ``importlib.metadata`` first (PEP 566), falls back to
    ``hft_evaluator.__version__`` if the package defines it. Never
    raises — producer flow must not fail on a version-query hiccup.
    """
    try:
        from importlib.metadata import version

        return version("hft-feature-evaluator")
    except Exception:
        pass
    try:
        import hft_evaluator

        return getattr(hft_evaluator, "__version__", "unknown")
    except Exception:
        return "unknown"


def _resolve_contract_version(pipeline: Any) -> str:
    """Extract the pipeline contract version used by the evaluator run.

    Different evaluator versions expose this through different fields;
    try the known candidates in priority order and fall back to
    ``"unknown"`` if none apply (the FeatureSet is still constructible,
    though consumers that verify contract_version at load time will
    refuse it).
    """
    schema = getattr(pipeline.loader, "schema", None)
    if schema is None:
        return "unknown"
    for attr in ("contract_version", "schema_version"):
        value = getattr(schema, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _resolve_source_feature_count(pipeline: Any) -> int:
    """Extract the source feature-axis width from the evaluator pipeline.

    Mirrors ``_resolve_contract_version`` — tries known schema fields.
    Raises RuntimeError if none apply, because the hashing layer
    REQUIRES a positive source_feature_count and we'd rather fail fast
    at production time than write a FeatureSet with a bogus value.
    """
    schema = getattr(pipeline.loader, "schema", None)
    if schema is None:
        raise RuntimeError(
            "Cannot resolve source_feature_count — pipeline.loader.schema "
            "is None. This indicates a broken evaluator pipeline instance."
        )
    for attr in ("feature_count", "n_features", "source_feature_count"):
        value = getattr(schema, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError(
        f"Cannot resolve source_feature_count from evaluator schema "
        f"(tried: feature_count, n_features, source_feature_count). "
        f"Schema type: {type(schema).__name__}"
    )
