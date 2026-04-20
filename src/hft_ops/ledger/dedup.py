"""
Experiment fingerprinting and deduplication.

The fingerprint is a deterministic SHA-256 hash computed from the resolved
experiment configuration + data directory hash + contract version. Two
manifests that resolve to identical effective configs produce the same
fingerprint, regardless of YAML formatting, file paths, or timestamps.

This prevents accidentally running identical experiments while allowing
intentional re-runs (the user can override the dedup check).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.paths import PipelinePaths
from hft_ops.provenance.lineage import hash_directory_manifest


class FingerprintNormalizationError(ValueError):
    """Raised when feature-selection normalization fails during fingerprint
    computation (Phase 4 Batch 4c.3).

    The fingerprint requires deterministic `feature_indices` — if the resolver
    cannot produce them (missing registry file, malformed JSON, unimportable
    preset module, tampered content_hash), we HARD-FAIL rather than silently
    fingerprinting an un-normalized config. Soft-failing would create the same
    class of ledger-conflation bug as Phase 3 §3.3b (distinct configs that
    hash identically because a shared input was skipped).
    """

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


def _load_config_as_dict(path: Path) -> Dict[str, Any]:
    """Load a TOML or YAML config file as a dict.

    Note: this raw loader is used for NON-trainer configs (TOML extractor
    configs, JSON metrics files). For trainer YAMLs, use
    ``_load_trainer_config_resolved`` instead — the fingerprint must reflect
    the RESOLVED effective dict after ``_base:`` inheritance is expanded,
    otherwise base mutations silently leave dependent fingerprints unchanged
    (CRITICAL ledger-conflation bug; plan §3.3b).
    """
    if not path.exists():
        return {}

    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            with open(path, "rb") as f:
                return tomllib.load(f)
        elif suffix in (".yaml", ".yml"):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        elif suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


# Cache the dynamically loaded trainer merge.py module so we don't re-load
# it on every fingerprint computation (may be called O(N_experiments) per sweep).
_TRAINER_MERGE_MODULE_CACHE: Optional[Any] = None


def _load_trainer_merge_module(paths: "PipelinePaths") -> Optional[Any]:
    """Load trainer ``merge.py`` via ``spec_from_file_location`` (cached).

    We deliberately do NOT ``import lobtrainer.config.merge`` because the
    trainer's top-level ``__init__.py`` eagerly imports torch/numpy and many
    other heavy deps. hft-ops must stay installable without torch.

    This helper loads ``<trainer_dir>/src/lobtrainer/config/merge.py`` directly
    by file path, bypassing the package ``__init__.py``. Since ``merge.py``
    only depends on ``yaml`` and stdlib (verified), this works in a minimal
    hft-ops env.

    Returns:
        The loaded module, or None if the trainer path doesn't exist / the
        file can't be parsed. In the None case, callers fall back to raw
        YAML loading (fingerprint becomes non-base-aware in that env — this
        should only happen in degraded CI environments).
    """
    global _TRAINER_MERGE_MODULE_CACHE
    if _TRAINER_MERGE_MODULE_CACHE is not None:
        return _TRAINER_MERGE_MODULE_CACHE

    merge_path = (
        paths.trainer_dir / "src" / "lobtrainer" / "config" / "merge.py"
    )
    if not merge_path.exists():
        return None

    try:
        import importlib.util  # stdlib
        spec = importlib.util.spec_from_file_location(
            "_hft_ops_trainer_merge", merge_path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None

    _TRAINER_MERGE_MODULE_CACHE = module
    return module


# Phase 4 Batch 4c.3 (2026-04-16): load trainer's feature_presets.py module
# to normalize `feature_preset:` → `feature_indices` in the fingerprint.
#
# Unlike merge.py (pure stdlib + yaml), feature_presets.py has package-level
# imports (`from lobtrainer.constants.feature_index import ...`) that in turn
# import from `hft_contracts`. Plain `spec_from_file_location` is insufficient
# because those absolute imports must resolve via `sys.path`. Empirical check
# on 2026-04-16: `lobtrainer/__init__.py` is docstring-only (no torch), and
# `lobtrainer/constants/__init__.py` is equally lightweight — so adding
# `lob-model-trainer/src` to sys.path is safe (does not drag in torch).
# `hft_contracts` is already a hft-ops dependency, so the final
# `from hft_contracts import ...` inside `feature_index.py` resolves cleanly.
_TRAINER_FEATURE_PRESETS_MODULE_CACHE: Optional[Any] = None


def _load_trainer_feature_presets_module(paths: "PipelinePaths") -> Any:
    """Load trainer ``feature_presets.py`` (cached; bypasses torch-pulling __init__).

    The naive `spec_from_file_location` approach fails because
    `feature_presets.py` contains::

        from lobtrainer.constants.feature_index import FeatureIndex, ExperimentalFeatureIndex

    which triggers Python to load `lobtrainer/__init__.py` → `lobtrainer.data
    .__init__` → `dataset.py` → `import torch`. In the bare hft-ops venv
    (torch-free by design), this crashes.

    Fix: stub the `lobtrainer.constants.feature_index` chain in `sys.modules`
    with a proxy module that re-exports `FeatureIndex` and
    `ExperimentalFeatureIndex` from the SSoT `hft_contracts` package (which
    IS a hft-ops dependency and is what the trainer's `feature_index.py`
    itself re-exports). Also stub the parent packages (`lobtrainer`,
    `lobtrainer.constants`) to bypass their `__init__.py` execution.

    This is surgical: only the specific import path feature_presets.py uses
    is stubbed. If a future feature_presets.py adds a new cross-module
    import that this stub doesn't cover, the load fails with a clear error
    rather than silently behaving differently.

    Returns:
        The loaded module — guaranteed to expose ``FEATURE_PRESETS: Dict[str,
        Tuple[int, ...]]``.

    Raises:
        FingerprintNormalizationError: If the trainer directory, the
        presets file, or any of the transitive imports cannot be resolved.
        HARD-fail (see R11 rationale): silently skipping preset resolution
        would let two equivalent configs fingerprint differently.
    """
    global _TRAINER_FEATURE_PRESETS_MODULE_CACHE
    if _TRAINER_FEATURE_PRESETS_MODULE_CACHE is not None:
        return _TRAINER_FEATURE_PRESETS_MODULE_CACHE

    presets_path = (
        paths.trainer_dir
        / "src"
        / "lobtrainer"
        / "constants"
        / "feature_presets.py"
    )
    if not presets_path.exists():
        raise FingerprintNormalizationError(
            f"Trainer feature_presets.py not found at {presets_path}. "
            f"Fingerprint cannot normalize `feature_preset:` in a config "
            f"without this file. Either (a) migrate the config to "
            f"`data.feature_set: <name>_v1` (Phase 4 FeatureSet registry), "
            f"or (b) run hft-ops in a checkout that includes the trainer."
        )

    # Stub the lobtrainer.constants.feature_index chain BEFORE loading
    # feature_presets.py, so the internal `from lobtrainer.constants
    # .feature_index import ...` resolves to our stub (re-exporting from
    # hft_contracts SSoT) instead of executing the heavy lobtrainer
    # package __init__.
    import types
    try:
        from hft_contracts import FeatureIndex, ExperimentalFeatureIndex
    except ImportError as exc:
        raise FingerprintNormalizationError(
            f"hft_contracts must provide FeatureIndex + ExperimentalFeatureIndex "
            f"for the trainer feature_presets.py stub. Original error: {exc!r}"
        ) from exc

    # Only install stubs if real modules aren't already loaded (idempotent; if
    # the user's env DOES have lobtrainer installed with torch, respect it).
    _stubs_installed: list[str] = []
    if "lobtrainer.constants.feature_index" not in sys.modules:
        stub_fi = types.ModuleType("lobtrainer.constants.feature_index")
        stub_fi.FeatureIndex = FeatureIndex
        stub_fi.ExperimentalFeatureIndex = ExperimentalFeatureIndex
        sys.modules["lobtrainer.constants.feature_index"] = stub_fi
        _stubs_installed.append("lobtrainer.constants.feature_index")

        if "lobtrainer.constants" not in sys.modules:
            sys.modules["lobtrainer.constants"] = types.ModuleType("lobtrainer.constants")
            _stubs_installed.append("lobtrainer.constants")

        if "lobtrainer" not in sys.modules:
            sys.modules["lobtrainer"] = types.ModuleType("lobtrainer")
            _stubs_installed.append("lobtrainer")

    # Phase 6 6A.4 (2026-04-17): ALWAYS roll back stubs via try/finally
    # (previously rolled back only on exception — success path left stubs
    # polluting sys.modules). Pre-existing real-package entries survive because
    # `_stubs_installed` only contains names we SET here. After exec_module
    # completes, the module object retains its own references to the imports it
    # used; removing the sys.modules entries does NOT destroy them.
    try:
        import importlib.util  # stdlib
        spec = importlib.util.spec_from_file_location(
            "_hft_ops_trainer_feature_presets", presets_path
        )
        if spec is None or spec.loader is None:
            raise FingerprintNormalizationError(
                f"Could not create importlib spec for {presets_path}. "
                f"This usually indicates the file is corrupt or a Python "
                f"version mismatch."
            )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise FingerprintNormalizationError(
                f"Failed to load trainer feature_presets.py from {presets_path}: "
                f"{exc!r}. hft-ops fingerprint requires this module to normalize "
                f"`feature_preset:` configs. Confirm `hft_contracts` provides "
                f"FeatureIndex + ExperimentalFeatureIndex symbols."
            ) from exc

        # Verify the expected surface exists (defensive — catch silent API drift).
        if not hasattr(module, "FEATURE_PRESETS") or not isinstance(
            module.FEATURE_PRESETS, dict
        ):
            raise FingerprintNormalizationError(
                f"Loaded feature_presets.py does not expose `FEATURE_PRESETS: "
                f"Dict[str, Tuple[int, ...]]`. Module API has drifted — update "
                f"hft-ops dedup.py normalization to match."
            )
    finally:
        # Roll back stubs on BOTH success and failure paths (6A.4 fix).
        # Pre-existing real-package entries (when a user has lobtrainer + torch
        # installed) were NEVER added to _stubs_installed, so they survive.
        for name in _stubs_installed:
            sys.modules.pop(name, None)

    _TRAINER_FEATURE_PRESETS_MODULE_CACHE = module
    return module


# Enhancement B (user-approved 2026-04-16): LRU-cache the
# FeatureSetRegistry.get() → feature_indices resolution. At 150-point sweeps
# this saves 150× disk read + JSON parse + SHA-256 verify (roughly 1.5-15s
# per sweep). Keyed on (registry_dir_str, name) — content_hash is NOT in the
# key because computing it defeats the cache purpose; cache invalidates
# naturally across process lifetime (sweep grids run in one process). For
# test isolation, the cache is clearable via `_cached_resolve_feature_set_indices.cache_clear()`.
@functools.lru_cache(maxsize=256)
def _cached_resolve_feature_set_indices(
    registry_dir_str: str,
    feature_set_name: str,
) -> Tuple[int, ...]:
    """Process-local cached FeatureSetRegistry.get(name) → sorted indices tuple.

    Cache key deliberately excludes content_hash. The resolver runs
    `FeatureSet.verify_integrity()` on load (raises on tamper), so cached
    results reflect the on-disk content at first access. If a FeatureSet is
    REGENERATED during a single process lifetime (unusual — sweep runs are
    short), call `cache_clear()` before recomputing fingerprints.
    """
    # Local import to avoid circular dep (feature_sets module imports from
    # canonical_hash which hft-ops installs at package level).
    from hft_ops.feature_sets.registry import FeatureSetRegistry

    registry = FeatureSetRegistry(Path(registry_dir_str))
    fs = registry.get(feature_set_name, verify=True)
    # Normalize via sorted+deduped as a final belt-and-suspenders — the
    # FeatureSet schema already enforces this invariant via `FeatureSet.build`
    # but double-sorting is idempotent and cheap.
    return tuple(sorted(set(int(i) for i in fs.feature_indices)))


def _normalize_feature_selection_for_fingerprint(
    effective_train: Dict[str, Any],
    paths: "PipelinePaths",
) -> None:
    """Resolve `feature_set` / `feature_preset` / `feature_indices` → canonical
    `feature_indices` tuple IN PLACE (Phase 4 Batch 4c.3).

    Mutates ``effective_train["data"]`` so that two logically-equivalent
    configs (one with `feature_set: X`, another with the resolved
    `feature_indices: [sorted+deduped equivalent]`) produce byte-equal
    fingerprints. Resolution precedence matches DataConfig mutual exclusion:
    at most one selection field is set on any well-formed config.

    Semantics:
      - `feature_set` (str) → load `contracts/feature_sets/<name>.json` via
        `FeatureSetRegistry.get(name, verify=True)` (LRU-cached). Raises on
        missing / malformed / tampered file.
      - `feature_preset` (str) → load trainer's `FEATURE_PRESETS` dict via
        `_load_trainer_feature_presets_module`. Raises if module unavailable.
      - `feature_indices` (list[int]) → pass through, sorted + deduped for
        canonical form.
      - None of the three → no-op.

    HARD-FAIL policy: any resolution failure raises `FingerprintNormalizationError`.
    Silent fallback would re-introduce the §3.3b class of ledger-conflation bug
    (distinct broken configs fingerprinting identically because a shared input
    was skipped).

    ORDER-INVARIANT: fingerprint treats selection as a SET. Trainer-side
    execution may use curated order from FEATURE_PRESETS[preset] (e.g.,
    `PRESET_SHORT_TERM_40` is human-curated, not sorted); that is a training
    detail, not a selection identity.
    """
    data_block = effective_train.get("data")
    if not isinstance(data_block, dict):
        return

    # Count the number of selection fields present — DataConfig mutual
    # exclusion guarantees ≤ 1, but we defensively handle the manifest-edit
    # case where a user may have an invalid config. If > 1 is set, we hard-
    # fail: the fingerprint cannot choose which to prefer.
    has_fs = data_block.get("feature_set") is not None
    has_fp = data_block.get("feature_preset") is not None
    has_fi = data_block.get("feature_indices") is not None

    if sum((has_fs, has_fp, has_fi)) == 0:
        return  # No selection; nothing to normalize
    if sum((has_fs, has_fp, has_fi)) > 1:
        raise FingerprintNormalizationError(
            f"Config has multiple feature-selection fields set: "
            f"feature_set={has_fs}, feature_preset={has_fp}, "
            f"feature_indices={has_fi}. DataConfig mutual exclusion forbids "
            f"this; trainer would raise at __post_init__. Fingerprint "
            f"cannot choose between them — set at most one."
        )

    if has_fs:
        feature_set_name = data_block["feature_set"]
        if not isinstance(feature_set_name, str) or not feature_set_name:
            raise FingerprintNormalizationError(
                f"feature_set must be a non-empty string, got "
                f"{feature_set_name!r}."
            )
        try:
            indices = _cached_resolve_feature_set_indices(
                str(paths.feature_sets_dir), feature_set_name
            )
        except Exception as exc:
            raise FingerprintNormalizationError(
                f"Failed to resolve feature_set={feature_set_name!r} from "
                f"registry {paths.feature_sets_dir}: {exc!r}. "
                f"Verify (a) the file exists, (b) content_hash matches, "
                f"(c) contract_version is current."
            ) from exc
        data_block.pop("feature_set")
        data_block["feature_indices"] = list(indices)
    elif has_fp:
        preset_name = data_block["feature_preset"]
        if not isinstance(preset_name, str) or not preset_name:
            raise FingerprintNormalizationError(
                f"feature_preset must be a non-empty string, got "
                f"{preset_name!r}."
            )
        module = _load_trainer_feature_presets_module(paths)
        preset_map = module.FEATURE_PRESETS
        preset_lower = preset_name.lower()
        if preset_lower not in preset_map:
            raise FingerprintNormalizationError(
                f"Unknown feature_preset={preset_name!r}. "
                f"Available: {sorted(preset_map.keys())}."
            )
        raw_indices = preset_map[preset_lower]
        indices = tuple(sorted(set(int(i) for i in raw_indices)))
        data_block.pop("feature_preset")
        data_block["feature_indices"] = list(indices)
    else:  # has_fi — already canonical form; just normalize ordering
        raw_indices = data_block["feature_indices"]
        if not isinstance(raw_indices, (list, tuple)):
            raise FingerprintNormalizationError(
                f"feature_indices must be list/tuple of int, got "
                f"{type(raw_indices).__name__}."
            )
        try:
            indices = tuple(sorted(set(int(i) for i in raw_indices)))
        except (TypeError, ValueError) as exc:
            raise FingerprintNormalizationError(
                f"feature_indices contains non-integer values: "
                f"{raw_indices!r}. {exc!r}"
            ) from exc
        data_block["feature_indices"] = list(indices)


def _load_trainer_config_resolved(
    path: Path,
    paths: "PipelinePaths",
) -> Dict[str, Any]:
    """Load a trainer YAML AND resolve ``_base:`` inheritance.

    Phase 3.3b fix: the fingerprint must hash the effective RESOLVED config,
    not the pre-inheritance YAML with its sparse ``_base: [...]`` + overrides.
    Without this, mutating a shared base (e.g., bumping ``train.epochs`` in
    ``bases/train/regression_default.yaml``) would leave every dependent
    experiment's fingerprint unchanged — silently conflating pre/post-mutation
    runs in the ledger.

    Error propagation policy (refined per post-Batch-1 audit):
    - **Soft errors** (file missing, YAML parse failure, non-dict YAML, raw
      load failure): return ``{}`` and log a warning. These are pre-existing
      fail-safe cases — configs that don't load at all don't produce
      meaningful fingerprints; better to surface an empty hash component
      than crash the sweep.
    - **Hard errors** (inheritance cycle, depth exceeded, malformed ``_base``):
      PROPAGATE the ``ValueError``. These indicate a MALFORMED experiment
      config that MUST be fixed before running — silently returning ``{}``
      would produce fingerprint collisions for distinct-but-broken configs,
      re-creating the ledger-conflation bug §3.3b was introduced to fix.

    Args:
        path: Absolute path to the trainer YAML.
        paths: PipelinePaths for locating the trainer merge.py.

    Returns:
        Fully resolved config dict (``_base:`` expanded and stripped). Empty
        dict on soft errors. Raises on hard errors.

    Raises:
        ValueError: Inheritance cycle, depth exceeded, or malformed ``_base``.
        FileNotFoundError: Referenced base config not found (propagates from
            ``resolve_inheritance``).
    """
    if not path.exists():
        return {}
    if path.suffix.lower() not in (".yaml", ".yml"):
        return _load_config_as_dict(path)

    merge_mod = _load_trainer_merge_module(paths)
    if merge_mod is None:
        # Fall back to raw YAML (fingerprint becomes base-unaware in this env).
        logger.warning(
            "Trainer merge.py unavailable for %s; fingerprint will be "
            "base-resolution-unaware (pre-§3.3b behavior). Install lobtrainer "
            "or ensure PipelinePaths.trainer_dir points to a valid trainer.",
            path,
        )
        return _load_config_as_dict(path)

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        # Soft: file I/O or YAML-syntax failure. Log + empty component.
        logger.warning(
            "Trainer config %s could not be parsed for fingerprinting (%s): "
            "%s. Fingerprint will use empty training component.",
            path, type(exc).__name__, exc,
        )
        return {}

    if not isinstance(raw, dict):
        logger.warning(
            "Trainer config %s YAML did not parse to a dict (got %s). "
            "Fingerprint will use empty training component.",
            path, type(raw).__name__,
        )
        return {}

    # HARD errors — let them propagate. ValueError (cycle/depth/malformed
    # _base) and FileNotFoundError (missing base file) indicate a broken
    # config that MUST be fixed; silently swallowing them recreates the
    # ledger-conflation bug.
    return merge_mod.resolve_inheritance(raw, path.resolve())


def _resolve_inline_trainer_config(
    inline_cfg: Dict[str, Any],
    manifest_path: Path,
    paths: "PipelinePaths",
) -> Dict[str, Any]:
    """Resolve ``_base:`` in an inline trainer_config dict from a manifest.

    Phase 3.3b: when a manifest uses inline ``trainer_config: {_base: [...], ...}``
    (wrapper-less Phase 1 form), the fingerprint must hash the resolved
    effective dict, not the sparse inline form.

    Path resolution convention: **relative ``_base:`` paths in inline
    trainer_configs are ALWAYS rooted at ``<trainer_dir>/configs/``**,
    regardless of manifest location. This matches the runtime behavior in
    ``hft-ops/src/hft_ops/stages/training.py::_absolutize_inline_base_paths``.
    We pass a fictitious file path inside trainer_configs so
    ``resolve_inheritance`` (which uses ``config_path.parent`` for relative
    resolution) gets the correct root.

    Error propagation policy (refined per post-Batch-1 audit):
    - **Soft fallback** (no ``_base`` key, missing merge module): return
      ``inline_cfg`` unchanged. These are benign — config-without-inheritance
      or degraded CI environment.
    - **Hard errors** (inheritance cycle, depth exceeded, malformed ``_base``):
      PROPAGATE ``ValueError``. Distinct-but-broken inline configs must not
      hash to the same fingerprint.

    Raises:
        ValueError: Inheritance cycle, depth exceeded, or malformed ``_base``.
        FileNotFoundError: Referenced base config not found.
    """
    import copy

    if "_base" not in inline_cfg:
        return inline_cfg

    merge_mod = _load_trainer_merge_module(paths)
    if merge_mod is None:
        logger.warning(
            "Trainer merge.py unavailable for inline trainer_config "
            "fingerprinting (manifest %s); falling back to non-resolved "
            "inline dict.",
            manifest_path,
        )
        return inline_cfg

    mutable = copy.deepcopy(inline_cfg)
    # Relative _base paths are rooted at the trainer configs dir (matches
    # _absolutize_inline_base_paths runtime convention). resolve_inheritance
    # resolves relative paths via `config_path.parent`, so we pass a
    # fictitious file path INSIDE trainer_configs so parent == trainer_configs.
    trainer_configs = (paths.trainer_dir / "configs").resolve()
    fake_config_path = trainer_configs / "__inline_trainer_config__.yaml"
    # Hard errors propagate (cycle, depth, malformed _base)
    return merge_mod.resolve_inheritance(mutable, fake_config_path)


def _extract_fingerprint_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields that affect experiment outcomes.

    Strips metadata fields (name, description, tags, output paths, log levels)
    that do not affect numerical results. This ensures that changing only the
    experiment name does not produce a different fingerprint.
    """
    exclude_keys = {
        "name", "description", "tags", "version",
        "output_dir", "log_level", "verbose",
        "experiment",
        # Phase 8C-α post-audit round-2 (2026-04-20 architect-Q7 wire-in):
        # `importance` is an OBSERVATION (post-training permutation
        # importance), NOT a treatment. Enabling importance on an existing
        # experiment does NOT change what gets trained; it only adds a
        # post-hoc analysis. Fingerprint-including would create
        # Phase-3-§3.3b-class ledger conflation: same trained model →
        # different fingerprints depending on whether the operator asked
        # for importance. Excluded for the same reason `artifacts[]` /
        # `gate_reports[]` are excluded. Locked by regression test
        # `test_importance_field_excluded_from_fingerprint`.
        "importance",
        # Round-3 post-audit Agent-3 H2 defensive add: `artifacts` lives
        # on ``ExperimentRecord`` (output side) and structurally does
        # NOT flow into `compute_fingerprint` (which reads the manifest
        # config tree, not the record). Blacklisting it here is
        # defense-in-depth: if a FUTURE refactor ever serializes record
        # fields into the fingerprint input — e.g., for "fingerprint
        # includes observed gate_reports" hypothetical — the strip
        # catches it before it becomes a Phase-3-§3.3b-class
        # ledger-conflation bug. Matches the symmetry with
        # `importance` (both are observations).
        "artifacts",
    }

    def _strip(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in d.items():
            if k in exclude_keys:
                continue
            if isinstance(v, dict):
                stripped = _strip(v)
                if stripped:
                    result[k] = stripped
            else:
                result[k] = v
        return result

    return _strip(config)


def compute_fingerprint(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
) -> str:
    """Compute a deterministic fingerprint for an experiment.

    The fingerprint incorporates:
    1. Extractor config (outcome-affecting fields only)
    2. Trainer config (outcome-affecting fields only)
    3. Backtest parameters
    4. Data directory manifest (file names + sizes)
    5. Contract version

    Args:
        manifest: The resolved experiment manifest.
        paths: Pipeline paths for resolving config file locations.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    fp, _ = compute_fingerprint_explain(manifest, paths)
    return fp


def compute_fingerprint_explain(
    manifest: ExperimentManifest,
    paths: PipelinePaths,
) -> Tuple[str, Dict[str, Any]]:
    """Compute fingerprint AND return the normalized components dict.

    Enhancement A (Phase 4 Batch 4c.3, 2026-04-16): self-service debugging
    for "why do these two manifests fingerprint differently?" Both the
    returned dict AND the hash are deterministic; two manifests that
    normalize identically will produce byte-equal components dicts AND
    equal hashes. Diff the components to locate divergence.

    Returns:
        (fingerprint_hex, components_dict). The components dict is the
        exact structure fed to canonical_json_blob + sha256_hex.
    """
    components: Dict[str, Any] = {}

    if manifest.stages.extraction.config:
        ext_path = paths.resolve(manifest.stages.extraction.config)
        ext_cfg = _load_config_as_dict(ext_path)
        components["extraction"] = _extract_fingerprint_fields(ext_cfg)

    # Materialize the effective trainer config from EITHER legacy-path
    # (training.config → load YAML) OR inline (training.trainer_config dict).
    # This ensures fingerprint equivalence between the two entry points.
    #
    # Phase 3.3b: trainer configs are loaded with _base: inheritance RESOLVED
    # so the fingerprint reflects the effective dict, not the sparse pre-
    # inheritance YAML. Without this, mutating a shared base silently leaves
    # dependent fingerprints unchanged — a ledger-conflation bug.
    effective_train: Optional[Dict[str, Any]] = None
    if manifest.stages.training.config:
        train_path = paths.resolve(manifest.stages.training.config)
        train_cfg = _load_trainer_config_resolved(train_path, paths)
        effective_train = _extract_fingerprint_fields(train_cfg)
    elif manifest.stages.training.trainer_config is not None:
        # Inline trainer_config dict may also use `_base:` (multi-base or
        # single). Resolve via trainer configs dir.
        manifest_path = Path(manifest.manifest_path) if manifest.manifest_path else Path.cwd()
        resolved_inline = _resolve_inline_trainer_config(
            dict(manifest.stages.training.trainer_config),
            manifest_path,
            paths,
        )
        effective_train = _extract_fingerprint_fields(resolved_inline)

    if effective_train is not None:
        for key, value in manifest.stages.training.overrides.items():
            parts = key.split(".")
            target = effective_train
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        # Round-3 post-audit Agent-3 H2 fix: re-strip after overrides are
        # merged. Overrides can inject observation-tier keys (e.g.,
        # `overrides["importance.enabled"] = True`) AFTER the initial
        # `_extract_fingerprint_fields` stripped them — a hidden path
        # for observation fields to leak into fingerprints. Pass 2 of
        # the blacklist ensures observations-via-overrides also strip.
        effective_train = _extract_fingerprint_fields(effective_train)

        if manifest.stages.training.horizon_value is not None:
            effective_train["_horizon_value"] = manifest.stages.training.horizon_value

        # Phase 4 Batch 4c.3 (2026-04-16): normalize feature_set /
        # feature_preset / feature_indices → canonical `feature_indices` tuple
        # so that logically-equivalent configs produce byte-equal fingerprints.
        # Prior to 4c.3: `pass` placeholder left fields intact → two configs
        # differing only in selection-field representation fingerprinted
        # DIFFERENTLY (known limitation documented in plan §4c.3).
        #
        # Insertion point: AFTER overrides applied (L336-343) and AFTER
        # horizon_value injection (L345-346), so `--override data.feature_set=X`
        # CLI overrides are also normalized.
        #
        # HARD-FAIL policy: any resolution failure raises
        # `FingerprintNormalizationError`. Silent fallback would re-introduce
        # the §3.3b class of ledger-conflation bug.
        _normalize_feature_selection_for_fingerprint(effective_train, paths)

        components["training"] = effective_train

    # Validation stage is an OBSERVATION, not a TREATMENT — two runs that
    # differ only in gate thresholds (min_ic=0.05 vs 0.03) should dedupe as
    # the same experiment. So we DO NOT include validation in the fingerprint.
    # (Note: stages.validation.enabled=True AND a failing gate may prevent a
    # run from completing, but the training that WOULD happen is identical.)

    if manifest.stages.backtesting.enabled:
        from dataclasses import asdict
        components["backtest"] = asdict(manifest.stages.backtesting.params)

    # Phase 5 FULL-A fingerprint coverage extension (2026-04-17).
    #
    # Stage coverage as of Phase 5 FULL-A:
    #   extraction         — TOML config hashed (above, L624-627)
    #   training           — full _base: resolution + overrides + feature-set normalization (above, L637-682)
    #   signal_export      — script/split/extra_args hashed HERE (only when non-default, back-compat guard)
    #   backtesting        — params dataclass (above, L690-692) + script/extra_args HERE (only when non-default)
    #   validation         — EXCLUDED (observation, not treatment; see L684-688 rationale)
    #   raw_analysis,
    #   dataset_analysis   — EXCLUDED (exploratory; not outcome-affecting)
    #
    # Known structural debt: stage coverage is OPT-IN per-stage. A future
    # FingerprintPolicy refactor to opt-out would require bumping
    # `[contract.schema_version]` + re-fingerprinting live records; deferred to
    # Phase 6+ (BQP Phase 10 §6.1 — canonical_hash is REUSED verbatim by
    # Phase 11, so fingerprint contract changes require cross-repo coordination).
    #
    # Back-compat invariant: manifests using default signal_export and default
    # backtesting.script WITHOUT extra_args produce IDENTICAL fingerprints
    # pre/post this block — the new components are added ONLY when a researcher
    # overrides defaults (which current production manifests never do today).
    # Phase 5 FULL-A post-audit fix (Agent 1 H3): derive defaults from the
    # dataclass fields themselves so schema.py changes auto-propagate. Prior
    # hardcoded strings silently drifted if schema defaults changed.
    from dataclasses import fields as _dc_fields
    from hft_ops.manifest.schema import SignalExportStage as _SES, BacktestingStage as _BTS
    _SES_DEFAULTS = {f.name: f.default for f in _dc_fields(_SES)}
    _BTS_DEFAULTS = {f.name: f.default for f in _dc_fields(_BTS)}
    _DEFAULT_SIGNAL_EXPORT_SCRIPT = _SES_DEFAULTS["script"]
    _DEFAULT_SIGNAL_EXPORT_SPLIT = _SES_DEFAULTS["split"]
    _DEFAULT_BACKTEST_SCRIPT = _BTS_DEFAULTS["script"]

    if manifest.stages.signal_export.enabled:
        se_fields: Dict[str, Any] = {}
        if manifest.stages.signal_export.script != _DEFAULT_SIGNAL_EXPORT_SCRIPT:
            se_fields["script"] = manifest.stages.signal_export.script
        if manifest.stages.signal_export.split and manifest.stages.signal_export.split != _DEFAULT_SIGNAL_EXPORT_SPLIT:
            se_fields["split"] = manifest.stages.signal_export.split
        if manifest.stages.signal_export.extra_args:
            se_fields["extra_args"] = list(manifest.stages.signal_export.extra_args)
        if se_fields:
            components["signal_export"] = se_fields

    if manifest.stages.backtesting.enabled:
        bt_script_fields: Dict[str, Any] = {}
        if manifest.stages.backtesting.script != _DEFAULT_BACKTEST_SCRIPT:
            bt_script_fields["script"] = manifest.stages.backtesting.script
        if manifest.stages.backtesting.extra_args:
            bt_script_fields["extra_args"] = list(manifest.stages.backtesting.extra_args)
        if bt_script_fields:
            components["backtest_script"] = bt_script_fields

    if manifest.stages.extraction.output_dir:
        data_dir = paths.resolve(manifest.stages.extraction.output_dir)
        if data_dir.exists():
            components["data_manifest"] = hash_directory_manifest(data_dir)

    components["contract_version"] = manifest.experiment.contract_version

    # If NONE of the outcome-affecting stages contributed content (i.e., this
    # is a metadata-only / retroactive manifest with all stages disabled and
    # no trainer/extractor config referenced), include the experiment.name in
    # the fingerprint so distinct metadata records don't collide. For normal
    # run manifests, the extraction/training blocks already dominate the hash
    # and the name exclusion preserves the intended "re-run dedup" behavior.
    if not any(k in components for k in ("extraction", "training", "backtest", "data_manifest")):
        components["_metadata_only_name"] = manifest.experiment.name

    # Phase 4 Batch 4c hardening: single-source canonical form via
    # hft_contracts.canonical_hash. Byte-parity with prior inline impl
    # locked by hft-contracts/tests/test_canonical_hash.py::TestMonorepoConventionAlignment.
    from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex
    fp = sha256_hex(canonical_json_blob(components))
    return fp, components


def check_duplicate(
    fingerprint: str,
    ledger_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Check if an experiment with this fingerprint already exists.

    Args:
        fingerprint: The experiment fingerprint to check.
        ledger_dir: Path to the ledger directory.

    Returns:
        The matching index entry dict if a duplicate exists, None otherwise.
    """
    index_path = ledger_dir / "index.json"
    if not index_path.exists():
        return None

    # Phase 8B MUST-FIX (Agent 1 BUG-1, 2026-04-20): previously this reader
    # did a direct ``json.load`` bypassing ``ExperimentLedger._load_index``,
    # which silently broke the strict-mode contract AND allowed duplicate
    # registration when on-disk envelope was stale (a rebuilt envelope had
    # more entries than the pre-rebuild snapshot dedup saw). Fixed by
    # routing through ``ExperimentLedger`` so the same envelope-detection +
    # strict-mode dispatch runs here as in all other ledger consumers.
    #
    # Exception dispatch:
    #   - ``StaleLedgerIndexError`` (strict mode + stale index) → propagate.
    #     This is the Phase 8B CI-fails-fast contract: dedup is the most
    #     frequent code path; masking staleness here would defeat the whole
    #     trackability substrate.
    #   - ``OSError`` / ``json.JSONDecodeError`` → return None (preserves
    #     the prior permissive behavior for non-staleness errors like
    #     ``PermissionError``). Under non-strict mode, malformed JSON is
    #     handled internally by ``_load_index`` and never leaks here.
    from hft_ops.ledger.ledger import ExperimentLedger, StaleLedgerIndexError

    try:
        ledger = ExperimentLedger(ledger_dir)
    except StaleLedgerIndexError:
        # Strict-mode staleness signal — fail fast; do NOT silently return
        # None (that would mask the staleness from the caller, re-opening
        # the silent-omission class Phase 8B exists to eliminate).
        raise
    except (OSError, json.JSONDecodeError):
        # Non-staleness construction failures (permission denied, etc.) —
        # permissive behavior: caller can proceed without dedup.
        return None

    entry = ledger.find_by_fingerprint(fingerprint)

    # Phase 8A.1 (2026-04-20): SWEEP_FAILURE records share their
    # fingerprint with the would-be successful record for the same
    # treatment — so retries can match for downstream analysis. If
    # check_duplicate returned a SWEEP_FAILURE match, operators retrying
    # a failed grid point would be silently blocked as "already run".
    # Filter them out so retries run.
    #
    # Post-audit (agent-C M1): reference the enum value instead of a
    # string literal so a future rename of RecordType.SWEEP_FAILURE
    # breaks the filter at import-time rather than silently shipping.
    #
    # Locked by ``test_scheduler_parallel.py::TestDedupSkipsSweepFailure``.
    from hft_contracts.experiment_record import RecordType
    if entry is not None and entry.get("record_type") == RecordType.SWEEP_FAILURE.value:
        return None
    return entry
