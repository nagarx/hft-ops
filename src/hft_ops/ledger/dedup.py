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

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from hft_ops.manifest.schema import ExperimentManifest
from hft_ops.paths import PipelinePaths
from hft_ops.provenance.lineage import hash_directory_manifest

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


def _load_config_as_dict(path: Path) -> Dict[str, Any]:
    """Load a TOML or YAML config file as a dict."""
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
    components: Dict[str, Any] = {}

    if manifest.stages.extraction.config:
        ext_path = paths.resolve(manifest.stages.extraction.config)
        ext_cfg = _load_config_as_dict(ext_path)
        components["extraction"] = _extract_fingerprint_fields(ext_cfg)

    if manifest.stages.training.config:
        train_path = paths.resolve(manifest.stages.training.config)
        train_cfg = _load_config_as_dict(train_path)
        effective_train = _extract_fingerprint_fields(train_cfg)

        for key, value in manifest.stages.training.overrides.items():
            parts = key.split(".")
            target = effective_train
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        if manifest.stages.training.horizon_value is not None:
            effective_train["_horizon_value"] = manifest.stages.training.horizon_value

        components["training"] = effective_train

    if manifest.stages.backtesting.enabled:
        from dataclasses import asdict
        components["backtest"] = asdict(manifest.stages.backtesting.params)

    if manifest.stages.extraction.output_dir:
        data_dir = paths.resolve(manifest.stages.extraction.output_dir)
        if data_dir.exists():
            components["data_manifest"] = hash_directory_manifest(data_dir)

    components["contract_version"] = manifest.experiment.contract_version

    serialized = json.dumps(components, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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

    try:
        with open(index_path, "r") as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for entry in index:
        if entry.get("fingerprint") == fingerprint:
            return entry

    return None
