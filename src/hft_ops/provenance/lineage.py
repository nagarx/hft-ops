"""
Provenance capture for experiment lineage.

Captures git state, config file hashes, data directory manifests, and
timestamps to enable full traceability of every experiment. Each
ExperimentRecord carries a Provenance object that answers: "what code,
config, and data produced this result?"

Design (UNIFIED_PIPELINE_ARCHITECTURE_PLAN.md, Phase 4, Section 6.3):
    provenance = {
        git_commit_hash, git_dirty, git_branch,
        config_hashes: {extractor: sha256, trainer: sha256, manifest: sha256},
        data_dir_hash: sha256 of sorted filenames + sizes,
        contract_version, export_timestamp_utc
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GitInfo:
    """Captured git state at experiment time."""

    commit_hash: str = ""
    branch: str = ""
    dirty: bool = False
    short_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GitInfo:
        return cls(
            commit_hash=data.get("commit_hash", ""),
            branch=data.get("branch", ""),
            dirty=data.get("dirty", False),
            short_hash=data.get("short_hash", ""),
        )


def capture_git_info(repo_dir: Path) -> GitInfo:
    """Capture current git state from a repository directory.

    Args:
        repo_dir: Path to the git repository root.

    Returns:
        GitInfo with commit hash, branch, and dirty status.
        Returns empty GitInfo if git is not available or dir is not a repo.
    """
    info = GitInfo()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.commit_hash = result.stdout.strip()
            info.short_hash = info.commit_hash[:8]
    except (subprocess.SubprocessError, FileNotFoundError):
        return info

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.branch = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.dirty = bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return info


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 hash string, or empty string if file not found.
    """
    if not path.exists():
        return ""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config_dict(config: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a config dict.

    Serializes the dict to sorted JSON for deterministic hashing.

    Args:
        config: Configuration dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_directory_manifest(dir_path: Path) -> str:
    """Compute a hash representing a directory's file manifest.

    Hashes the sorted list of (relative_path, file_size) tuples for all
    files in the directory. This detects file additions, deletions, and
    size changes without reading file contents (which could be terabytes).

    Args:
        dir_path: Path to the directory.

    Returns:
        Hex-encoded SHA-256 hash, or empty string if directory not found.
    """
    if not dir_path.exists():
        return ""

    entries: list[tuple[str, int]] = []
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel = fpath.relative_to(dir_path)
            try:
                size = fpath.stat().st_size
            except OSError:
                size = -1
            entries.append((str(rel), size))

    entries.sort()
    manifest_str = json.dumps(entries)
    return hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()


@dataclass
class Provenance:
    """Full provenance record for an experiment.

    Captures everything needed to trace an experiment back to the exact
    code, config, and data that produced it.
    """

    git: GitInfo = field(default_factory=GitInfo)
    config_hashes: Dict[str, str] = field(default_factory=dict)
    data_dir_hash: str = ""
    contract_version: str = ""
    timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "git": self.git.to_dict(),
            "config_hashes": self.config_hashes,
            "data_dir_hash": self.data_dir_hash,
            "contract_version": self.contract_version,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Provenance:
        return cls(
            git=GitInfo.from_dict(data.get("git", {})),
            config_hashes=data.get("config_hashes", {}),
            data_dir_hash=data.get("data_dir_hash", ""),
            contract_version=data.get("contract_version", ""),
            timestamp_utc=data.get("timestamp_utc", ""),
        )


def build_provenance(
    pipeline_root: Path,
    *,
    manifest_path: Optional[Path] = None,
    extractor_config_path: Optional[Path] = None,
    trainer_config_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    contract_version: str = "",
) -> Provenance:
    """Build a complete provenance record.

    Args:
        pipeline_root: Path to HFT-pipeline-v2 root (for git info).
        manifest_path: Path to the experiment manifest YAML.
        extractor_config_path: Path to the extractor TOML config.
        trainer_config_path: Path to the trainer YAML config.
        data_dir: Path to the data export directory.
        contract_version: Contract schema version string.

    Returns:
        Provenance with all captured information.
    """
    git = capture_git_info(pipeline_root)
    now = datetime.now(timezone.utc).isoformat()

    config_hashes: Dict[str, str] = {}
    if manifest_path:
        config_hashes["manifest"] = hash_file(manifest_path)
    if extractor_config_path:
        config_hashes["extractor"] = hash_file(extractor_config_path)
    if trainer_config_path:
        config_hashes["trainer"] = hash_file(trainer_config_path)

    data_hash = ""
    if data_dir:
        data_hash = hash_directory_manifest(data_dir)

    return Provenance(
        git=git,
        config_hashes=config_hashes,
        data_dir_hash=data_hash,
        contract_version=contract_version,
        timestamp_utc=now,
    )
