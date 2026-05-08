"""
Pipeline path resolution.

Resolves all module directories, config files, and data paths from a single
pipeline_root. Every path used by hft-ops is derived here -- no hardcoded
paths anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelinePaths:
    """Resolves all module paths from a single pipeline_root.

    Args:
        pipeline_root: Absolute path to the HFT-pipeline-v2 directory.
            All module directories are siblings within this root.
    """

    pipeline_root: Path

    def __post_init__(self) -> None:
        # Phase α-1.1 / #PY-83 fix (2026-05-10): use Path.absolute() not
        # Path.resolve() to preserve symlink-source lineage. Per α-3 / #PY-79
        # lesson — when `data/` is symlinked to external mount, resolve()
        # derefs the symlink at start, breaking downstream relpath logic
        # that expects monorepo-root-anchored paths. absolute() preserves
        # the symlink-source so subsequent walks find contracts/, etc.
        if not self.pipeline_root.is_absolute():
            object.__setattr__(
                self, "pipeline_root", self.pipeline_root.absolute()
            )

    # -- Module directories --------------------------------------------------

    @property
    def reconstructor_dir(self) -> Path:
        return self.pipeline_root / "MBO-LOB-reconstructor"

    @property
    def extractor_dir(self) -> Path:
        return self.pipeline_root / "feature-extractor-MBO-LOB"

    @property
    def raw_analyzer_dir(self) -> Path:
        return self.pipeline_root / "MBO-LOB-analyzer"

    @property
    def dataset_analyzer_dir(self) -> Path:
        return self.pipeline_root / "lob-dataset-analyzer"

    @property
    def models_dir(self) -> Path:
        return self.pipeline_root / "lob-models"

    @property
    def trainer_dir(self) -> Path:
        return self.pipeline_root / "lob-model-trainer"

    @property
    def backtester_dir(self) -> Path:
        return self.pipeline_root / "lob-backtester"

    # -- Shared infrastructure -----------------------------------------------

    @property
    def contracts_dir(self) -> Path:
        return self.pipeline_root / "contracts"

    @property
    def contract_toml(self) -> Path:
        return self.contracts_dir / "pipeline_contract.toml"

    @property
    def feature_sets_dir(self) -> Path:
        """Directory holding content-addressed FeatureSet JSON artifacts.

        Phase 4 registry location (see `contracts/feature_sets/SCHEMA.md`).
        Committed to git as a contract surface; mutable-add-only (new
        FeatureSets accumulate over time; each file is immutable once
        written).
        """
        return self.contracts_dir / "feature_sets"

    @property
    def hft_contracts_dir(self) -> Path:
        return self.pipeline_root / "hft-contracts"

    @property
    def hft_ops_dir(self) -> Path:
        return self.pipeline_root / "hft-ops"

    # -- Data & artifacts ----------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.pipeline_root / "data"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def ledger_dir(self) -> Path:
        return self.hft_ops_dir / "ledger"

    @property
    def experiments_dir(self) -> Path:
        return self.hft_ops_dir / "experiments"

    @property
    def runs_dir(self) -> Path:
        """Per-experiment run output directories."""
        return self.ledger_dir / "runs"

    # -- Convenience ---------------------------------------------------------

    def resolve(self, relative_path: str) -> Path:
        """Make a path absolute relative to pipeline_root, preserving symlink-source.

        Phase α-1.1 / #PY-83 fix (2026-05-10): default behavior is
        `.absolute()` (preserve symlink-source lineage) per α-3 / #PY-79
        lesson. For cases where the canonical filesystem path is required
        (cache-key inputs, content-addressed hashes, lineage manifests
        where symlink-equivalence must collapse), use `canonical()` instead.

        Used by 47+ call sites in hft-ops (manifest/loader, cli, validators,
        stage runners, ledger). NONE of these need symlink-deref; they all
        pass through `os.path.relpath` or file existence checks where
        symlink-source preservation is correct.
        """
        return (self.pipeline_root / relative_path).absolute()

    def canonical(self, relative_path: str) -> Path:
        """Resolve a path to its canonical filesystem location (DEREFERENCES symlinks).

        Phase α-1.1 / #PY-83 (2026-05-10): explicit escape hatch for the
        rare cases where symlink-equivalence must collapse, e.g.:
        - Content-addressed cache keys (two configs pointing through different
          symlinks but to same physical input must collide)
        - Lineage manifests where data_dir hash must be canonical
        - Symlink-tree builders computing relative-symlink portability

        Default consumers should use `resolve()` instead — symlink-source
        preservation is correct for path substitution, relpath, and most
        I/O paths.

        Note: `extraction_cache.py` + `feature_sets/producer.py` already
        bypass `paths.resolve()` by calling `(extractor_dir / x).resolve()`
        directly for cache-key purposes — they do not need migration.
        """
        return (self.pipeline_root / relative_path).resolve()

    def validate(self) -> list[str]:
        """Check that critical directories exist. Returns list of errors."""
        errors: list[str] = []
        required = [
            ("pipeline_root", self.pipeline_root),
            ("contracts", self.contracts_dir),
            ("contract_toml", self.contract_toml),
            ("extractor", self.extractor_dir),
            ("trainer", self.trainer_dir),
            ("backtester", self.backtester_dir),
        ]
        for name, path in required:
            if not path.exists():
                errors.append(f"Missing required path: {name} = {path}")
        return errors

    @classmethod
    def auto_detect(cls) -> PipelinePaths:
        """Auto-detect pipeline root by walking up from hft-ops/src/hft_ops/.

        Phase α-1.1 / #PY-83 fix (2026-05-10): use `.absolute()` not
        `.resolve()` so that hft-ops checked out under a symlinked
        directory still detects the correct pipeline root via symlink-source
        lineage. Mirrors α-3 / #PY-79 fix in
        lob-model-trainer/feature_set_resolver.py:442.
        """
        here = Path(__file__).absolute()
        # hft-ops/src/hft_ops/paths.py -> walk up 4 levels to pipeline root
        candidate = here.parent.parent.parent.parent
        if (candidate / "contracts" / "pipeline_contract.toml").exists():
            return cls(pipeline_root=candidate)
        raise FileNotFoundError(
            f"Cannot auto-detect pipeline root from {here}. "
            f"Checked: {candidate}. "
            "Set pipeline_root explicitly or use --pipeline-root CLI flag."
        )
