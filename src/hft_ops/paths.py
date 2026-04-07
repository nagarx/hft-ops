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
        if not self.pipeline_root.is_absolute():
            object.__setattr__(
                self, "pipeline_root", self.pipeline_root.resolve()
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
        """Resolve a path relative to pipeline_root."""
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
        """Auto-detect pipeline root by walking up from hft-ops/src/hft_ops/."""
        here = Path(__file__).resolve()
        # hft-ops/src/hft_ops/paths.py -> walk up 4 levels to pipeline root
        candidate = here.parent.parent.parent.parent
        if (candidate / "contracts" / "pipeline_contract.toml").exists():
            return cls(pipeline_root=candidate)
        raise FileNotFoundError(
            f"Cannot auto-detect pipeline root from {here}. "
            f"Checked: {candidate}. "
            "Set pipeline_root explicitly or use --pipeline-root CLI flag."
        )
