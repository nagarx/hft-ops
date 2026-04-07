"""
Experiment record: immutable, self-contained record of a completed experiment.

Each record captures the full configuration snapshot, provenance, results,
and metadata needed to reproduce and compare experiments. Records are
append-only -- once written, they are never modified (except the `notes`
field for post-experiment observations).

Design reference: UNIFIED_PIPELINE_ARCHITECTURE_PLAN.md, Phase 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_ops.provenance.lineage import Provenance


@dataclass
class ExperimentRecord:
    """Complete record of a single experiment run.

    Attributes:
        experiment_id: Unique identifier ({name}_{timestamp}_{fingerprint[:8]}).
        name: Human-readable experiment name (from manifest).
        manifest_path: Absolute path to the source manifest YAML.
        fingerprint: SHA-256 of resolved config for dedup.

        provenance: Full provenance (git, config hashes, data hash, timestamp).
        contract_version: Pipeline contract version at time of experiment.

        extraction_config: Full extractor TOML as dict.
        training_config: Full trainer YAML as dict.
        backtest_params: Backtest parameters as dict.

        training_metrics: Training results (accuracy, f1, per-class, etc.).
        backtest_metrics: Backtest results (return, sharpe, drawdown, etc.).
        dataset_health: Key stats from dataset analysis.

        tags: User-defined tags for filtering.
        hypothesis: What the experiment aims to test.
        description: Detailed experiment description.
        notes: Post-experiment observations (mutable field).

        created_at: ISO 8601 creation timestamp.
        duration_seconds: Wall-clock time for full pipeline.
        status: completed | failed | partial.
        stages_completed: Which stages ran successfully.
    """

    experiment_id: str = ""
    name: str = ""
    manifest_path: str = ""
    fingerprint: str = ""

    provenance: Provenance = field(default_factory=Provenance)
    contract_version: str = ""

    extraction_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    backtest_params: Dict[str, Any] = field(default_factory=dict)

    training_metrics: Dict[str, Any] = field(default_factory=dict)
    backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    dataset_health: Dict[str, Any] = field(default_factory=dict)

    tags: List[str] = field(default_factory=list)
    hypothesis: str = ""
    description: str = ""
    notes: str = ""

    created_at: str = ""
    duration_seconds: float = 0.0
    status: str = "pending"
    stages_completed: List[str] = field(default_factory=list)

    # Sweep metadata (populated when this record is part of a sweep)
    sweep_id: str = ""
    """Sweep identifier linking this record to its parent sweep."""

    axis_values: Dict[str, str] = field(default_factory=dict)
    """Axis name -> selected label for this grid point (e.g., {"model": "tlob", "horizon": "H10"})."""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentRecord:
        """Deserialize from a dict."""
        prov_data = data.pop("provenance", {})
        record = cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })
        record.provenance = Provenance.from_dict(prov_data)
        return record

    def save(self, path: Path) -> None:
        """Save record to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> ExperimentRecord:
        """Load record from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def index_entry(self) -> Dict[str, Any]:
        """Create a lightweight index entry for fast ledger queries.

        Contains enough metadata for filtering and comparison without
        loading the full record.
        """
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "contract_version": self.contract_version,
            "tags": self.tags,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "stages_completed": self.stages_completed,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
            "training_metrics": {
                k: v for k, v in self.training_metrics.items()
                if k in (
                    "accuracy", "macro_f1", "macro_precision", "macro_recall",
                    "best_val_accuracy", "best_val_macro_f1", "best_epoch",
                )
            },
            "backtest_metrics": {
                k: v for k, v in self.backtest_metrics.items()
                if k in (
                    "total_return", "sharpe_ratio", "max_drawdown",
                    "win_rate", "total_trades",
                )
            },
            "model_type": self.training_config.get("model", {}).get("model_type", ""),
            "labeling_strategy": self.training_config.get("data", {}).get(
                "labeling_strategy", ""
            ),
            "sweep_id": self.sweep_id,
            "axis_values": self.axis_values,
        }
