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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from hft_ops.provenance.lineage import Provenance


class RecordType(str, Enum):
    """Type of experiment a ledger record represents.

    Introduced in Phase 1.3 to accommodate the full scope of past/future
    experiments. Not every "experiment" produces a trainer ``history.json`` —
    many analytical studies (E7-E16) produce only analyzer/evaluator output,
    and some post-hoc calibrations (E6) produce only signals.

    Each type has a reduced-fidelity schema:

    - ``training``: Full trainer run. ``training_metrics`` populated, optionally
      ``backtest_metrics`` too. The default.
    - ``analysis``: Diagnostic study (no training, no backtest). Results live
      in ``training_metrics`` as a free-form dict OR in ``notes``.
    - ``calibration``: Post-hoc calibration of an existing trained model.
      References a parent training record via ``parent_experiment_id``.
    - ``backtest``: Backtest-only experiment (pre-existing model, pre-existing
      signals). ``backtest_metrics`` populated.
    - ``evaluation``: 5-path feature-evaluator run producing a classification
      table / feature profiles. ``training_metrics`` holds the summary.
    - ``sweep_aggregate``: Aggregate record for a multi-run script (e.g.,
      e4_baselines.py that runs 5 models). Sub-results live in ``sub_records``.
    """

    TRAINING = "training"
    ANALYSIS = "analysis"
    CALIBRATION = "calibration"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"
    SWEEP_AGGREGATE = "sweep_aggregate"


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

    # Phase 4 Batch 4c.4 (2026-04-16): optional reference to the FeatureSet
    # registry entry used at trainer time. Top-level (not nested under
    # `Provenance.config_hashes`) because it's a structured reference with
    # identity (name + content_hash), not an opaque config-hash. Query
    # pattern: `record.feature_set_ref["name"] == "momentum_v1"`.
    # None iff the trainer did not use `DataConfig.feature_set`
    # (legacy path, explicit feature_indices, or feature_preset).
    # See PA §13.4.2 for why this is NOT in Provenance.config_hashes
    # (that dict's implicit contract is values are SHA-256 hex; forcing a
    # structured reference into it would violate that).
    feature_set_ref: Optional[Dict[str, str]] = None

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

    # Phase 1.3 record-typing fields
    record_type: str = "training"
    """Type of record. One of: training, analysis, calibration, backtest, evaluation,
    sweep_aggregate. Use the ``RecordType`` enum's ``.value`` for type-safety.
    Default ``training`` preserves backward compat with pre-Phase-1.3 records."""

    sub_records: List[Dict[str, Any]] = field(default_factory=list)
    """For ``sweep_aggregate`` records: per-sub-experiment summaries (typically
    {"name": ..., "training_metrics": {...}, "config_diff": {...}}).
    Empty for non-aggregate types."""

    parent_experiment_id: str = ""
    """For ``calibration`` and ``backtest`` records: the experiment_id of the
    upstream record this one depends on (e.g., calibration → its trained model)."""

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
            "record_type": self.record_type,
            "parent_experiment_id": self.parent_experiment_id,
            "retroactive": self.provenance.retroactive,
            # Phase 4 Batch 4c.4: surface feature_set_ref in index for
            # `hft-ops ledger list --feature-set <name>` filtering. Empty dict
            # (not None) when unset, matches other Dict default conventions.
            "feature_set_ref": self.feature_set_ref or {},
        }
