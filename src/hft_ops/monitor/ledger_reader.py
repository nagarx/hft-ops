"""LedgerReader — projects the hft-ops experiment ledger (``records/*.json``)
into flat ``LedgerRow``s for the unified monitor surface.

Torch-free: imports ONLY ``hft_contracts.experiment_record`` (pure-python) + stdlib.
Read-only: never writes or rebuilds ``index.json`` (the stale envelope is ignored;
we read the 186 record files directly, matching the ledger's own skip-malformed policy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hft_contracts.experiment_record import ExperimentRecord

# primary-metric resolution, by record_type. Priority lists are "most-informative first".
_TRAINING_METRIC_PRIORITY = (
    "test_ic",          # regression (the edge metric)
    "best_val_ic",
    "test_macro_f1",    # classification
    "best_val_macro_f1",
    "test_accuracy",
    "best_val_accuracy",
)
_BACKTEST_METRIC_PRIORITY = (
    "total_return",
    "option_return",
    "sharpe_ratio",
    "sharpe",
    "win_rate",
)


def _first_finite(metrics: Mapping[str, Any], priority) -> tuple[Optional[float], str]:
    for key in priority:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return float(v), key
    return None, ""


def _resolve_primary_metric(ie: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """Resolve the single headline scalar for a record, dispatched on record_type.
    Returns (value, metric_name); (None, "") for aggregate/analysis/calibration records."""
    rt = ie.get("record_type", "")
    if rt in ("training", "evaluation"):
        return _first_finite(ie.get("training_metrics") or {}, _TRAINING_METRIC_PRIORITY)
    if rt in ("backtest", "backtesting"):
        return _first_finite(ie.get("backtest_metrics") or {}, _BACKTEST_METRIC_PRIORITY)
    return None, ""


@dataclass(frozen=True)
class LedgerRow:
    """Flat projection of one ExperimentRecord (from ``index_entry()``) for the monitor table."""

    experiment_id: str
    name: str
    status: str
    record_type: str
    created_at: str
    model_type: str
    labeling_strategy: str
    fingerprint: str
    experiment_provenance_hash: str
    compatibility_fingerprint: str
    contract_version: str
    tags: tuple[str, ...]
    primary_metric: Optional[float]
    primary_metric_name: str
    gate_statuses: Mapping[str, str]      # {stage: status} from gate_reports
    source_path: str


class LedgerReader:
    """Reads ``<ledger_records_dir>/*.json`` -> ``list[LedgerRow]`` (newest first).

    Robust by design: a malformed / unloadable record is recorded into
    ``read_errors`` (a list of ``(path, error)``) and skipped, never raised —
    mirroring ``ExperimentLedger`` skip-malformed behavior. Read-only.
    """

    def __init__(self, ledger_records_dir: Path | str):
        self.records_dir = Path(ledger_records_dir)
        self.read_errors: list[tuple[str, str]] = []

    def read_all(self) -> list[LedgerRow]:
        self.read_errors = []
        rows: list[LedgerRow] = []
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                record = ExperimentRecord.load(str(path))
                ie = record.index_entry()
            except Exception as exc:  # malformed JSON / schema mismatch -> skip + record
                self.read_errors.append((str(path), f"{type(exc).__name__}: {exc}"))
                continue
            rows.append(self._project(ie, str(path)))
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return rows

    @staticmethod
    def _project(ie: Mapping[str, Any], source_path: str) -> LedgerRow:
        primary, primary_name = _resolve_primary_metric(ie)
        gate_statuses: dict[str, str] = {}
        gate_reports = ie.get("gate_reports")
        if isinstance(gate_reports, dict):
            for stage, report in gate_reports.items():
                if isinstance(report, dict) and "status" in report:
                    gate_statuses[stage] = str(report["status"])
        return LedgerRow(
            experiment_id=str(ie.get("experiment_id", "")),
            name=str(ie.get("name", "")),
            status=str(ie.get("status", "")),
            record_type=str(ie.get("record_type", "")),
            created_at=str(ie.get("created_at", "")),
            model_type=str(ie.get("model_type", "")),
            labeling_strategy=str(ie.get("labeling_strategy", "")),
            fingerprint=str(ie.get("fingerprint", "")),
            experiment_provenance_hash=str(ie.get("experiment_provenance_hash", "")),
            compatibility_fingerprint=str(ie.get("compatibility_fingerprint", "")),
            contract_version=str(ie.get("contract_version", "")),
            tags=tuple(ie.get("tags") or ()),
            primary_metric=primary,
            primary_metric_name=primary_name,
            gate_statuses=gate_statuses,
            source_path=source_path,
        )
