"""build_monitor_table — the single entry point that fuses the ledger rows and
the discovery verdicts into one normalized, filterable ``MonitorRow`` surface,
annotated with drift flags.

Read-only. Torch-free (only the sibling monitor modules + stdlib at module scope).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from .discovery_reader import DiscoveryVerdictReader
from .drift import DriftReport, detect_drift
from .ledger_reader import LedgerReader, LedgerRow


@dataclass(frozen=True)
class MonitorRow:
    """One row of the unified surface — a ledger experiment OR a discovery verdict."""

    kind: str                       # "ledger" | "discovery"
    id: str                         # experiment_id | probe_id
    name: str                       # name | study
    source: str                     # "hft-ops/ledger" | source_tree
    status_or_verdict: str          # status (ledger) | verdict (discovery)
    edge: Optional[bool]            # None (ledger / gate-out) | any_tradeable_edge (discovery)
    authority: str                  # "" (ledger) | verdict_authority
    primary_metric: Optional[float]
    primary_metric_name: str
    dsr: Optional[float]            # deflated_sharpe_ratio (discovery) | None
    provenance_id: str              # experiment_provenance_hash[:12] | config_sha256[:12]
    stats_version: str              # contract_version (ledger) | provenance.hft_metrics_version
    created_at: str
    drift_flags: tuple[str, ...]    # drift kinds whose subject matched this row
    source_path: str


@dataclass(frozen=True)
class MonitorTable:
    rows: tuple[MonitorRow, ...]
    drift: DriftReport
    ledger_read_errors: tuple[str, ...]
    discovery_read_errors: tuple[str, ...]


def _index_findings(findings) -> dict:
    """Index drift findings by ``(subject_kind, subject) -> {kinds}`` so each row
    collects the flags it owns by BOTH its id and its name. F5-BUG-3: fingerprint
    findings are name-keyed, but rows were previously looked up by id only, so the
    name-keyed flags were always dropped (name != experiment_id on every record)."""
    idx: dict = defaultdict(set)
    for f in findings:
        idx[(f.subject_kind, f.subject)].add(f.kind)
    return idx


def _row_drift_flags(idx: dict, *keys) -> tuple:
    """Union the drift kinds across the ``(subject_kind, subject)`` keys a row owns."""
    out: set = set()
    for key in keys:
        out |= idx.get(key, set())
    return tuple(sorted(out))


def _ledger_to_row(row: LedgerRow, idx: dict) -> MonitorRow:
    prov_id = (row.experiment_provenance_hash or row.fingerprint or "")[:12]
    return MonitorRow(
        kind="ledger",
        id=row.experiment_id,
        name=row.name,
        source="hft-ops/ledger",
        status_or_verdict=row.status,
        edge=None,
        authority="",
        primary_metric=row.primary_metric,
        primary_metric_name=row.primary_metric_name,
        dsr=None,
        provenance_id=prov_id,
        stats_version=row.contract_version,
        created_at=row.created_at,
        drift_flags=_row_drift_flags(idx, ("experiment_id", row.experiment_id), ("name", row.name)),
        source_path=row.source_path,
    )


def _verdict_to_row(v, idx: dict) -> MonitorRow:
    prov_id = (v.provenance.config_sha256 or "")[:12]
    return MonitorRow(
        kind="discovery",
        id=v.probe_id,
        name=v.study,
        source=v.source_tree,
        status_or_verdict=v.verdict,
        edge=v.any_tradeable_edge,
        authority=v.verdict_authority,
        primary_metric=None,
        primary_metric_name="",
        dsr=v.deflated_sharpe_ratio,
        provenance_id=prov_id,
        stats_version=v.provenance.hft_metrics_version or "",
        created_at=v.provenance.run_timestamp_utc or "",
        drift_flags=_row_drift_flags(idx, ("probe_id", v.probe_id), ("name", v.study)),
        source_path=v.source_path,
    )


def _merge_discovery(ledger_row: MonitorRow, verdict_row: MonitorRow) -> MonitorRow:
    """Collapse a discovery-origin ledger row + its on-disk discovery verdict
    (matched on the shared config_sha256/fingerprint) into ONE row, so a probe
    that is BOTH a registered ledger record AND a verdict JSON is not double-
    counted. Prefer the ENRICHED ledger row as the base (it carries the
    experiment_id, provenance_id, created_at, contract_version, primary metric);
    adopt the verdict's research columns (the verdict-string + edge + authority +
    dsr) which the ledger projection cannot surface; union the drift flags both
    rows own."""
    return replace(
        ledger_row,
        status_or_verdict=verdict_row.status_or_verdict,
        edge=verdict_row.edge,
        authority=verdict_row.authority,
        dsr=verdict_row.dsr,
        drift_flags=tuple(
            sorted(set(ledger_row.drift_flags) | set(verdict_row.drift_flags))
        ),
    )


def build_monitor_table(
    repo_root: Path | str,
    *,
    ledger_records_dir: Path | str | None = None,
    kind: Optional[str] = None,
    source_tree: Optional[str] = None,
    edge_only: bool = False,
    status: Optional[str] = None,
    name_contains: Optional[str] = None,
    with_drift: bool = True,
) -> MonitorTable:
    repo_root = Path(repo_root)
    if ledger_records_dir is None:
        ledger_records_dir = repo_root / "hft-ops" / "ledger" / "records"
    ledger_records_dir = Path(ledger_records_dir)

    ledger_reader = LedgerReader(ledger_records_dir)
    ledger_rows = ledger_reader.read_all()
    discovery_reader = DiscoveryVerdictReader(repo_root)
    verdicts = discovery_reader.read_all()

    drift = (
        detect_drift(ledger_rows, verdicts, ledger_records_dir=ledger_records_dir)
        if with_drift
        else DriftReport(findings=())
    )
    idx = _index_findings(drift.findings)

    # Dedup (else double-count): a discovery probe registered to the ledger
    # (``record_type="discovery"``, whose ``fingerprint`` IS the probe
    # ``config_sha256`` — set by ``record_from_verdict``) AND present as an
    # on-disk verdict would otherwise appear twice. Collapse the pair on the
    # shared config_sha256/fingerprint: the enriched ledger row absorbs the
    # verdict's verdict-string (+ edge / authority / dsr). Keyed on the FULL
    # 64-hex hash (NOT the ``[:12]`` provenance_id) so there is no truncation
    # collision. ``drift`` is computed above on the RAW (un-deduped) inputs so
    # its findings stay correct; the merged row unions both subjects' flags.
    discovery_fps = {
        r.fingerprint
        for r in ledger_rows
        if r.record_type == "discovery" and r.fingerprint
    }
    verdict_overlay: dict[str, MonitorRow] = {}      # config_sha256 -> verdict row
    standalone_verdict_rows: list[MonitorRow] = []
    for v in verdicts:
        fp = v.provenance.config_sha256
        vrow = _verdict_to_row(v, idx)
        if fp and fp in discovery_fps:
            verdict_overlay[fp] = vrow               # collapsed into its ledger row
        else:
            standalone_verdict_rows.append(vrow)

    rows: list[MonitorRow] = []
    for lr in ledger_rows:
        mrow = _ledger_to_row(lr, idx)
        if lr.record_type == "discovery" and lr.fingerprint in verdict_overlay:
            mrow = _merge_discovery(mrow, verdict_overlay[lr.fingerprint])
        rows.append(mrow)
    rows += standalone_verdict_rows

    if kind:
        rows = [r for r in rows if r.kind == kind]
    if source_tree:
        rows = [r for r in rows if r.source == source_tree]
    if edge_only:
        rows = [r for r in rows if r.edge is True]
    if status:
        rows = [r for r in rows if r.status_or_verdict == status]
    if name_contains:
        needle = name_contains.lower()
        rows = [r for r in rows if needle in r.name.lower()]

    return MonitorTable(
        rows=tuple(rows),
        drift=drift,
        ledger_read_errors=tuple(f"{p}: {e}" for p, e in ledger_reader.read_errors),
        discovery_read_errors=tuple(f"{p}: {e}" for p, e in discovery_reader.read_errors),
    )
