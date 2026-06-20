"""Drift detection over the unified monitor inputs (ledger rows + discovery verdicts).

'Drift' here is concrete and observational — never an auto-fix:
  1. ledger_index_disk_drift   — the cached index.json envelope disagrees with records/* on disk
  2. fingerprint_divergence    — same experiment name, different config fingerprint (or same
                                 config, different provenance hash = data/feature/model drift)
  3. stale_verdict             — a discovery verdict produced under an older hft_metrics than installed
  4. schema_version_mismatch   — a record on a non-current contract_version, or a verdict that
                                 only normalized with parse warnings (a new divergent harness shape)

Read-only: ledger_index_disk_drift REPORTS the stale envelope; it never rebuilds it.
Torch-free: stdlib only at module scope; the version/schema lookups are lazy (inside functions).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class DriftFinding:
    kind: str
    severity: str       # "info" | "warn" | "error"
    subject: str        # the value to match a row against
    detail: str
    subject_kind: str   # "name" | "experiment_id" | "probe_id" | "path" — how `subject`
    #                     routes to a MonitorRow. F5-BUG-3: name-keyed findings
    #                     (fingerprint_divergence) were dropped because the table only
    #                     looked rows up by id; routing by subject_kind fixes it.


@dataclass(frozen=True)
class DriftReport:
    findings: tuple[DriftFinding, ...]

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warn")

    def by_kind(self, kind: str) -> tuple[DriftFinding, ...]:
        return tuple(f for f in self.findings if f.kind == kind)


def _ver_tuple(s) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple. A pre-release / dev / rc
    suffix (any non-numeric trailing characters in a component, e.g. ``0.1.24-dev``,
    ``0.1.24.dev0``, ``0.1.24rc1``) sorts BEFORE the corresponding final release
    (``0.1.24``): the numeric release parts are extracted, then a final marker —
    ``0`` for a pre-release, ``1`` for a final release — is appended so
    ``0.1.24-dev < 0.1.24`` (M2: the prior strip-to-digits made them compare EQUAL,
    so a stale dev-build verdict was never flagged). Assumes a consistent component
    count across the versions compared (true for a single package's version history,
    which is the only caller — ``hft-metrics``)."""
    out: list[int] = []
    is_prerelease = False
    for part in str(s).split("."):
        # the LEADING digit run is this component's release number; anything after it
        # (a `-dev` / `rc1` / `dev0` tag) marks a pre-release. Joining ALL digits was
        # the bug — it pulled the `1` out of `24rc1` and read 241.
        i = 0
        while i < len(part) and part[i].isdigit():
            i += 1
        if part[i:]:  # a non-digit tail -> pre-release tag
            is_prerelease = True
        out.append(int(part[:i]) if part[:i] else 0)
    out.append(0 if is_prerelease else 1)
    return tuple(out)


def current_hft_metrics_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("hft-metrics")
    except Exception:
        return None


def current_schema_version() -> Optional[str]:
    try:
        from hft_contracts import SCHEMA_VERSION
        return str(SCHEMA_VERSION)
    except Exception:
        return None


# --- the four checks (pure functions) -------------------------------------------

def ledger_index_disk_drift(records_dir: Path | str) -> list[DriftFinding]:
    records_dir = Path(records_dir)
    disk_count = len(list(records_dir.glob("*.json")))
    index_path = records_dir.parent / "index.json"
    if not index_path.exists():
        return [DriftFinding("ledger_index_disk_drift", "info", str(index_path),
                             f"no index.json envelope; {disk_count} records on disk",
                             subject_kind="path")]
    try:
        env = json.loads(index_path.read_text())
    except Exception as exc:
        return [DriftFinding("ledger_index_disk_drift", "warn", str(index_path),
                             f"index.json unreadable ({type(exc).__name__}); {disk_count} records on disk",
                             subject_kind="path")]
    entries = env.get("entries") if isinstance(env, dict) else env
    idx_count = len(entries) if isinstance(entries, (list, dict)) else 0
    if idx_count != disk_count:
        return [DriftFinding("ledger_index_disk_drift", "warn", str(index_path),
                             f"index.json caches {idx_count} entries but {disk_count} records on disk "
                             f"(stale envelope — the monitor reads records/* directly)",
                             subject_kind="path")]
    return []


# sweep roll-ups: their fingerprints differ by sweep SIZE, not config -> not drift.
_AGGREGATE_RECORD_TYPES = frozenset({"sweep_aggregate", "sweep_failure"})


def fingerprint_divergence(ledger_rows: Sequence) -> list[DriftFinding]:
    """Flag a name whose COMPLETED, non-aggregate records carry > 1 distinct config
    fingerprint (genuine config drift between re-runs) or > 1 provenance hash under
    one fingerprint (data/feature/model drift under one config).

    F5-BUG-2: only completed, non-aggregate records can constitute genuine drift.
    Excluding non-completed runs (failed/partial debugging re-runs — e.g.
    ``e5_60s_importance_audit``, 5/6 failed) and ``sweep_aggregate`` roll-ups
    (different sweep sizes -> different fingerprints by construction — e.g. the bare
    ``cycle5_multi_arm`` name) removes the two documented false-positive classes
    without losing genuine completed-record drift (e.g. the ``cycle5_multi_arm``
    per-arm ``temporal_ridge`` records, which still flag)."""
    findings: list[DriftFinding] = []
    by_name: dict[str, list] = defaultdict(list)
    for row in ledger_rows:
        if row.status != "completed":
            continue
        if row.record_type in _AGGREGATE_RECORD_TYPES:
            continue
        by_name[row.name].append(row)
    for name, group in by_name.items():
        fps = {r.fingerprint for r in group if r.fingerprint}
        if len(fps) > 1:
            findings.append(DriftFinding(
                "fingerprint_divergence", "warn", name,
                f"{len(group)} records share name {name!r} but have {len(fps)} distinct "
                f"fingerprints (config drift between re-runs)",
                subject_kind="name",
            ))
        by_fp: dict[str, set] = defaultdict(set)
        for r in group:
            if r.fingerprint and r.experiment_provenance_hash:
                by_fp[r.fingerprint].add(r.experiment_provenance_hash)
        for fp, phs in by_fp.items():
            if len(phs) > 1:
                findings.append(DriftFinding(
                    "fingerprint_divergence", "warn", name,
                    f"name {name!r} fingerprint {fp[:12]} has {len(phs)} distinct "
                    f"experiment_provenance_hash (data/feature/model drift under one config)",
                    subject_kind="name",
                ))
    return findings


def stale_verdict_drift(verdicts: Sequence, installed_version: Optional[str]) -> list[DriftFinding]:
    if not installed_version:
        return []
    findings: list[DriftFinding] = []
    for v in verdicts:
        ver = v.provenance.hft_metrics_version
        if ver and _ver_tuple(ver) < _ver_tuple(installed_version):
            findings.append(DriftFinding(
                "stale_verdict", "info", v.probe_id,
                f"verdict produced under hft_metrics {ver} < installed {installed_version}; "
                f"re-validate per hft-rules §9 measurement-context",
                subject_kind="probe_id",
            ))
    return findings


def schema_version_mismatch_drift(
    ledger_rows: Sequence, verdicts: Sequence, current_contract_version: Optional[str]
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    if current_contract_version:
        for r in ledger_rows:
            if r.contract_version and r.contract_version != current_contract_version:
                findings.append(DriftFinding(
                    "schema_version_mismatch", "info", r.experiment_id,
                    f"record contract_version {r.contract_version} != current {current_contract_version}",
                    subject_kind="experiment_id",
                ))
    for v in verdicts:
        if v.parse_warnings:
            findings.append(DriftFinding(
                "schema_version_mismatch", "info", v.probe_id,
                f"discovery verdict normalized with warnings: {'; '.join(v.parse_warnings)}",
                subject_kind="probe_id",
            ))
    return findings


def detect_drift(
    ledger_rows: Sequence,
    verdicts: Sequence,
    *,
    ledger_records_dir: Path | str,
    current_contract_version: Optional[str] = None,
    installed_hft_metrics_version: Optional[str] = None,
) -> DriftReport:
    if current_contract_version is None:
        current_contract_version = current_schema_version()
    if installed_hft_metrics_version is None:
        installed_hft_metrics_version = current_hft_metrics_version()
    findings: list[DriftFinding] = []
    findings += ledger_index_disk_drift(ledger_records_dir)
    findings += fingerprint_divergence(ledger_rows)
    findings += stale_verdict_drift(verdicts, installed_hft_metrics_version)
    findings += schema_version_mismatch_drift(ledger_rows, verdicts, current_contract_version)
    return DriftReport(findings=tuple(findings))
