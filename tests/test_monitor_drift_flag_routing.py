"""TDD — F5-BUG-3: drift findings must reach MonitorRow.drift_flags.

A finding's ``subject`` is a *name* (fingerprint_divergence), an *experiment_id*
(schema mismatch on a ledger record), a *probe_id* (verdict drift), or a *path*
(index drift — a global). The table previously looked rows up by id ONLY, so every
name-keyed fingerprint_divergence was silently dropped (name != experiment_id on
all 186 ledger rows). The fix routes by an explicit ``DriftFinding.subject_kind``.
"""

from discovery_verdict import Verdict

from hft_ops.monitor.drift import DriftFinding
from hft_ops.monitor.ledger_reader import LedgerRow
from hft_ops.monitor.table import _index_findings, _ledger_to_row, _verdict_to_row


def _row(**kw):
    base = dict(
        experiment_id="exp-1", name="n", status="completed", record_type="training",
        created_at="2026-01-01T00:00:00+00:00", model_type="tlob", labeling_strategy="tb",
        fingerprint="f" * 64, experiment_provenance_hash="p" * 64,
        compatibility_fingerprint="c" * 64, contract_version="3.0", tags=(),
        primary_metric=0.1, primary_metric_name="test_ic", gate_statuses={},
        source_path="/x.json",
    )
    base.update(kw)
    return LedgerRow(**base)


def _verdict(**kw):
    base = dict(
        probe_id="P_A7", study="s", source_tree="glbx_discovery", source_path="/x.json",
        adapter_name="CommonCoreAdapter", verdict="FAIL", verdict_authority="UNSPECIFIED",
        any_tradeable_edge=False,
    )
    base.update(kw)
    return Verdict(**base)


def test_fingerprint_divergence_name_keyed_flag_reaches_ledger_row():
    # THE BUG: name != experiment_id, so a name-keyed finding never attached.
    findings = [
        DriftFinding("fingerprint_divergence", "warn", "dup-name",
                     "config drift", subject_kind="name"),
        DriftFinding("schema_version_mismatch", "info", "exp-1",
                     "old schema", subject_kind="experiment_id"),
    ]
    idx = _index_findings(findings)
    mrow = _ledger_to_row(_row(experiment_id="exp-1", name="dup-name"), idx)
    assert "fingerprint_divergence" in mrow.drift_flags   # attached by NAME
    assert "schema_version_mismatch" in mrow.drift_flags  # attached by experiment_id


def test_unrelated_row_gets_no_flags():
    findings = [
        DriftFinding("fingerprint_divergence", "warn", "dup-name",
                     "config drift", subject_kind="name"),
    ]
    idx = _index_findings(findings)
    mrow = _ledger_to_row(_row(experiment_id="exp-9", name="other"), idx)
    assert mrow.drift_flags == ()


def test_path_kind_finding_attaches_to_no_row_even_on_name_collision():
    # ledger_index_disk_drift is a global (subject=path); subject_kind routing must
    # keep it off rows even if a row's name happens to equal the path string.
    findings = [
        DriftFinding("ledger_index_disk_drift", "warn", "/x/index.json",
                     "stale envelope", subject_kind="path"),
    ]
    idx = _index_findings(findings)
    mrow = _ledger_to_row(_row(name="/x/index.json"), idx)
    assert mrow.drift_flags == ()


def test_probe_id_keyed_flag_reaches_verdict_row():
    findings = [
        DriftFinding("stale_verdict", "info", "P_A7",
                     "old hft_metrics", subject_kind="probe_id"),
    ]
    idx = _index_findings(findings)
    mrow = _verdict_to_row(_verdict(probe_id="P_A7"), idx)
    assert "stale_verdict" in mrow.drift_flags
