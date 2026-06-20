"""TDD — drift detection over the ledger rows + discovery verdicts.

Four concrete 'drift' notions: (1) the stale index.json envelope vs the records
on disk, (2) fingerprint divergence (same name / different config or provenance),
(3) a verdict produced under an older hft_metrics than installed, (4) a record /
verdict on a non-current schema. None are errors by default — they are observations.
"""

import json

from discovery_verdict import Verdict, VerdictProvenance

from hft_ops.monitor.drift import (
    DriftReport,
    detect_drift,
    fingerprint_divergence,
    ledger_index_disk_drift,
    schema_version_mismatch_drift,
    stale_verdict_drift,
)
from hft_ops.monitor.ledger_reader import LedgerRow


def _row(**kw):
    base = dict(
        experiment_id="e", name="n", status="completed", record_type="training",
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
        probe_id="P", study="s", source_tree="glbx_discovery", source_path="/x.json",
        adapter_name="CommonCoreAdapter", verdict="FAIL", verdict_authority="UNSPECIFIED",
        any_tradeable_edge=False,
    )
    base.update(kw)
    return Verdict(**base)


def test_ledger_index_disk_count_drift(tmp_path):
    recs = tmp_path / "records"
    recs.mkdir()
    for i in range(3):
        (recs / f"r{i}.json").write_text("{}")
    (tmp_path / "index.json").write_text(json.dumps({"entries": [{"experiment_id": "only-1"}], "schema": "1.6.0"}))
    findings = ledger_index_disk_drift(recs)
    assert any(f.kind == "ledger_index_disk_drift" and f.severity == "warn" for f in findings)


def test_no_index_is_not_an_error(tmp_path):
    recs = tmp_path / "records"
    recs.mkdir()
    (recs / "r.json").write_text("{}")
    findings = ledger_index_disk_drift(recs)
    assert all(f.severity != "error" for f in findings)


def test_fingerprint_divergence_same_name_diff_fingerprint():
    rows = [_row(name="dup", fingerprint="a" * 64), _row(name="dup", fingerprint="b" * 64)]
    findings = fingerprint_divergence(rows)
    assert any(f.kind == "fingerprint_divergence" for f in findings)


def test_no_fingerprint_divergence_when_consistent():
    rows = [_row(name="ok", fingerprint="a" * 64), _row(name="ok", fingerprint="a" * 64)]
    assert fingerprint_divergence(rows) == []


def test_schema_version_mismatch_is_info():
    rows = [_row(contract_version="2.2")]
    findings = schema_version_mismatch_drift(rows, [], current_contract_version="3.0")
    assert any(f.kind == "schema_version_mismatch" and f.severity == "info" for f in findings)


def test_stale_verdict_version_is_info():
    v = _verdict(provenance=VerdictProvenance(hft_metrics_version="0.1.15"))
    findings = stale_verdict_drift([v], installed_version="0.1.23")
    assert any(f.kind == "stale_verdict" for f in findings)


def test_stale_verdict_none_when_current():
    v = _verdict(provenance=VerdictProvenance(hft_metrics_version="0.1.23"))
    assert stale_verdict_drift([v], installed_version="0.1.23") == []


def test_detect_drift_aggregates(tmp_path):
    recs = tmp_path / "records"
    recs.mkdir()
    for i in range(3):
        (recs / f"r{i}.json").write_text("{}")
    (tmp_path / "index.json").write_text(json.dumps({"entries": [], "schema": "1.6.0"}))
    rows = [_row(name="dup", fingerprint="a" * 64), _row(name="dup", fingerprint="b" * 64)]
    report = detect_drift(
        rows, [], ledger_records_dir=recs,
        current_contract_version="3.0", installed_hft_metrics_version="0.1.23",
    )
    assert isinstance(report, DriftReport)
    assert report.n_warn >= 1
    kinds = {f.kind for f in report.findings}
    assert "ledger_index_disk_drift" in kinds
    assert "fingerprint_divergence" in kinds
