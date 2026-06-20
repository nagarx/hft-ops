"""TDD — the monitor renderers (text / markdown / json + drift)."""

import json

from hft_ops.monitor.drift import DriftFinding, DriftReport
from hft_ops.monitor.render import (
    render_drift_json,
    render_drift_text,
    render_json,
    render_markdown,
    render_text,
)
from hft_ops.monitor.table import MonitorRow, MonitorTable


def _table():
    rows = (
        MonitorRow(
            kind="discovery", id="P_A7", name="tradeflow", source="glbx_discovery",
            status_or_verdict="FAIL", edge=False, authority="UNSPECIFIED", primary_metric=None,
            primary_metric_name="", dsr=0.788, provenance_id="abc123", stats_version="0.1.18",
            created_at="2026-06-05T00:00:00+00:00", drift_flags=(), source_path="/x.json",
        ),
        MonitorRow(
            kind="ledger", id="exp1", name="cycle10", source="hft-ops/ledger",
            status_or_verdict="completed", edge=None, authority="", primary_metric=0.42,
            primary_metric_name="test_macro_f1", dsr=None, provenance_id="def456",
            stats_version="3.0", created_at="2026-05-19T00:00:00+00:00",
            drift_flags=("fingerprint_divergence",), source_path="/y.json",
        ),
    )
    return MonitorTable(rows=rows, drift=DriftReport(()), ledger_read_errors=(), discovery_read_errors=())


def test_render_json_roundtrips():
    data = json.loads(render_json(_table()))
    assert len(data["rows"]) == 2
    assert {r["id"] for r in data["rows"]} == {"P_A7", "exp1"}
    # edge None vs False must survive the JSON surface (load-bearing distinction)
    by_id = {r["id"]: r for r in data["rows"]}
    assert by_id["P_A7"]["edge"] is False
    assert by_id["exp1"]["edge"] is None


def test_render_markdown_has_header_and_rows():
    s = render_markdown(_table())
    assert s.startswith("|")
    assert "P_A7" in s and "cycle10" in s


def test_render_text_is_string_with_ids():
    s = render_text(_table())
    assert isinstance(s, str)
    assert "P_A7" in s and "cycle10" in s


def test_render_drift_text_and_json():
    rep = DriftReport((DriftFinding("ledger_index_disk_drift", "warn", "idx", "stale envelope"),))
    assert "stale envelope" in render_drift_text(rep)
    data = json.loads(render_drift_json(rep))
    assert data["n_warn"] == 1
    assert data["findings"][0]["kind"] == "ledger_index_disk_drift"
