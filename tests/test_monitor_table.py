"""TDD — build_monitor_table: the unified experiment x verdict x provenance x drift
surface. Ledger rows and discovery verdicts both normalize into MonitorRow; filters
(kind / source_tree / edge_only / status / name_contains) apply uniformly.
"""

import shutil
from pathlib import Path

import pytest

from hft_ops.monitor.table import MonitorRow, MonitorTable, build_monitor_table

LFIX = Path(__file__).parent / "fixtures" / "monitor_ledger"
DFIX = Path(__file__).parent / "fixtures" / "monitor_discovery"
AGG = "cycle10_r19_multi_seed_20260518T222513_aggregate.json"
TRAIN = "cycle10_r19_multi_seed__seed_43_20260518T234754_17678226.json"


def _repo(tmp_path):
    recs = tmp_path / "ledger" / "records"
    recs.mkdir(parents=True)
    for n in (AGG, TRAIN):
        shutil.copy(LFIX / n, recs / n)
    disc = tmp_path / "glbx_discovery" / "results"
    disc.mkdir(parents=True)
    shutil.copy(DFIX / "tradeflow_esnq_P_A7.json", disc / "tradeflow_esnq_P_A7.json")
    return recs


def test_build_unified_table_both_kinds(tmp_path):
    recs = _repo(tmp_path)
    table = build_monitor_table(tmp_path, ledger_records_dir=recs)
    assert isinstance(table, MonitorTable)
    assert {r.kind for r in table.rows} == {"ledger", "discovery"}
    assert all(isinstance(r, MonitorRow) for r in table.rows)

    disc = next(r for r in table.rows if r.kind == "discovery")
    assert disc.id == "P_A7"
    assert disc.source == "glbx_discovery"
    assert disc.status_or_verdict == "FAIL"
    assert disc.edge is False
    assert disc.dsr == pytest.approx(0.7880377956176312)
    assert disc.stats_version                       # hft_metrics_version from provenance

    led = next(r for r in table.rows if r.kind == "ledger")
    assert led.source == "hft-ops/ledger"
    assert led.edge is None                         # ledger rows never answer the edge question


def test_edge_only_filter_drops_non_edge(tmp_path):
    recs = _repo(tmp_path)
    table = build_monitor_table(tmp_path, ledger_records_dir=recs, edge_only=True)
    # tradeflow edge=False; ledger rows edge=None -> all dropped
    assert all(r.edge is True for r in table.rows)


def test_kind_filter(tmp_path):
    recs = _repo(tmp_path)
    table = build_monitor_table(tmp_path, ledger_records_dir=recs, kind="discovery")
    assert all(r.kind == "discovery" for r in table.rows)
    assert len(table.rows) == 1


def test_name_contains_filter(tmp_path):
    recs = _repo(tmp_path)
    table = build_monitor_table(tmp_path, ledger_records_dir=recs, kind="ledger", name_contains="cycle10")
    assert len(table.rows) >= 1
    assert all("cycle10" in r.name.lower() for r in table.rows)
