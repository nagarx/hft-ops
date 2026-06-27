"""TDD — build_monitor_table: the unified experiment x verdict x provenance x drift
surface. Ledger rows and discovery verdicts both normalize into MonitorRow; filters
(kind / source_tree / edge_only / status / name_contains) apply uniformly.
"""

import shutil
from pathlib import Path

import pytest

from hft_contracts.experiment_record import ExperimentRecord
from hft_ops.monitor.table import MonitorRow, MonitorTable, build_monitor_table

# config_sha256 of the tradeflow_esnq_P_A7.json discovery fixture (its
# provenance.config_sha256 — the probe treatment identity).
PA7_CONFIG_SHA = "a5ab0a3f2281f06d39649d9cdf2c9fd29eda7f9c3338bb968454e231cfc96290"

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


def test_discovery_probe_registered_and_verdict_dedup(tmp_path):
    """A discovery probe that is BOTH a registered ledger record
    (record_type="discovery", fingerprint == config_sha256) AND an on-disk
    verdict collapses to ONE row — the enriched ledger row keeps the verdict's
    verdict-string + edge. No double-count, no leftover discovery-kind row."""
    recs = tmp_path / "ledger" / "records"
    recs.mkdir(parents=True)
    rec = ExperimentRecord(
        experiment_id="P_A7_20260605T000000_a5ab0a3f",
        name="P_A7",
        fingerprint=PA7_CONFIG_SHA,
        record_type="discovery",
        status="completed",
        stages_completed=["discovery"],
        contract_version="3.0",
        training_metrics={"verdict": "FAIL", "any_tradeable_edge": False},
    )
    rec.save(recs / f"{rec.experiment_id}.json")
    disc = tmp_path / "glbx_discovery" / "results"
    disc.mkdir(parents=True)
    shutil.copy(DFIX / "tradeflow_esnq_P_A7.json", disc / "tradeflow_esnq_P_A7.json")

    table = build_monitor_table(tmp_path, ledger_records_dir=recs)

    # exactly ONE row references this probe's config_sha256
    matched = [r for r in table.rows if r.provenance_id == PA7_CONFIG_SHA[:12]]
    assert len(matched) == 1, f"probe double-counted: {matched}"
    row = matched[0]
    assert row.kind == "ledger"                              # enriched ledger row preferred
    assert row.id == "P_A7_20260605T000000_a5ab0a3f"
    assert row.status_or_verdict == "FAIL"                   # verdict-string kept
    assert row.edge is False                                 # verdict edge adopted
    assert row.dsr == pytest.approx(0.7880377956176312)      # verdict dsr adopted
    # no surviving standalone discovery row for the same probe
    assert not any(
        r.kind == "discovery" and Path(r.source_path).name == "tradeflow_esnq_P_A7.json"
        for r in table.rows
    )


def test_discovery_verdict_without_ledger_record_stays_standalone(tmp_path):
    """Sanity: when NO discovery-origin ledger record claims the probe, its
    verdict still appears as a standalone discovery row (no over-dedup)."""
    recs = tmp_path / "ledger" / "records"
    recs.mkdir(parents=True)
    disc = tmp_path / "glbx_discovery" / "results"
    disc.mkdir(parents=True)
    shutil.copy(DFIX / "tradeflow_esnq_P_A7.json", disc / "tradeflow_esnq_P_A7.json")

    table = build_monitor_table(tmp_path, ledger_records_dir=recs)
    disc_rows = [r for r in table.rows if r.kind == "discovery"]
    assert len(disc_rows) == 1
    assert disc_rows[0].id == "P_A7"
    assert disc_rows[0].status_or_verdict == "FAIL"
