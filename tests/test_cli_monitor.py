"""TDD — the `hft-ops monitor` CLI group (table / drift)."""

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from hft_ops.cli import main

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


def test_monitor_table_json(tmp_path):
    recs = _repo(tmp_path)
    r = CliRunner().invoke(
        main,
        ["monitor", "table", "--repo-root", str(tmp_path),
         "--ledger-records-dir", str(recs), "--format", "json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert {row["kind"] for row in data["rows"]} == {"ledger", "discovery"}


def test_monitor_table_edge_only(tmp_path):
    recs = _repo(tmp_path)
    r = CliRunner().invoke(
        main,
        ["monitor", "table", "--repo-root", str(tmp_path),
         "--ledger-records-dir", str(recs), "--edge-only", "--format", "json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert all(row["edge"] is True for row in data["rows"])


def test_monitor_drift_fail_on_warn_exits_1(tmp_path):
    recs = _repo(tmp_path)
    (tmp_path / "ledger" / "index.json").write_text(json.dumps({"entries": [], "schema": "1.6.0"}))
    r = CliRunner().invoke(
        main,
        ["monitor", "drift", "--repo-root", str(tmp_path),
         "--ledger-records-dir", str(recs), "--fail-on", "warn"],
    )
    assert r.exit_code == 1, r.output


def test_monitor_drift_clean_exits_0(tmp_path):
    recs = _repo(tmp_path)
    # index.json matching the 2 disk records -> no index drift; the 2 records have
    # distinct names -> no fingerprint divergence; only info-level (if any) remains.
    (tmp_path / "ledger" / "index.json").write_text(
        json.dumps({"entries": [{"id": 1}, {"id": 2}], "schema": "1.6.0"})
    )
    r = CliRunner().invoke(
        main,
        ["monitor", "drift", "--repo-root", str(tmp_path),
         "--ledger-records-dir", str(recs), "--fail-on", "error", "--format", "json"],
    )
    assert r.exit_code == 0, r.output
