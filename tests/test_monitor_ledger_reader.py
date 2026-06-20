"""TDD — the LedgerReader: hft-ops ledger records/*.json -> List[LedgerRow].

Uses 2 REAL ledger records (a sweep_aggregate + a training) copied into a tmp
ledger, plus a malformed file to prove skip-into-read_errors (never raise).
"""

import shutil
from pathlib import Path

from hft_ops.monitor.ledger_reader import LedgerReader, LedgerRow

FIX = Path(__file__).parent / "fixtures" / "monitor_ledger"
AGG = "cycle10_r19_multi_seed_20260518T222513_aggregate.json"
TRAIN = "cycle10_r19_multi_seed__seed_43_20260518T234754_17678226.json"


def _tmp_ledger(tmp_path, *names, with_broken=False):
    recs = tmp_path / "records"
    recs.mkdir()
    for name in names:
        shutil.copy(FIX / name, recs / name)
    if with_broken:
        (recs / "broken.json").write_text("{ this is not valid json")
    return recs


def test_reads_records_and_projects(tmp_path):
    recs = _tmp_ledger(tmp_path, AGG, TRAIN, with_broken=True)
    reader = LedgerReader(recs)
    rows = reader.read_all()

    assert len(rows) == 2                       # the broken file is skipped, not raised
    assert len(reader.read_errors) == 1
    assert "broken.json" in reader.read_errors[0][0]

    assert all(isinstance(r, LedgerRow) for r in rows)
    by_id = {r.experiment_id: r for r in rows}

    train = by_id["cycle10_r19_multi_seed__seed_43_20260518T234754_17678226"]
    assert train.record_type == "training"
    assert train.model_type == "tlob"
    assert train.labeling_strategy == "triple_barrier"
    assert train.primary_metric is not None      # resolved from training_metrics
    assert train.primary_metric_name             # the resolved metric's name (non-empty)
    assert "cycle10" in train.tags
    assert train.source_path.endswith(TRAIN)

    agg = by_id["cycle10_r19_multi_seed_20260518T222513_aggregate"]
    assert agg.record_type == "sweep_aggregate"


def test_rows_sorted_by_created_at_desc(tmp_path):
    recs = _tmp_ledger(tmp_path, AGG, TRAIN)
    rows = LedgerReader(recs).read_all()
    created = [r.created_at for r in rows]
    assert created == sorted(created, reverse=True)


def test_empty_ledger_returns_empty(tmp_path):
    recs = tmp_path / "records"
    recs.mkdir()
    reader = LedgerReader(recs)
    assert reader.read_all() == []
    assert reader.read_errors == []
