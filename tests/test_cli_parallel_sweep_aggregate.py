"""C1 e2e: the PARALLEL sweep path writes an indexed ``sweep_aggregate`` record.

``cli_parallel_sweep._write_sweep_aggregate`` called a diverged
``SweepAggregateWriter`` signature (``SweepAggregateWriter(paths=...)`` +
positional ``.write(child_summaries)``) — BOTH raise ``TypeError``, which a broad
``except Exception`` swallowed to a ``logger.warning``. Net effect: every
``--parallel > 1`` sweep silently produced ZERO aggregate records. This drives
the FIXED function end-to-end and locks:
  (a) the aggregate record is created with the correct counts + record_type,
  (b) it is indexed / queryable (the post-write ``_rebuild_index`` lands), and
  (c) the fail-loud boundary (R1): a genuine ``OSError`` still WARNs (children
      are already persisted), but a contract drift (``TypeError``) PROPAGATES
      instead of being silently downgraded.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 9 (C1 / R1 / R6.3).
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from hft_ops.cli_parallel_sweep import _write_sweep_aggregate
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.ledger.sweep_aggregate import SweepAggregateWriter
from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest, Stages


@pytest.fixture
def ledger_dir(tmp_path: Path):
    d = tmp_path / "ledger"
    (d / "records").mkdir(parents=True)
    return d


def _child(name, fingerprint, status="completed"):
    return {
        "experiment_id": f"exp_{name}",
        "name": name,
        "fingerprint": fingerprint,
        "axis_values": {},
        "status": status,
        "duration_seconds": 1.0,
        "training_metrics": {},
        "backtest_metrics": {},
    }


def _manifest():
    return ExperimentManifest(
        experiment=ExperimentHeader(name="par_sweep", contract_version="2.2"),
        stages=Stages(),
    )


class TestParallelSweepAggregate:
    def test_aggregate_record_created_with_counts_and_indexed(self, ledger_dir):
        paths = SimpleNamespace(ledger_dir=ledger_dir)
        children = [
            _child("p1", "a" * 64, "completed"),
            _child("p2", "b" * 64, "completed"),
            _child("p3", "c" * 64, "failed"),
        ]
        _write_sweep_aggregate(
            children,
            _manifest(),
            "sweepX_20260531",
            "sweepX",
            paths,
            SweepAggregateWriter,
        )

        # (a) aggregate record file exists with correct type + derived counts
        agg = ledger_dir / "records" / "sweepX_20260531_aggregate.json"
        assert agg.exists(), "parallel path did not write the aggregate record (C1)"
        data = json.loads(agg.read_text())
        assert data["record_type"] == "sweep_aggregate"
        # 2 completed + 1 failed → status "partial"; all 3 carried as sub_records
        assert data["status"] == "partial"
        assert len(data["sub_records"]) == 3

        # (b) indexed / queryable (post-write _rebuild_index landed)
        ledger = ExperimentLedger(ledger_dir)
        entries = [
            e
            for e in ledger.filter(sweep_id="sweepX_20260531")
            if e.get("record_type") == "sweep_aggregate"
        ]
        assert len(entries) == 1, "aggregate not indexed (missing _rebuild_index)"

    def test_genuine_oserror_warns_not_raises(self, ledger_dir, caplog):
        """R6.3: a real I/O error on write is swallowed-to-WARN (children are
        already persisted at this point); a long sweep must not crash."""
        paths = SimpleNamespace(ledger_dir=ledger_dir)

        class _OSErrWriter:
            def write(self, **kwargs):
                raise OSError("disk full")

        caplog.set_level(logging.WARNING)
        # Must NOT raise.
        _write_sweep_aggregate(
            [_child("p1", "a" * 64)], _manifest(), "s_io", "s_io", paths, _OSErrWriter
        )
        assert any(
            "sweep_aggregate write failed" in r.message for r in caplog.records
        ), "expected an I/O-tier WARN, got none"

    def test_contract_drift_typeerror_propagates(self, ledger_dir):
        """R1: a diverged writer signature (TypeError) PROPAGATES with its native
        traceback rather than being silently downgraded to a warning (the C1
        failure mode). Guards against re-introduction of the broad swallow."""
        paths = SimpleNamespace(ledger_dir=ledger_dir)

        class _BadSigWriter:
            def write(self, child_summaries):  # diverged: positional, not keyword
                raise AssertionError("unreachable with the correct keyword call")

        with pytest.raises(TypeError):
            _write_sweep_aggregate(
                [_child("p1", "a" * 64)],
                _manifest(),
                "s_bad",
                "s_bad",
                paths,
                _BadSigWriter,
            )
