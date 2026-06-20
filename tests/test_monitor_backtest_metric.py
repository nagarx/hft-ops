"""TDD — F5-BUG-4 (consumer-scope): _BACKTEST_METRIC_PRIORITY must contain ONLY
keys that ExperimentRecord.index_entry() actually projects, else they are dead
(``_first_finite`` does an exact ``metrics.get(key)``, so a key the producer never
emits can never resolve). Ground-truthed against the REAL producer projection, not
a hardcoded mirror — this is the cross-module contract (hft-rules §1/§6).

(The deeper "realized backtest P&L is never surfaced" half of F5-BUG-4 is
PRODUCER-side — no record is typed ``backtest`` and training records carry
``backtest_metrics={}`` — tracked separately; the monitor cannot manufacture data
the ledger does not store.)
"""

import dataclasses
from pathlib import Path

from hft_contracts.experiment_record import ExperimentRecord

from hft_ops.monitor.ledger_reader import _BACKTEST_METRIC_PRIORITY, _resolve_primary_metric

LFIX = Path(__file__).parent / "fixtures" / "monitor_ledger"
TRAIN = "cycle10_r19_multi_seed__seed_43_20260518T234754_17678226.json"


def test_no_dead_backtest_priority_keys_vs_index_entry():
    rec = ExperimentRecord.load(str(LFIX / TRAIN))
    stuffed = dataclasses.replace(
        rec, backtest_metrics={k: 1.0 for k in _BACKTEST_METRIC_PRIORITY}
    )
    projected = stuffed.index_entry().get("backtest_metrics", {})
    dead = [k for k in _BACKTEST_METRIC_PRIORITY if k not in projected]
    assert not dead, f"dead backtest priority keys (index_entry never projects them): {dead}"


def test_resolve_backtest_primary_prefers_total_return():
    ie = {
        "record_type": "backtest",
        "backtest_metrics": {"win_rate": 0.5, "sharpe_ratio": 1.2, "total_return": 0.05},
    }
    assert _resolve_primary_metric(ie) == (0.05, "total_return")


def test_resolve_backtest_primary_falls_through_to_next_live_key():
    ie = {"record_type": "backtesting", "backtest_metrics": {"win_rate": 0.61}}
    assert _resolve_primary_metric(ie) == (0.61, "win_rate")
