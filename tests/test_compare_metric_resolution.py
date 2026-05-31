"""Step 3 — `compare --metric` fail-loud on an unresolved (typo/unprojected) key.

The `--stages` footgun's twin (cluster #1 added the stages guard). A metric key
that resolves to None in EVERY entry silently renders a blank column —
indistinguishable from "all experiments scored blank". `find_unresolved_metric_keys`
makes it surfaceable so the CLI can WARN (metric keys are an OPEN dotted-path set
-> WARN, not the closed-set RAISE used for `--stages`).
"""

from __future__ import annotations

from hft_ops.ledger.comparator import find_unresolved_metric_keys


def _entry(*, training_metrics=None, backtest_metrics=None):
    """A minimal ledger index entry with nested metric namespaces."""
    return {
        "experiment_id": "e",
        "name": "n",
        "model_type": "tlob",
        "status": "completed",
        "created_at": "2026-05-31",
        "training_metrics": training_metrics or {},
        "backtest_metrics": backtest_metrics or {},
    }


def test_resolved_metric_not_unresolved():
    entries = [_entry(training_metrics={"macro_f1": 0.5})]
    assert find_unresolved_metric_keys(entries, ["training_metrics.macro_f1"]) == []


def test_typo_subkey_is_unresolved():
    entries = [_entry(training_metrics={"macro_f1": 0.5})]
    # 'macrof1' (missing underscore) resolves to None everywhere
    assert find_unresolved_metric_keys(
        entries, ["training_metrics.macrof1"]
    ) == ["training_metrics.macrof1"]


def test_unprojected_namespace_is_unresolved():
    entries = [_entry(training_metrics={"macro_f1": 0.5})]
    assert find_unresolved_metric_keys(
        entries, ["trainng_metrics.macro_f1"]
    ) == ["trainng_metrics.macro_f1"]


def test_resolved_in_at_least_one_entry_is_not_unresolved():
    entries = [
        _entry(training_metrics={}),                 # missing here
        _entry(training_metrics={"macro_f1": 0.7}),  # present here
    ]
    assert find_unresolved_metric_keys(entries, ["training_metrics.macro_f1"]) == []


def test_backtest_metric_resolution():
    entries = [_entry(backtest_metrics={"total_return": 0.02})]
    assert find_unresolved_metric_keys(entries, ["backtest_metrics.total_return"]) == []
    assert find_unresolved_metric_keys(
        entries, ["backtest_metrics.sharpe_ratio"]
    ) == ["backtest_metrics.sharpe_ratio"]


def test_mixed_only_unresolved_returned():
    entries = [_entry(training_metrics={"macro_f1": 0.5},
                      backtest_metrics={"total_return": 0.02})]
    out = find_unresolved_metric_keys(
        entries, ["training_metrics.macro_f1", "backtest_metrics.nope"]
    )
    assert out == ["backtest_metrics.nope"]


def test_empty_entries_all_keys_unresolved():
    # vacuous: nothing to resolve against -> every requested key is "unresolved"
    # (the CLI only WARNs when entries is non-empty, so this never false-alarms).
    assert find_unresolved_metric_keys([], ["training_metrics.macro_f1"]) == [
        "training_metrics.macro_f1"
    ]
