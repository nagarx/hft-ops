"""Phase V.B.4b: `compare_sweep_statistical` adapter unit tests.

Locks the hft-ops→hft-metrics adapter contract:
  T1: happy path (3 paired records → 3 PairwiseResults, correct shape).
  T2: unpaired labels (different regression_labels.npy per record) → ValueError.
  T3: < 2 records → ValueError.
  T4: unknown metric → ValueError.
  T5: missing signal file → ValueError.

Uses mock-based signal-dir resolution to isolate the adapter logic from the
manifest loader (integration coverage of that path is Agent-4 add-on #148).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from hft_metrics.pairwise import PairwiseResult

from hft_ops.ledger.statistical_compare import (
    _STATISTIC_FN_REGISTRY,
    compare_sweep_statistical,
)


def _write_record_signals(
    base_dir: Path,
    *,
    record_id: str,
    regression_labels: np.ndarray,
    predicted_returns: np.ndarray,
    primary_horizon_idx: int = 0,
) -> Path:
    """Write synthetic signal files for one record under ``base_dir/<record_id>/``.

    Returns the signal directory path (to be returned by the mocked
    ``_resolve_signal_dir``).
    """
    sig_dir = base_dir / record_id
    sig_dir.mkdir(parents=True, exist_ok=True)
    np.save(sig_dir / "regression_labels.npy", regression_labels)
    np.save(sig_dir / "predicted_returns.npy", predicted_returns)
    (sig_dir / "signal_metadata.json").write_text(json.dumps({
        "signal_type": "regression",
        "primary_horizon_idx": primary_horizon_idx,
        "horizons": [10, 60, 300],
    }))
    return sig_dir


def _make_entries(record_ids: List[str]) -> List[dict]:
    """Construct minimal ledger index entries for the given record IDs."""
    return [
        {
            "experiment_id": rid,
            "name": f"exp_{rid}",
            "record_type": "experiment",
            "axis_values": {"seed": rid.split("_")[-1]},
        }
        for rid in record_ids
    ]


# =============================================================================
# Test 1: happy path
# =============================================================================


class TestHappyPath:
    """T1: 3 records with paired labels → 3 PairwiseResults."""

    def test_returns_expected_pair_count_and_labels(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        n = 200
        shared_labels = rng.standard_normal((n, 3))  # (N, H=3), shared across records

        record_ids = ["rec_42", "rec_43", "rec_44"]
        signal_dirs = {}
        for rid in record_ids:
            # Each record has DIFFERENT predictions (treatment variety)
            # but IDENTICAL labels (paired).
            preds = rng.standard_normal((n, 3)) + 0.2 * shared_labels
            signal_dirs[rid] = _write_record_signals(
                tmp_path,
                record_id=rid,
                regression_labels=shared_labels,
                predicted_returns=preds,
            )

        entries = _make_entries(record_ids)

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            results, labels = compare_sweep_statistical(
                entries,
                ledger=None,   # mock bypasses ledger usage
                paths=None,    # mock bypasses paths usage
                metric="val_ic",
                n_bootstraps=200,   # small n_bootstraps for fast test
                seed=7,
            )

        # K=3 treatments → 3 pairs: (0,1), (0,2), (1,2)
        assert len(results) == 3, f"Expected 3 pairs, got {len(results)}"
        assert len(labels) == 3, f"Expected 3 labels, got {len(labels)}"

        # Every result is a PairwiseResult with well-formed scalars
        for r in results:
            assert isinstance(r, PairwiseResult)
            assert 0 <= r.i < r.j <= 2
            assert 0.0 <= r.p_value_raw <= 1.0
            assert 0.0 <= r.p_value_bh <= 1.0
            assert r.ci_lower <= r.ci_upper
            assert r.n_bootstraps == 200
            assert r.seed == 7

        # Labels carry axis_values (from our _make_entries fixture)
        # Format: "seed=42" / "seed=43" / "seed=44"
        assert labels[0] == "seed=42"
        assert labels[1] == "seed=43"
        assert labels[2] == "seed=44"


# =============================================================================
# Test 2: unpaired labels raise
# =============================================================================


class TestUnpairedLabels:
    """T2: different regression_labels.npy per record → ValueError."""

    def test_different_labels_raise(self, tmp_path: Path):
        rng = np.random.default_rng(100)
        n = 150

        # Record A has one set of labels; record B has DIFFERENT labels.
        labels_a = rng.standard_normal((n, 3))
        labels_b = rng.standard_normal((n, 3))   # different bytes
        preds = rng.standard_normal((n, 3))

        sig_a = _write_record_signals(
            tmp_path, record_id="rec_a",
            regression_labels=labels_a, predicted_returns=preds,
        )
        sig_b = _write_record_signals(
            tmp_path, record_id="rec_b",
            regression_labels=labels_b, predicted_returns=preds,
        )

        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(ValueError, match=r"unpaired labels detected"):
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=100,
                )

    def test_different_primary_horizon_raises(self, tmp_path: Path):
        """Same labels, different primary_horizon_idx → still unpaired."""
        rng = np.random.default_rng(101)
        n = 150
        shared_labels = rng.standard_normal((n, 3))
        preds = rng.standard_normal((n, 3))

        sig_a = _write_record_signals(
            tmp_path, record_id="rec_a",
            regression_labels=shared_labels, predicted_returns=preds,
            primary_horizon_idx=0,
        )
        sig_b = _write_record_signals(
            tmp_path, record_id="rec_b",
            regression_labels=shared_labels, predicted_returns=preds,
            primary_horizon_idx=1,   # different horizon
        )

        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(ValueError, match=r"primary_horizon_idx differs"):
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=100,
                )


# =============================================================================
# Test 3: < 2 records raises
# =============================================================================


class TestInsufficientRecords:
    """T3: < 2 child records → ValueError (nothing to compare)."""

    def test_zero_records_raises(self):
        with pytest.raises(ValueError, match=r"need >= 2 child records"):
            compare_sweep_statistical([], ledger=None, paths=None)

    def test_one_record_raises(self):
        entries = _make_entries(["only_one"])
        with pytest.raises(ValueError, match=r"need >= 2 child records"):
            compare_sweep_statistical(entries, ledger=None, paths=None)


# =============================================================================
# Test 4: unknown metric raises
# =============================================================================


class TestUnknownMetric:
    """T4: unsupported --metric value → ValueError with actionable hint."""

    def test_unknown_metric_raises(self):
        entries = _make_entries(["a", "b"])
        with pytest.raises(ValueError, match=r"unsupported metric 'foo_metric'"):
            compare_sweep_statistical(
                entries, ledger=None, paths=None,
                metric="foo_metric",
            )

    def test_registry_contains_val_ic(self):
        """Sanity-check the MVP dispatch table."""
        assert "val_ic" in _STATISTIC_FN_REGISTRY


# =============================================================================
# Test 5: missing signal file raises
# =============================================================================


class TestMissingSignalFile:
    """T5: one record's signal dir has partial files → ValueError."""

    def test_missing_predicted_returns_raises(self, tmp_path: Path):
        rng = np.random.default_rng(200)
        n = 100
        labels = rng.standard_normal((n, 3))
        preds = rng.standard_normal((n, 3))

        # Record A has complete signals
        sig_a = _write_record_signals(
            tmp_path, record_id="rec_a",
            regression_labels=labels, predicted_returns=preds,
        )

        # Record B has labels + metadata but NO predicted_returns.npy
        sig_b = tmp_path / "rec_b"
        sig_b.mkdir()
        np.save(sig_b / "regression_labels.npy", labels)
        (sig_b / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
        }))

        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(ValueError, match=r"missing required signal file"):
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=100,
                )
