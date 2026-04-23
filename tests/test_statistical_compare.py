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
            results, labels, diag = compare_sweep_statistical(
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

        # V.1 L3.3: diagnostics payload surfaces observability
        assert diag.n_treatments == 3
        assert diag.n_bootstraps == 200
        assert diag.metric == "val_ic"
        assert diag.n_samples_paired == diag.n_samples_raw  # clean input
        assert diag.n_dropped_nonfinite == 0
        assert diag.drop_fraction == 0.0


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


# =============================================================================
# Phase V.1 L1.3: NaN-row drop threshold (Agent 2 H2 closure)
# =============================================================================


class TestNaNDropThreshold:
    """L1.3 (2026-04-21): silent NaN-row drop pre-V.1 was a §8 rule
    violation. Now: drop only when under threshold; raise with per-record
    breakdown when over."""

    def _fresh_signals(self, tmp_path: Path, n: int, nan_rows: int = 0):
        """Helper: build 2 records with `nan_rows` rows poisoned in preds."""
        rng = np.random.default_rng(321)
        labels = rng.standard_normal((n, 3))
        preds_a = rng.standard_normal((n, 3)) + 0.2 * labels
        preds_b = rng.standard_normal((n, 3)) + 0.1 * labels
        if nan_rows > 0:
            # Poison the FIRST nan_rows rows of preds_a
            preds_a[:nan_rows, :] = np.nan
        sig_a = _write_record_signals(
            tmp_path, record_id="rec_a",
            regression_labels=labels, predicted_returns=preds_a,
        )
        sig_b = _write_record_signals(
            tmp_path, record_id="rec_b",
            regression_labels=labels, predicted_returns=preds_b,
        )
        return sig_a, sig_b

    def test_under_threshold_drops_and_proceeds(self, tmp_path: Path):
        """3% NaN rows → below default 5% threshold → drop + proceed."""
        n = 200
        sig_a, sig_b = self._fresh_signals(tmp_path, n, nan_rows=6)  # 3%
        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            results, labels, diag = compare_sweep_statistical(
                entries, ledger=None, paths=None,
                metric="val_ic", n_bootstraps=100, seed=42,
            )
        assert len(results) == 1
        assert diag.n_dropped_nonfinite == 6
        assert diag.n_samples_raw == n
        assert diag.n_samples_paired == n - 6
        assert diag.drop_fraction == 0.03

    def test_over_threshold_raises_with_breakdown(self, tmp_path: Path):
        """10% NaN rows → above default 5% threshold → raise with per-record breakdown."""
        n = 200
        sig_a, sig_b = self._fresh_signals(tmp_path, n, nan_rows=20)  # 10%
        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(
                ValueError, match=r"NaN-row drop fraction 10\.00% exceeds max_drop_frac"
            ) as exc_info:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=100, seed=42,
                )
        # Per-record breakdown present in error message
        assert "rec_a" in str(exc_info.value)
        assert "y_nonfinite=" in str(exc_info.value)

    def test_max_drop_frac_override_allows_proceed(self, tmp_path: Path):
        """max_drop_frac=1.0 disables the guard — for pathological cases
        where operator knows what they're doing. Locks that the override
        is HONORED (not silently re-enforced)."""
        n = 200
        sig_a, sig_b = self._fresh_signals(tmp_path, n, nan_rows=100)  # 50%
        entries = _make_entries(["rec_a", "rec_b"])
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            results, labels, diag = compare_sweep_statistical(
                entries, ledger=None, paths=None,
                metric="val_ic", n_bootstraps=100, seed=42,
                max_drop_frac=1.0,
            )
        assert diag.drop_fraction == 0.5

    def test_invalid_max_drop_frac_raises(self):
        """max_drop_frac outside [0, 1] → ValueError."""
        entries = _make_entries(["a", "b"])
        with pytest.raises(ValueError, match=r"max_drop_frac must be in"):
            compare_sweep_statistical(
                entries, ledger=None, paths=None,
                metric="val_ic", max_drop_frac=1.5,
            )
        with pytest.raises(ValueError, match=r"max_drop_frac must be in"):
            compare_sweep_statistical(
                entries, ledger=None, paths=None,
                metric="val_ic", max_drop_frac=-0.1,
            )


# =============================================================================
# Phase V.1 L3.2: Label uniqueness check
# =============================================================================


class TestLabelUniqueness:
    """L3.2 (2026-04-21): sweep grid-points with COLLIDING axis_values
    would produce ambiguous pairwise output rows. Fail-loud per §5 + §8."""

    def test_duplicate_labels_raise(self, tmp_path: Path):
        """Two records with identical axis_values → ambiguous labels → raise."""
        rng = np.random.default_rng(99)
        n = 100
        shared_labels = rng.standard_normal((n, 3))
        preds_a = rng.standard_normal((n, 3))
        preds_b = rng.standard_normal((n, 3))

        sig_a = _write_record_signals(
            tmp_path, record_id="rec_a",
            regression_labels=shared_labels, predicted_returns=preds_a,
        )
        sig_b = _write_record_signals(
            tmp_path, record_id="rec_b",
            regression_labels=shared_labels, predicted_returns=preds_b,
        )

        # Make both entries carry the SAME axis_values → labels collide
        entries = [
            {
                "experiment_id": "rec_a",
                "name": "exp_rec_a",
                "record_type": "experiment",
                "axis_values": {"seed": "42"},  # identical
            },
            {
                "experiment_id": "rec_b",
                "name": "exp_rec_b",
                "record_type": "experiment",
                "axis_values": {"seed": "42"},  # identical
            },
        ]
        signal_dirs = {"rec_a": sig_a, "rec_b": sig_b}

        def fake_resolve(record_entry, ledger, paths):
            return signal_dirs[record_entry["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(
                ValueError, match=r"duplicate treatment labels"
            ) as exc_info:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
        assert "seed=42" in str(exc_info.value)


# =============================================================================
# V.1.5 SDR-8: axis-semantic cross-check against metadata horizons
# =============================================================================


class TestAxisSemanticCheck:
    """V.1.5 follow-up (2026-04-23) — SDR-8 closure.

    `_load_record_signals` previously accepted any (N, H) or (N,) labels
    array with ``ndim in (1, 2)`` without validating which axis is which.
    A transposed (H, N) array would silently pass (shape_0 == N, ndim==2)
    and downstream ``arr[:, primary_horizon_idx]`` would read the wrong
    axis — producing numerically valid but semantically nonsense results.

    V.1.5 adds: when `meta["horizons"]` is a non-empty list, cross-check
    `labels_arr.shape[1] == len(horizons)`. Mismatch raises fail-loud with
    transpose vs genuine-skew diagnostic.

    Absent `horizons` field (pre-Phase-II legacy manifests) → silent pass
    for back-compat.
    """

    def _write_bad_shape_signals(
        self, base_dir: Path, record_id: str,
        *, labels_shape: tuple, horizons_meta: list,
    ) -> Path:
        """Write a bad-shape record fixture. labels_shape can be anything;
        preds_shape always matches (to isolate the axis-mismatch, not the
        shape-mismatch assertion which is covered elsewhere)."""
        rng = np.random.default_rng(42)
        labels = rng.standard_normal(labels_shape)
        preds = rng.standard_normal(labels_shape)
        sig_dir = base_dir / record_id
        sig_dir.mkdir(parents=True, exist_ok=True)
        np.save(sig_dir / "regression_labels.npy", labels)
        np.save(sig_dir / "predicted_returns.npy", preds)
        (sig_dir / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
            "horizons": horizons_meta,
        }))
        return sig_dir

    def test_transposed_array_raises_with_transpose_hint(self, tmp_path: Path):
        """labels shape (3, N) but horizons=[10,60,300] → 'TRANSPOSED' hint."""
        sig = self._write_bad_shape_signals(
            tmp_path, "rec_bad",
            labels_shape=(3, 200),   # transposed (H, N) — matches shape[0]
            horizons_meta=[10, 60, 300],
        )
        entries = _make_entries(["rec_bad"]) + _make_entries(["rec_bad2"])
        # Second record also written so the adapter passes the < 2 check
        sig2 = self._write_bad_shape_signals(
            tmp_path, "rec_bad2",
            labels_shape=(3, 200), horizons_meta=[10, 60, 300],
        )
        signal_dirs = {"rec_bad": sig, "rec_bad2": sig2}

        def fake_resolve(e, l, p):
            return signal_dirs[e["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(ValueError) as exc_info:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
        assert "TRANSPOSED" in str(exc_info.value)
        assert "(3, 200)" in str(exc_info.value)
        assert "(N, 3)" in str(exc_info.value)

    def test_genuine_shape_mismatch_raises_with_schema_hint(self, tmp_path: Path):
        """labels shape (N, 2) with horizons=[10,60,300] → 'Signal-export
        schema mismatch' (neither axis matches 3 horizons)."""
        sig = self._write_bad_shape_signals(
            tmp_path, "rec_skew",
            labels_shape=(200, 2),   # ndim=2 but shape[1]=2 != 3
            horizons_meta=[10, 60, 300],
        )
        sig2 = self._write_bad_shape_signals(
            tmp_path, "rec_skew2",
            labels_shape=(200, 2), horizons_meta=[10, 60, 300],
        )
        entries = _make_entries(["rec_skew"]) + _make_entries(["rec_skew2"])
        signal_dirs = {"rec_skew": sig, "rec_skew2": sig2}

        def fake_resolve(e, l, p):
            return signal_dirs[e["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            with pytest.raises(ValueError) as exc_info:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
        assert "schema mismatch" in str(exc_info.value)
        assert "neither axis" in str(exc_info.value)

    def test_happy_path_shape_matches_horizons_passes(self, tmp_path: Path):
        """labels shape (N, 3) with horizons=[10,60,300] → passes silently."""
        sig = self._write_bad_shape_signals(
            tmp_path, "rec_good",
            labels_shape=(200, 3),
            horizons_meta=[10, 60, 300],
        )
        sig2 = self._write_bad_shape_signals(
            tmp_path, "rec_good2",
            labels_shape=(200, 3),
            horizons_meta=[10, 60, 300],
        )
        entries = _make_entries(["rec_good"]) + _make_entries(["rec_good2"])
        signal_dirs = {"rec_good": sig, "rec_good2": sig2}

        def fake_resolve(e, l, p):
            return signal_dirs[e["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            # Must NOT raise the axis-mismatch error.
            # (May still raise other errors — e.g., SHA paired check — because
            # we generated random arrays with different seeds per call.
            # We only assert the axis check did NOT fire.)
            try:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
            except ValueError as e:
                assert "TRANSPOSED" not in str(e), (
                    f"Valid (N, H=3) should not trigger axis-mismatch; got: {e}"
                )
                assert "schema mismatch" not in str(e), (
                    f"Valid (N, H=3) should not trigger schema-mismatch; got: {e}"
                )
                # SHA-paired-labels error is separate + expected here.

    def test_legacy_manifest_without_horizons_passes(self, tmp_path: Path):
        """Pre-Phase-II manifest (no 'horizons' key) → no axis check runs
        (graceful back-compat; labels shape can be anything ndim-valid)."""
        # Write a fixture with labels (200, 2) but NO horizons in metadata
        rng = np.random.default_rng(42)
        labels = rng.standard_normal((200, 2))
        preds = rng.standard_normal((200, 2))
        sig_dir = tmp_path / "rec_legacy"
        sig_dir.mkdir(parents=True, exist_ok=True)
        np.save(sig_dir / "regression_labels.npy", labels)
        np.save(sig_dir / "predicted_returns.npy", preds)
        (sig_dir / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
            # horizons intentionally absent — legacy manifest
        }))
        sig_dir2 = tmp_path / "rec_legacy2"
        sig_dir2.mkdir(parents=True, exist_ok=True)
        np.save(sig_dir2 / "regression_labels.npy", labels)
        np.save(sig_dir2 / "predicted_returns.npy", preds)
        (sig_dir2 / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
        }))
        entries = _make_entries(["rec_legacy", "rec_legacy2"])
        signal_dirs = {"rec_legacy": sig_dir, "rec_legacy2": sig_dir2}

        def fake_resolve(e, l, p):
            return signal_dirs[e["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            # Legacy manifest: axis check MUST NOT fire (no 'horizons' key)
            # Other errors may still occur but not the new axis check.
            try:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
            except ValueError as e:
                assert "TRANSPOSED" not in str(e)
                assert "horizon-axis mismatch" not in str(e)

    def test_empty_horizons_list_is_treated_as_legacy(self, tmp_path: Path):
        """`horizons: []` (empty list) is NOT a drift signal — skip check.
        Possible in incomplete Phase-II manifests; graceful handling."""
        rng = np.random.default_rng(42)
        labels = rng.standard_normal((200, 3))
        preds = rng.standard_normal((200, 3))
        sig_dir = tmp_path / "rec_empty_horizons"
        sig_dir.mkdir(parents=True, exist_ok=True)
        np.save(sig_dir / "regression_labels.npy", labels)
        np.save(sig_dir / "predicted_returns.npy", preds)
        (sig_dir / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
            "horizons": [],   # empty — treated as legacy
        }))
        sig_dir2 = tmp_path / "rec_empty_horizons2"
        sig_dir2.mkdir(parents=True, exist_ok=True)
        np.save(sig_dir2 / "regression_labels.npy", labels)
        np.save(sig_dir2 / "predicted_returns.npy", preds)
        (sig_dir2 / "signal_metadata.json").write_text(json.dumps({
            "signal_type": "regression",
            "primary_horizon_idx": 0,
            "horizons": [],
        }))
        entries = _make_entries(["rec_empty_horizons", "rec_empty_horizons2"])
        signal_dirs = {
            "rec_empty_horizons": sig_dir,
            "rec_empty_horizons2": sig_dir2,
        }

        def fake_resolve(e, l, p):
            return signal_dirs[e["experiment_id"]]

        with patch(
            "hft_ops.ledger.statistical_compare._resolve_signal_dir",
            side_effect=fake_resolve,
        ):
            try:
                compare_sweep_statistical(
                    entries, ledger=None, paths=None,
                    metric="val_ic", n_bootstraps=10,
                )
            except ValueError as e:
                assert "TRANSPOSED" not in str(e)
                assert "schema mismatch" not in str(e)
