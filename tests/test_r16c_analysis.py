"""
Unit tests for hft_ops.ledger.r16c_analysis (Sub-cycle 4b, 2026-05-12).

Covers the helper primitives + verdict classifier + dataclass invariants.
Full end-to-end ``analyze_r16c_sweep`` integration is covered separately
in ``test_analyze_r16c_cli.py`` via synthetic-fixture subprocess tests.

Per Agent I T3 verdict 2026-05-12: 9 cases (5 unit + 4 invariant); skip
hypothesis library (no precedent dep in repo); use concrete hand-crafted
cases instead.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hft_ops.ledger.r16c_analysis import (
    CANONICAL_THRESHOLD_LABELS,
    DEFAULT_BOOTSTRAP_SEED,
    EXPECTED_CELLS,
    EXPECTED_GRID_POINTS,
    H1_DROP_TOP_K,
    H1_MEAN_FLOOR,
    H1_TARGET_THRESHOLD,
    H4_MEAN_FLOOR,
    N_BLOCKS_MIN,
    DecisionGateOutcome,
    R16cAnalysisError,
    R16cCellResult,
    R16cH5InvariantError,
    R16cIncompleteSweepError,
    _classify_verdict,
    _drop_top_k_by_abs_per_seed,
    _pooled_block_bootstrap_mean_ci,
    _verify_h5_bit_exact,
    outcome_to_json_dict,
    render_verdict,
)


# =============================================================================
# Constants invariants (THRESHOLD-CANON per Agent H E10)
# =============================================================================


class TestConstants:
    """Lock pre-registered constants — manifest L51-114 cite-trail."""

    def test_canonical_threshold_labels_count(self):
        assert len(CANONICAL_THRESHOLD_LABELS) == 8, (
            f"lob-backtester/scripts/run_regression_backtest.py:275-284 "
            f"declares 8 cost thresholds; r16c_analysis pinned "
            f"{len(CANONICAL_THRESHOLD_LABELS)}"
        )

    def test_canonical_threshold_labels_contains_deep_itm(self):
        assert "deep_itm_1.4bps" in CANONICAL_THRESHOLD_LABELS

    def test_h1_target_is_deep_itm(self):
        """H1 PRIMARY is conjunctive at deep_itm_1.4bps (manifest L51)."""
        assert H1_TARGET_THRESHOLD == "deep_itm_1.4bps"

    def test_h1_mean_floor_is_one_percent(self):
        """+1% per manifest L61."""
        assert H1_MEAN_FLOOR == 0.01

    def test_h1_drop_top_k_is_five(self):
        """Drop-top-5 per manifest L62-63 + Wave 3 Agent A precedent."""
        assert H1_DROP_TOP_K == 5

    def test_h4_mean_floor_is_negative_half_percent(self):
        """-0.5% per manifest L114."""
        assert H4_MEAN_FLOOR == -0.005

    def test_expected_grid_points_is_40(self):
        """2 model × 2 return_type × 10 seeds = 40 per manifest L18."""
        assert EXPECTED_GRID_POINTS == 40

    def test_expected_cells_is_4(self):
        """{ridge, tlob} × {point, peak} = 4 cells."""
        assert len(EXPECTED_CELLS) == 4
        assert ("temporal_ridge", "peak_return") in EXPECTED_CELLS
        assert ("temporal_ridge", "point_return") in EXPECTED_CELLS
        assert ("tlob", "peak_return") in EXPECTED_CELLS
        assert ("tlob", "point_return") in EXPECTED_CELLS


# =============================================================================
# _pooled_block_bootstrap_mean_ci (per Agent G Q2 + Agent H E3 N_SIZE_FLOOR)
# =============================================================================


class TestPooledBlockBootstrapMeanCi:
    """Single-array MBB CI on the mean. Specialized for R-16c per-trade pooled.

    Documented-intentional-near-duplicate of hft-metrics block_bootstrap_ci
    (which is paired (x, y)); see r16c_analysis.py module docstring + Agent G
    Q2 verdict 2026-05-12.
    """

    def test_known_mean_recovered_within_ci(self):
        """Known-mean array → bootstrap CI brackets the true mean."""
        rng = np.random.RandomState(42)
        arr = rng.normal(loc=0.05, scale=0.10, size=1000)  # true mean = 0.05
        est, ci_lo, ci_hi, n_repl, bl = _pooled_block_bootstrap_mean_ci(
            arr, n_bootstraps=500, seed=42,
        )
        assert abs(est - np.mean(arr)) < 1e-10  # exact pooled mean
        assert ci_lo < est < ci_hi  # est lies within CI
        assert n_repl == 0  # finite arr → no replacements
        assert bl >= 1

    def test_deterministic_same_seed(self):
        """Same arr + seed → identical (est, CI, n_repl) — determinism."""
        rng = np.random.RandomState(123)
        arr = rng.normal(loc=0.0, scale=1.0, size=500)
        r1 = _pooled_block_bootstrap_mean_ci(arr, n_bootstraps=200, seed=42)
        r2 = _pooled_block_bootstrap_mean_ci(arr, n_bootstraps=200, seed=42)
        assert r1 == r2, "same arr + seed must produce identical 5-tuple"

    def test_different_seed_different_ci(self):
        """Different seed → different CI bounds (sanity — RNG actually used)."""
        rng = np.random.RandomState(0)
        arr = rng.normal(0.0, 1.0, size=500)
        r1 = _pooled_block_bootstrap_mean_ci(arr, n_bootstraps=200, seed=42)
        r2 = _pooled_block_bootstrap_mean_ci(arr, n_bootstraps=200, seed=43)
        # est is identical (always the same data mean); CI bounds differ
        assert r1[0] == r2[0]
        assert r1[1] != r2[1] or r1[2] != r2[2]

    def test_n_lt_3_edge_case(self):
        """n < 3 → mirror hft-metrics edge-case behavior: CI = (m, m, m)."""
        arr = np.array([0.5, 0.5], dtype=np.float64)
        est, ci_lo, ci_hi, n_repl, bl = _pooled_block_bootstrap_mean_ci(arr)
        assert est == ci_lo == ci_hi == 0.5
        assert n_repl == 0
        assert bl == 1

    def test_empty_array(self):
        """Empty array → mean = NaN; defensive (Agent H E5 INSUFFICIENT_DATA cue)."""
        arr = np.array([], dtype=np.float64)
        est, ci_lo, ci_hi, n_repl, bl = _pooled_block_bootstrap_mean_ci(arr)
        assert math.isnan(est) and math.isnan(ci_lo) and math.isnan(ci_hi)

    def test_block_length_floor_raises_blocks_above_min(self):
        """Default cube-root rule might produce n_blocks < 5; helper adjusts
        block_length down to maintain ≥N_BLOCKS_MIN per Agent H E3.
        """
        # n=20, cube-root=ceil(20^(1/3))=ceil(2.71)=3 → n_blocks=20//3=6 (no floor)
        # Test: n=8, cube-root=ceil(2.0)=2 → n_blocks=4 < 5 → block_length adjusted
        # to max(1, 8//5)=1 → n_blocks=8
        arr = np.linspace(-0.01, 0.01, num=8, dtype=np.float64)
        _, _, _, _, bl_used = _pooled_block_bootstrap_mean_ci(
            arr, n_bootstraps=100, seed=42,
        )
        # Adjusted block_length should produce >= N_BLOCKS_MIN blocks
        n_blocks = max(1, len(arr) // max(1, bl_used))
        assert n_blocks >= N_BLOCKS_MIN, (
            f"n_blocks={n_blocks} should be >= {N_BLOCKS_MIN} after adjustment"
        )


# =============================================================================
# _drop_top_k_by_abs_per_seed (per Agent G Q4 + Wave 3 Agent A precedent)
# =============================================================================


class TestDropTopKByAbs:
    """Drop top-K by ABS magnitude per Agent G Q4 verdict 2026-05-12.

    Wave 3 Agent A precedent: R-16a "top 5 trades = 123.2% of return" used
    ABS magnitude (symmetric outlier-robustness). NOT top-K by VALUE
    (which would only drop biggest gains, biased toward negative result).
    """

    def test_known_top_k_dropped_by_abs(self):
        """Verify ABS-magnitude semantic: top-K can be either sign."""
        # 10 elements: 5 mild + 3 large positive + 2 large negative
        arr = np.array([0.01, 0.02, 0.01, -0.02, 0.01, 5.0, -4.0, 3.0, 0.5, -2.0])
        # Top-5 by abs: 5.0, -4.0, 3.0, -2.0, 0.5 → kept: [0.01, 0.02, 0.01, -0.02, 0.01]
        per_seed_means, overall = _drop_top_k_by_abs_per_seed([arr], k=5)
        kept_mean = np.mean([0.01, 0.02, 0.01, -0.02, 0.01])
        assert len(per_seed_means) == 1
        assert abs(per_seed_means[0] - kept_mean) < 1e-10
        assert abs(overall - kept_mean) < 1e-10

    def test_multi_seed_mean_of_per_seed(self):
        """Per-seed drop → mean across seeds (NOT pooled drop)."""
        # 3 seeds, each with top-1 of magnitude 10.0; rest are ones
        s1 = np.array([1.0, 1.0, 10.0, 1.0])
        s2 = np.array([1.0, 1.0, -10.0, 1.0])
        s3 = np.array([1.0, 1.0, 1.0, 5.0])
        per_seed, overall = _drop_top_k_by_abs_per_seed([s1, s2, s3], k=1)
        # Each seed drops its single largest-by-abs: mean of [1,1,1] = 1.0 each
        # Seed 3 drops 5.0: mean [1,1,1] = 1.0
        assert all(abs(m - 1.0) < 1e-10 for m in per_seed)
        assert abs(overall - 1.0) < 1e-10

    def test_k_greater_than_n_returns_zero(self):
        """Defensive: k >= n_trades → 0.0 (per Agent H E5 INSUFFICIENT_DATA cue)."""
        arr = np.array([1.0, 2.0, 3.0])
        per_seed, overall = _drop_top_k_by_abs_per_seed([arr], k=5)
        assert per_seed == (0.0,)
        assert overall == 0.0

    def test_empty_seed_array(self):
        """Empty seed array → contributes 0.0 to per-seed."""
        per_seed, overall = _drop_top_k_by_abs_per_seed([np.array([])], k=5)
        assert per_seed == (0.0,)

    def test_no_seeds(self):
        """Empty list of seeds → empty tuple + 0.0 overall."""
        per_seed, overall = _drop_top_k_by_abs_per_seed([], k=5)
        assert per_seed == tuple()
        assert overall == 0.0


# =============================================================================
# _verify_h5_bit_exact (per Agent G Q5 + Agent H E6 CRITICAL)
# =============================================================================


class TestVerifyH5BitExact:
    """Ridge × N-seed bit-exact invariant via SHA-256 on predicted_returns.npy.

    Per Phase A.3 REDESIGN: Ridge is RNG-FREE — identical input → identical
    output. Any divergence → ABORT exit code 2.
    """

    def test_identical_arrays_pass(self, tmp_path: Path):
        """Two dirs with byte-identical predicted_returns.npy → all_equal."""
        rng = np.random.RandomState(42)
        arr = rng.normal(0, 1, 100).astype(np.float64)
        dir1 = tmp_path / "seed_1"
        dir2 = tmp_path / "seed_2"
        dir1.mkdir()
        dir2.mkdir()
        np.save(dir1 / "predicted_returns.npy", arr)
        np.save(dir2 / "predicted_returns.npy", arr)
        all_equal, diag = _verify_h5_bit_exact("ridge_test", [dir1, dir2])
        assert all_equal is True
        assert diag is None

    def test_different_arrays_fail_with_diagnostic(self, tmp_path: Path):
        """Two dirs with DIFFERENT predicted_returns.npy → fail + diagnostic."""
        rng = np.random.RandomState(42)
        arr1 = rng.normal(0, 1, 100).astype(np.float64)
        arr2 = rng.normal(0, 1, 100).astype(np.float64)  # different draw
        dir1 = tmp_path / "seed_1"
        dir2 = tmp_path / "seed_2"
        dir1.mkdir()
        dir2.mkdir()
        np.save(dir1 / "predicted_returns.npy", arr1)
        np.save(dir2 / "predicted_returns.npy", arr2)
        all_equal, diag = _verify_h5_bit_exact("ridge_test", [dir1, dir2])
        assert all_equal is False
        assert diag is not None
        assert "ridge_test" in diag
        assert "distinct SHA" in diag

    def test_missing_file_returns_false_with_diagnostic(self, tmp_path: Path):
        """Missing predicted_returns.npy → false + actionable diagnostic."""
        dir1 = tmp_path / "seed_1"
        dir1.mkdir()
        # NO predicted_returns.npy file
        all_equal, diag = _verify_h5_bit_exact("ridge_test", [dir1])
        assert all_equal is False
        assert "missing" in diag.lower()

    def test_empty_signal_dirs_is_noop(self):
        """Empty list of signal dirs → no-op (True + descriptive msg)."""
        all_equal, diag = _verify_h5_bit_exact("ridge_test", [])
        assert all_equal is True


# =============================================================================
# _classify_verdict — 4-way decision-gate logic (per manifest L108-132)
# =============================================================================


class TestClassifyVerdict:
    """Map (5 gates, observed values) → 4-way verdict + exit code per manifest."""

    def test_all_pass_returns_go(self):
        verdict, exit_code = _classify_verdict(
            h1_mean_ok=True, h1_ci_ok=True, h1_drop_top5_ok=True,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=0.025, h1_ci_low_observed=0.005,
            h1_drop_top5_observed=0.005,
        )
        assert verdict == "GO"
        assert exit_code == 0

    def test_h5_fail_overrides_all_other_gates_to_abort(self):
        verdict, exit_code = _classify_verdict(
            h1_mean_ok=True, h1_ci_ok=True, h1_drop_top5_ok=True,
            h4_negative_control_ok=True, h5_invariant_ok=False,
            h1_mean_observed=0.025, h1_ci_low_observed=0.005,
            h1_drop_top5_observed=0.005,
        )
        assert verdict == "ABORT"
        assert exit_code == 2

    def test_h1a_fail_returns_refute(self):
        """Standard REFUTE: H5 OK + mean below floor → REFUTE."""
        verdict, exit_code = _classify_verdict(
            h1_mean_ok=False, h1_ci_ok=True, h1_drop_top5_ok=True,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=-0.005, h1_ci_low_observed=-0.002,
            h1_drop_top5_observed=0.005,
        )
        assert verdict == "REFUTE"
        assert exit_code == 1

    def test_borderline_returns_indeterminate(self):
        """Borderline: mean in (0, +1%), CI crosses 0, drop-top-5 marginal."""
        verdict, exit_code = _classify_verdict(
            h1_mean_ok=False, h1_ci_ok=False, h1_drop_top5_ok=False,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=0.005,         # in (0, 0.01)
            h1_ci_low_observed=-0.001,      # crosses 0
            h1_drop_top5_observed=-0.002,   # in (-0.005, 0)
        )
        assert verdict == "INDETERMINATE"
        assert exit_code == 1

    def test_h4_fail_returns_refute_when_h1_pass(self):
        """H1 all pass but H4 fail → REFUTE (cherry-pick detected)."""
        verdict, exit_code = _classify_verdict(
            h1_mean_ok=True, h1_ci_ok=True, h1_drop_top5_ok=True,
            h4_negative_control_ok=False, h5_invariant_ok=True,
            h1_mean_observed=0.025, h1_ci_low_observed=0.005,
            h1_drop_top5_observed=0.005,
        )
        assert verdict == "REFUTE"
        assert exit_code == 1


# =============================================================================
# Dataclass + JSON serialization
# =============================================================================


class TestDataclassSerialization:
    """Frozen dataclasses round-trip through JSON for downstream tooling."""

    def test_outcome_to_json_dict_includes_all_gates(self):
        outcome = DecisionGateOutcome(
            verdict="REFUTE",
            h1_mean_ok=False, h1_ci_ok=False, h1_drop_top5_ok=False,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=-0.0034, h1_ci_low_observed=-0.01,
            h1_ci_high_observed=0.003, h1_drop_top5_observed=-0.005,
            h4_mean_observed=-0.0034, h5_failed_cells=tuple(),
            reasons=("H1a FAIL", "H1b FAIL", "H1c FAIL", "H4 PASS", "H5 PASS"),
            exit_code=1,
        )
        d = outcome_to_json_dict(outcome, {})
        assert d["verdict"] == "REFUTE"
        assert d["exit_code"] == 1
        assert d["gates"]["h1_mean_ok"] is False
        assert d["gates"]["h5_invariant_ok"] is True
        assert d["observed"]["h1_mean"] == -0.0034
        assert "H1a FAIL" in d["reasons"]
        # JSON-serializable (no errors)
        import json
        json.dumps(d)

    def test_render_verdict_includes_verdict_and_gates(self):
        outcome = DecisionGateOutcome(
            verdict="GO",
            h1_mean_ok=True, h1_ci_ok=True, h1_drop_top5_ok=True,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=0.025, h1_ci_low_observed=0.005,
            h1_ci_high_observed=0.045, h1_drop_top5_observed=0.010,
            h4_mean_observed=0.012, h5_failed_cells=tuple(),
            reasons=("All gates PASS",), exit_code=0,
        )
        out = render_verdict(outcome, {})
        assert "VERDICT: GO" in out
        assert "Exit code: 0" in out
        assert "H1a" in out and "H5" in out


# =============================================================================
# Exception class hierarchy (per Agent G Q6 + Q8)
# =============================================================================


class TestExceptionHierarchy:
    """Distinct exit codes via subclass discrimination."""

    def test_incomplete_sweep_is_r16c_analysis_error(self):
        assert issubclass(R16cIncompleteSweepError, R16cAnalysisError)
        assert issubclass(R16cAnalysisError, ValueError)

    def test_h5_invariant_is_r16c_analysis_error(self):
        assert issubclass(R16cH5InvariantError, R16cAnalysisError)

    def test_can_catch_either_as_base(self):
        try:
            raise R16cIncompleteSweepError("missing 1 of 40 records")
        except R16cAnalysisError as e:
            assert "missing" in str(e)
        try:
            raise R16cH5InvariantError("ridge_peak: 2 distinct SHAs")
        except R16cAnalysisError as e:
            assert "ridge_peak" in str(e)
