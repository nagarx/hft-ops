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
# h5_bit_exact field contract (Phase A hygiene fix 2026-05-13)
# =============================================================================


class TestH5BitExactContract:
    """Lock h5_bit_exact field as Optional[bool] per docstring at r16c_analysis.py:196.

    Pre-Phase-A code at r16c_analysis.py:751 + :782 used
    ``(model_type == "temporal_ridge" and h5_invariant_ok)`` which evaluates to
    bool ``False`` for TLOB. This violated the dataclass docstring contract
    "Ridge cells only; None for TLOB" and would silently mislead downstream
    consumers (e.g., ``if cell.h5_bit_exact is None`` filters). Phase A hygiene
    cycle 2026-05-13 fixed both sites to ``(None if model_type != "temporal_ridge"
    else h5_invariant_ok)``. These tests lock the corrected contract.
    """

    def test_h5_bit_exact_field_typing_is_optional_bool(self):
        """Field annotation must be Optional[bool] per dataclass."""
        from typing import Optional, get_type_hints
        hints = get_type_hints(R16cCellResult)
        assert hints["h5_bit_exact"] == Optional[bool], (
            f"R16cCellResult.h5_bit_exact must be Optional[bool]; "
            f"got {hints['h5_bit_exact']}"
        )

    def test_ridge_cell_h5_bit_exact_accepts_true(self):
        """Ridge cell with h5_invariant_ok=True → field is bool True."""
        cell = R16cCellResult(
            arm_label="temporal_ridge__point_return",
            threshold_label="deep_itm_1.4bps",
            n_seeds=1, n_total_trades=100,
            mean_opt_ret=0.0, ci_low=-0.001, ci_high=0.001,
            n_nonfinite_replaced=0, block_length_used=9,
            drop_top5_per_seed=(0.0,), drop_top5_mean=0.0,
            h5_bit_exact=True,
            insufficient_data=False,
        )
        assert cell.h5_bit_exact is True
        assert isinstance(cell.h5_bit_exact, bool)

    def test_ridge_cell_h5_bit_exact_accepts_false(self):
        """Ridge cell with h5_invariant_ok=False → field is bool False (genuine failure)."""
        cell = R16cCellResult(
            arm_label="temporal_ridge__peak_return",
            threshold_label="deep_itm_1.4bps",
            n_seeds=1, n_total_trades=100,
            mean_opt_ret=0.0, ci_low=-0.001, ci_high=0.001,
            n_nonfinite_replaced=0, block_length_used=9,
            drop_top5_per_seed=(0.0,), drop_top5_mean=0.0,
            h5_bit_exact=False,
            insufficient_data=False,
        )
        assert cell.h5_bit_exact is False
        assert isinstance(cell.h5_bit_exact, bool)

    def test_tlob_cell_h5_bit_exact_must_be_none_not_false(self):
        """TLOB cell — h5_bit_exact MUST be None per docstring (NOT False).

        Defense-in-depth against pre-Phase-A bug at r16c_analysis.py:751+782
        where TLOB cells emitted h5_bit_exact=False instead of None.
        """
        cell = R16cCellResult(
            arm_label="tlob__peak_return",
            threshold_label="deep_itm_1.4bps",
            n_seeds=8, n_total_trades=5775,
            mean_opt_ret=-0.0001, ci_low=-0.0002, ci_high=0.0,
            n_nonfinite_replaced=0, block_length_used=9,
            drop_top5_per_seed=(-0.0001,) * 8, drop_top5_mean=-0.0001,
            h5_bit_exact=None,
            insufficient_data=False,
        )
        assert cell.h5_bit_exact is None
        # Explicit defense: not False (the pre-Phase-A bug value).
        assert cell.h5_bit_exact is not False

    def test_outcome_to_json_dict_serializes_none_h5_bit_exact_as_null(self):
        """JSON round-trip: TLOB cell h5_bit_exact=None must serialize as JSON null.

        Verifies outcome_to_json_dict + json.dumps + json.loads round-trip
        preserves None (not coerced to False or stripped from the dict).
        """
        import json
        tlob_cell = R16cCellResult(
            arm_label="tlob__peak_return",
            threshold_label="deep_itm_1.4bps",
            n_seeds=8, n_total_trades=5775,
            mean_opt_ret=-0.0001, ci_low=-0.0002, ci_high=0.0,
            n_nonfinite_replaced=0, block_length_used=9,
            drop_top5_per_seed=(-0.0001,) * 8, drop_top5_mean=-0.0001,
            h5_bit_exact=None,
            insufficient_data=False,
        )
        outcome = DecisionGateOutcome(
            verdict="REFUTE",
            h1_mean_ok=False, h1_ci_ok=False, h1_drop_top5_ok=False,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=0.0, h1_ci_low_observed=-0.001, h1_ci_high_observed=0.001,
            h1_drop_top5_observed=0.0, h4_mean_observed=0.0,
            h5_failed_cells=tuple(),
            reasons=("test",),
            exit_code=1,
        )
        cell_results = {("tlob", "peak_return", "deep_itm_1.4bps"): tlob_cell}
        json_dict = outcome_to_json_dict(outcome, cell_results)
        cell_key = "tlob__peak_return__deep_itm_1.4bps"
        assert json_dict["cell_results"][cell_key]["h5_bit_exact"] is None
        # JSON round-trip preserves None as null.
        json_str = json.dumps(json_dict)
        decoded = json.loads(json_str)
        assert decoded["cell_results"][cell_key]["h5_bit_exact"] is None

    def test_outcome_to_json_dict_serializes_ridge_h5_bit_exact_as_bool(self):
        """JSON round-trip: Ridge cell h5_bit_exact=True must serialize as JSON true."""
        import json
        ridge_cell = R16cCellResult(
            arm_label="temporal_ridge__point_return",
            threshold_label="deep_itm_1.4bps",
            n_seeds=1, n_total_trades=100,
            mean_opt_ret=0.0, ci_low=-0.001, ci_high=0.001,
            n_nonfinite_replaced=0, block_length_used=9,
            drop_top5_per_seed=(0.0,), drop_top5_mean=0.0,
            h5_bit_exact=True,
            insufficient_data=False,
        )
        outcome = DecisionGateOutcome(
            verdict="REFUTE",
            h1_mean_ok=False, h1_ci_ok=False, h1_drop_top5_ok=False,
            h4_negative_control_ok=True, h5_invariant_ok=True,
            h1_mean_observed=0.0, h1_ci_low_observed=-0.001, h1_ci_high_observed=0.001,
            h1_drop_top5_observed=0.0, h4_mean_observed=0.0,
            h5_failed_cells=tuple(),
            reasons=("test",),
            exit_code=1,
        )
        cell_results = {("temporal_ridge", "point_return", "deep_itm_1.4bps"): ridge_cell}
        json_dict = outcome_to_json_dict(outcome, cell_results)
        cell_key = "temporal_ridge__point_return__deep_itm_1.4bps"
        assert json_dict["cell_results"][cell_key]["h5_bit_exact"] is True
        # JSON round-trip preserves True as true.
        json_str = json.dumps(json_dict)
        decoded = json.loads(json_str)
        assert decoded["cell_results"][cell_key]["h5_bit_exact"] is True


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


# =============================================================================
# #PY-180 close (2026-05-13): DOLLAR → FRACTION unit conversion at load boundary
# =============================================================================


class TestLoadPerTradePnlsUnitConversion:
    """Lock the #PY-180 fix: _load_per_trade_pnls converts DOLLAR per-trade
    arrays to FRACTION-of-capital at load boundary so downstream gates
    (H1/H4) operate in fraction units consistent with manifest pre-registration.

    Producer (lob-backtester/scripts/run_regression_backtest.py:139) dumps USD
    per-trade pnls. Analyzer constants H1_MEAN_FLOOR=0.01 and H4_MEAN_FLOOR=-0.005
    are fractional (+1.0% and -0.5%). Without the load-boundary conversion,
    the analyzer would compare $4.05/trade dollar mean against 0.01 fraction
    threshold → semantically wrong dispatch.
    """

    def test_dollar_to_fraction_conversion_default_capital(self, tmp_path):
        from hft_ops.ledger.r16c_analysis import (
            DEFAULT_INITIAL_CAPITAL,
            _load_per_trade_pnls,
        )
        # Synthesize a DOLLAR-unit per-trade array
        dollar_pnls = np.array([400.0, 600.0, -200.0, 1000.0, -50.0], dtype=np.float64)
        path = tmp_path / "test_run__option_trade_pnls__deep_itm_1.4bps.npy"
        np.save(path, dollar_pnls)

        arr_frac = _load_per_trade_pnls(
            pnls_dir=tmp_path,
            run_name="test_run",
            threshold_label="deep_itm_1.4bps",
        )

        # Expected: arr_frac = dollar_pnls / 100_000 (DEFAULT_INITIAL_CAPITAL)
        expected = dollar_pnls / DEFAULT_INITIAL_CAPITAL
        np.testing.assert_array_almost_equal(arr_frac, expected, decimal=12)
        # Sanity: $400 / $100,000 = 0.004 = 0.4%
        assert arr_frac[0] == pytest.approx(0.004)

    def test_dollar_to_fraction_conversion_custom_capital(self, tmp_path):
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        # If a future R-cycle uses initial_capital=$50,000, fraction should reflect that
        dollar_pnls = np.array([500.0, -250.0, 1000.0], dtype=np.float64)
        path = tmp_path / "custom_run__option_trade_pnls__deep_itm_1.4bps.npy"
        np.save(path, dollar_pnls)

        arr_frac = _load_per_trade_pnls(
            pnls_dir=tmp_path,
            run_name="custom_run",
            threshold_label="deep_itm_1.4bps",
            initial_capital=50_000.0,
        )

        # $500 / $50,000 = 0.01 = 1.0%
        assert arr_frac[0] == pytest.approx(0.01)
        # $-250 / $50,000 = -0.005 = -0.5%
        assert arr_frac[1] == pytest.approx(-0.005)

    def test_initial_capital_zero_raises(self, tmp_path):
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        path = tmp_path / "run__option_trade_pnls__atm_5bps.npy"
        np.save(path, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="initial_capital must be finite and > 0"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="run",
                threshold_label="atm_5bps",
                initial_capital=0.0,
            )

    def test_initial_capital_negative_raises(self, tmp_path):
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        path = tmp_path / "run__option_trade_pnls__atm_5bps.npy"
        np.save(path, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="initial_capital must be finite and > 0"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="run",
                threshold_label="atm_5bps",
                initial_capital=-100.0,
            )

    def test_initial_capital_nan_raises(self, tmp_path):
        """NaN initial_capital MUST raise (pre-mid-impl-gate guard `<= 0` silently
        passed NaN since NaN comparisons always return False; mid-impl gate Agent A
        flagged as BLOCKING)."""
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        path = tmp_path / "run__option_trade_pnls__atm_5bps.npy"
        np.save(path, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="initial_capital must be finite and > 0"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="run",
                threshold_label="atm_5bps",
                initial_capital=math.nan,
            )

    def test_initial_capital_inf_raises(self, tmp_path):
        """Inf initial_capital MUST raise (pre-mid-impl-gate guard `<= 0` silently
        passed Inf since `inf <= 0` is False → division produced 0.0 silently)."""
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        path = tmp_path / "run__option_trade_pnls__atm_5bps.npy"
        np.save(path, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="initial_capital must be finite and > 0"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="run",
                threshold_label="atm_5bps",
                initial_capital=math.inf,
            )

    def test_initial_capital_negative_inf_raises(self, tmp_path):
        """Negative-Inf initial_capital MUST raise (both finiteness AND positivity
        checks fail; either alone is sufficient)."""
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        path = tmp_path / "run__option_trade_pnls__atm_5bps.npy"
        np.save(path, np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="initial_capital must be finite and > 0"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="run",
                threshold_label="atm_5bps",
                initial_capital=-math.inf,
            )

    def test_missing_file_returns_none_with_legitimate_skip(self, tmp_path):
        # Behavior preservation: file missing + status not provided OR 'completed' → None
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        result = _load_per_trade_pnls(
            pnls_dir=tmp_path,
            run_name="absent_run",
            threshold_label="deep_itm_1.4bps",
            record_status="completed",
        )
        assert result is None  # legitimate n_trades=0 skip

    def test_missing_file_with_failed_status_raises(self, tmp_path):
        # Behavior preservation: file missing + status='failed' → §8 fail-loud
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        with pytest.raises(R16cIncompleteSweepError, match="SWEEP FAILURE"):
            _load_per_trade_pnls(
                pnls_dir=tmp_path,
                run_name="failed_run",
                threshold_label="deep_itm_1.4bps",
                record_status="failed",
            )


# =============================================================================
# Wave 2 Agent F gap closure (2026-05-13): _resolve_backtest_pnls_dir resolver
# tests. PRE-CYCLE-4b ZERO tests existed for this 4-candidate fallback chain.
# =============================================================================


class TestResolveBacktestPnlsDir:
    """Lock the 4-candidate resolver fallback chain.

    Agent 2 mid-impl audit 2026-05-12 caught off-by-one walk-up arithmetic
    that produced `outputs/outputs/backtests` doubled path. The current chain
    (1) paths.backtester_dir / outputs / backtests (PRODUCTION),
    (2) <root>/lob-backtester/outputs/backtests (TEST/MANUAL),
    (3) sig_dir walk-up-5 / outputs / backtests (ALT LAYOUT),
    (4) Path("outputs") / "backtests" (CWD FALLBACK).

    Wave 2 Agent F flagged this resolver had ZERO tests despite being the
    most fragile join point between producer-side dump location and
    consumer-side analyzer load. These tests close that gap.
    """

    def _build_mock_paths(self, root: Path, backtester_dir):
        """Mock PipelinePaths supporting .pipeline_root and optional .backtester_dir.

        Phase A hygiene 2026-05-13: corrected from .root to .pipeline_root to
        match the production PipelinePaths attribute (paths.py:25). The
        original MockPaths from #PY-184 cycle used .root which never matched
        the production code path at r16c_analysis.py:467 (paths.pipeline_root).
        Test-side regression closed simultaneously with Phase A code hygiene.
        """
        class MockPaths:
            def __init__(self, r, bt):
                self.pipeline_root = r
                if bt is not None:
                    self.backtester_dir = bt
        return MockPaths(root, backtester_dir)

    def _build_mock_record_entry(self, signal_dir: Path):
        """Mock record entry providing signal_dir for resolve_signal_dir."""
        return {
            "experiment_id": "test_exp",
            "signal_export_output_dir": str(signal_dir),
            "axis_values": {"model_type": "temporal_ridge", "return_type": "peak_return"},
        }

    def test_candidate_1_paths_backtester_dir_found(self, tmp_path, monkeypatch):
        """Production case: orchestrator default path via paths.backtester_dir."""
        from hft_ops.ledger.r16c_analysis import _resolve_backtest_pnls_dir
        # Set up: backtester_dir/outputs/backtests exists
        backtester_dir = tmp_path / "backtester"
        target = backtester_dir / "outputs" / "backtests"
        target.mkdir(parents=True)
        sig_dir = tmp_path / "exp" / "signals" / "test"
        sig_dir.mkdir(parents=True)
        paths = self._build_mock_paths(tmp_path, backtester_dir)
        # Mock resolve_signal_dir to return our sig_dir directly
        monkeypatch.setattr(
            "hft_ops.ledger.r16c_analysis.resolve_signal_dir",
            lambda *args, **kwargs: sig_dir,
        )
        record = self._build_mock_record_entry(sig_dir)
        result = _resolve_backtest_pnls_dir(record, ledger=None, paths=paths)
        assert result == target

    def test_candidate_2_explicit_lob_backtester_layout_found(self, tmp_path, monkeypatch):
        """Test/manual case: <root>/lob-backtester/outputs/backtests when candidate 1 misses."""
        from hft_ops.ledger.r16c_analysis import _resolve_backtest_pnls_dir
        # Candidate 1 directory does NOT exist
        # Candidate 2 directory EXISTS
        target = tmp_path / "lob-backtester" / "outputs" / "backtests"
        target.mkdir(parents=True)
        sig_dir = tmp_path / "exp" / "signals" / "test"
        sig_dir.mkdir(parents=True)
        # backtester_dir set but its outputs/backtests does NOT exist
        backtester_dir = tmp_path / "nonexistent_backtester"
        paths = self._build_mock_paths(tmp_path, backtester_dir)
        monkeypatch.setattr(
            "hft_ops.ledger.r16c_analysis.resolve_signal_dir",
            lambda *args, **kwargs: sig_dir,
        )
        record = self._build_mock_record_entry(sig_dir)
        result = _resolve_backtest_pnls_dir(record, ledger=None, paths=paths)
        assert result == target

    def test_candidate_3_signal_dir_walkup_found(self, tmp_path, monkeypatch):
        """Alternative layout: walking up 5 levels from signal_dir lands at root."""
        from hft_ops.ledger.r16c_analysis import _resolve_backtest_pnls_dir
        # signal_dir is at outputs/experiments/<point>/signals/test/
        # parent.parent.parent.parent.parent => <root>
        # then / outputs / backtests
        sig_dir = tmp_path / "level5" / "outputs" / "experiments" / "point" / "signals" / "test"
        sig_dir.mkdir(parents=True)
        target = tmp_path / "level5" / "outputs" / "backtests"
        target.mkdir(parents=True)
        # Make sure candidates 1 + 2 DO NOT match
        paths = self._build_mock_paths(tmp_path, None)  # no backtester_dir
        monkeypatch.setattr(
            "hft_ops.ledger.r16c_analysis.resolve_signal_dir",
            lambda *args, **kwargs: sig_dir,
        )
        record = self._build_mock_record_entry(sig_dir)
        result = _resolve_backtest_pnls_dir(record, ledger=None, paths=paths)
        assert result == target

    def test_all_candidates_miss_raises_actionable_error(self, tmp_path, monkeypatch):
        """No candidate found → R16cIncompleteSweepError with searched-paths enumeration."""
        from hft_ops.ledger.r16c_analysis import _resolve_backtest_pnls_dir
        # NO candidate directories exist anywhere
        sig_dir = tmp_path / "exp" / "signals" / "test"
        sig_dir.mkdir(parents=True)
        paths = self._build_mock_paths(tmp_path, None)
        monkeypatch.setattr(
            "hft_ops.ledger.r16c_analysis.resolve_signal_dir",
            lambda *args, **kwargs: sig_dir,
        )
        record = self._build_mock_record_entry(sig_dir)
        with pytest.raises(R16cIncompleteSweepError, match="no per-trade pnls directory found"):
            _resolve_backtest_pnls_dir(record, ledger=None, paths=paths)

    def test_resolver_error_message_lists_searched_paths(self, tmp_path, monkeypatch):
        """Error message must enumerate all candidate paths for debuggability."""
        from hft_ops.ledger.r16c_analysis import _resolve_backtest_pnls_dir
        sig_dir = tmp_path / "exp" / "signals" / "test"
        sig_dir.mkdir(parents=True)
        backtester_dir = tmp_path / "bt"
        paths = self._build_mock_paths(tmp_path, backtester_dir)
        monkeypatch.setattr(
            "hft_ops.ledger.r16c_analysis.resolve_signal_dir",
            lambda *args, **kwargs: sig_dir,
        )
        record = self._build_mock_record_entry(sig_dir)
        try:
            _resolve_backtest_pnls_dir(record, ledger=None, paths=paths)
            pytest.fail("Expected R16cIncompleteSweepError")
        except R16cIncompleteSweepError as e:
            msg = str(e)
            # Each candidate path component should appear in error message
            assert "outputs" in msg and "backtests" in msg
            assert "Searched:" in msg
            # Backtester_dir path should appear (candidate 1)
            assert str(backtester_dir) in msg or "bt" in msg


# =============================================================================
# #PY-180 close: empirical smoke-test reproduction on R-16a fixtures
# =============================================================================


class TestPy180SmokeOnR16aFixtures:
    """Verify #PY-180 fix on actual R-16a per-trade .npy fixtures.

    The Sub-cycle 4b smoke-test produced REFUTE verdict on R-16a Ridge×Peak
    deep_itm_1.4bps cell. Post-fix, the same REFUTE must reproduce because
    H1(b) bootstrap CI bound is sign-invariant (CI crosses zero in DOLLAR
    units IFF it crosses zero in FRACTION units — unit conversion preserves
    sign).

    Locked here as a regression test: if a future change to load-time
    conversion breaks the empirical REFUTE reproduction, this test fails.
    """

    FIXTURE_DIR = Path(__file__).parent.parent.parent / "lob-backtester" / "outputs" / "backtests" / "r16a_smoke_test"

    def test_r16a_ridge_peak_deep_itm_loads_to_fraction(self):
        """R-16a Ridge×Peak deep_itm_1.4bps loaded post-fix: DOLLAR mean ~$4.05 → FRACTION ~0.0000405."""
        from hft_ops.ledger.r16c_analysis import _load_per_trade_pnls
        if not self.FIXTURE_DIR.exists():
            pytest.skip(f"R-16a smoke fixtures not present at {self.FIXTURE_DIR}")

        arr = _load_per_trade_pnls(
            pnls_dir=self.FIXTURE_DIR,
            run_name="R-16a_ridge_peak_H60_smoke",
            threshold_label="deep_itm_1.4bps",
        )
        assert arr is not None, "expected R-16a ridge×peak deep_itm_1.4bps fixture"
        # Verify mean is small fraction (dollar mean ~$4.05 → fraction ~4.05e-5)
        mean_frac = float(np.mean(arr))
        # Sanity-check fraction units: |mean| should be << 1 (it's a small per-trade fraction)
        assert abs(mean_frac) < 0.01, (
            f"Expected fraction-of-capital units; mean_frac={mean_frac:.6f} is too large. "
            f"Pre-fix this would be ~$4.05 (DOLLAR units)."
        )
        # Empirical R-16a result: ~$4.05 / $100,000 = ~4.05e-5
        # Allow loose tolerance since exact value depends on bootstrap fixture
        assert -0.001 < mean_frac < 0.001, (
            f"Expected very small fractional mean (~4e-5 from R-16a); got {mean_frac:.6f}"
        )

    def test_r16a_ridge_peak_ci_crosses_zero_refute(self):
        """Empirical reproduction of Sub-cycle 4b REFUTE verdict at fraction units."""
        from hft_ops.ledger.r16c_analysis import (
            _load_per_trade_pnls,
            _pooled_block_bootstrap_mean_ci,
        )
        if not self.FIXTURE_DIR.exists():
            pytest.skip(f"R-16a smoke fixtures not present at {self.FIXTURE_DIR}")
        arr = _load_per_trade_pnls(
            pnls_dir=self.FIXTURE_DIR,
            run_name="R-16a_ridge_peak_H60_smoke",
            threshold_label="deep_itm_1.4bps",
        )
        if arr is None or len(arr) < 10:
            pytest.skip("R-16a fixture too small for bootstrap CI")
        mean_pooled, ci_lo, ci_hi, n_nonfinite, bl_used = _pooled_block_bootstrap_mean_ci(
            arr, n_bootstraps=10000, seed=42,
        )
        # REFUTE requires CI crosses zero
        assert ci_lo < 0 < ci_hi, (
            f"Expected CI to cross zero (REFUTE per Sub-cycle 4b empirical); "
            f"got ci_lo={ci_lo:.6e}, ci_high={ci_hi:.6e}"
        )
        # CI bounds should be SMALL fractions (sub-percent magnitudes)
        assert abs(ci_lo) < 0.01 and abs(ci_hi) < 0.01, (
            f"Expected fraction-unit CI bounds (sub-1%); got "
            f"ci_lo={ci_lo:.4f}, ci_hi={ci_hi:.4f}"
        )
