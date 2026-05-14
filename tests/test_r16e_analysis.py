"""
Tests for ``hft_ops.ledger.r16e_analysis`` (Cycle 9 / R-16e EXTENDED, 2026-05-14).

Focused on unit-test coverage of:
- FROM-IMPORT integrity (r16c → r16e reuse contract)
- ``_paired_bootstrap_ic_ratio`` primitive (NEW R-16e)
- ``_classify_verdict_r16e`` decision-gate logic (parametric)
- ``R16eCellResult`` + ``R16eRatioResult`` + ``R16eDecisionGateOutcome`` dataclass invariants
- Public surface (canonical __all__-equivalent)

End-to-end ``analyze_r16e_sweep`` integration deferred to a follow-up cycle
(requires synthetic 40-record ledger fixture; same pattern as r16c integration
tests at ``test_r16c_analysis.py`` ~900 LOC). The CLI wrapper at
``scripts/analyze_r16e.py`` is exercised via the live sweep post-completion.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from hft_ops.ledger import r16e_analysis as m
from hft_ops.ledger.r16e_analysis import (
    # Constants (LOCKED pre-sweep per hft-rules §13)
    CANONICAL_THRESHOLD_LABELS,
    EXPECTED_GRID_POINTS,
    EXPECTED_CELLS,
    H1_TARGET_THRESHOLD,
    H1_MEAN_FLOOR,
    H1_IC_FLOOR,
    H2_RATIO_FLOOR,
    H1_PRIMARY_CELL,
    H6_DIAGNOSTIC_CELLS,
    # Exception classes
    R16eAnalysisError,
    R16eIncompleteSweepError,
    R16eH4InvariantError,
    # Result dataclasses
    R16eCellResult,
    R16eRatioResult,
    R16eDecisionGateOutcome,
    # Primitives (NEW)
    _paired_bootstrap_ic_ratio,
    _classify_verdict_r16e,
    _extract_test_ic,
)


class TestFromImportIntegrity:
    """Verify r16c → r16e FROM-IMPORT contract is satisfied."""

    def test_canonical_thresholds_match_r16c(self):
        """R-16e re-exports r16c's CANONICAL_THRESHOLD_LABELS — must match exactly."""
        from hft_ops.ledger.r16c_analysis import (
            CANONICAL_THRESHOLD_LABELS as r16c_labels,
        )
        assert CANONICAL_THRESHOLD_LABELS == r16c_labels
        assert len(CANONICAL_THRESHOLD_LABELS) == 8

    def test_h1_target_threshold_in_canonical_set(self):
        """H1_TARGET_THRESHOLD must be one of the 8 canonical thresholds."""
        assert H1_TARGET_THRESHOLD in CANONICAL_THRESHOLD_LABELS

    def test_r16e_exception_inherits_r16c_hierarchy(self):
        """R-16e exception classes subclass r16c base for existing
        exit-code dispatch logic compatibility."""
        from hft_ops.ledger.r16c_analysis import (
            R16cAnalysisError,
            R16cIncompleteSweepError,
            R16cH5InvariantError,
        )
        assert issubclass(R16eAnalysisError, R16cAnalysisError)
        assert issubclass(R16eIncompleteSweepError, R16cIncompleteSweepError)
        assert issubclass(R16eH4InvariantError, R16cH5InvariantError)


class TestPreRegisteredConstants:
    """LOCKED constants per hft-rules §13 must match cycle9_r16e manifest."""

    def test_expected_grid_points_matches_manifest(self):
        # 2 model × 2 return × 10 seed = 40 grid points per manifest L?
        assert EXPECTED_GRID_POINTS == 40

    def test_expected_cells_4_arms(self):
        """4 cells per cycle9_r16e: ridge×point, ridge×smoothed, tlob×point, tlob×smoothed."""
        assert len(EXPECTED_CELLS) == 4
        assert ("temporal_ridge", "point_return") in EXPECTED_CELLS
        assert ("temporal_ridge", "smoothed_return") in EXPECTED_CELLS
        assert ("tlob", "point_return") in EXPECTED_CELLS
        assert ("tlob", "smoothed_return") in EXPECTED_CELLS

    def test_no_peak_return_in_r16e(self):
        """R-16e uses {point, smoothed} (E8 diagnostic at H60-hold),
        NOT {point, peak} (R-16c's F7 falsification scope)."""
        for (_, return_type) in EXPECTED_CELLS:
            assert return_type != "peak_return", \
                f"R-16e should NOT include peak_return (that's R-16c's scope); got {return_type}"

    def test_h1_primary_cell_is_ridge_point(self):
        """R-16e's H1 PRIMARY is Ridge × Point × H60 (R-16d's headline finding)."""
        assert H1_PRIMARY_CELL == ("temporal_ridge", "point_return")

    def test_h6_diagnostic_cells_are_smoothed(self):
        """H6 E8 diagnostic operates on smoothed × {Ridge, TLOB}."""
        assert len(H6_DIAGNOSTIC_CELLS) == 2
        for (mt, rt) in H6_DIAGNOSTIC_CELLS:
            assert rt == "smoothed_return"

    def test_h1_mean_floor_is_zero_pct(self):
        """R-16e H1 mean floor is 0% (looser than R-16c's +1%)
        since H1c IC floor is the gating constraint."""
        assert H1_MEAN_FLOOR == 0.0

    def test_h1_ic_floor_matches_claude_md_predictive_threshold(self):
        """H1c IC floor is 0.05 per CLAUDE.md 'predictive IC > 0.05 → tradeable'."""
        assert H1_IC_FLOOR == 0.05

    def test_h2_ratio_floor_below_r16d_observed(self):
        """H2 ratio floor 1.5 — R-16d single-seed observed 2.585;
        multi-seed CI lower bound > 1.5 is the acceptance gate."""
        assert H2_RATIO_FLOOR == 1.5
        # Sanity: R-16d observed Ratio must clear the floor
        r16d_observed_ratio = 2.585
        assert r16d_observed_ratio > H2_RATIO_FLOOR


class TestPairedBootstrapIcRatio:
    """Unit tests for NEW R-16e primitive _paired_bootstrap_ic_ratio."""

    def test_perfect_ratio_recovery(self):
        """Ridge_IC=0.15, TLOB constant 0.06 → ratio = 2.5 with tight CI."""
        ridge_ic = 0.15
        tlob_ic = np.array([0.06] * 10, dtype=np.float64)
        ratio, lo, hi = _paired_bootstrap_ic_ratio(
            ridge_ic, tlob_ic, n_bootstraps=1000, seed=42,
        )
        assert math.isclose(ratio, 2.5, rel_tol=1e-6)
        # CI should be tight when TLOB is constant (no seed variance)
        assert math.isclose(lo, 2.5, rel_tol=1e-6)
        assert math.isclose(hi, 2.5, rel_tol=1e-6)

    def test_variable_tlob_widens_ci(self):
        """TLOB with seed variance widens CI."""
        ridge_ic = 0.15
        tlob_ic = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.05, 0.06, 0.07, 0.06, 0.06], dtype=np.float64)
        ratio, lo, hi = _paired_bootstrap_ic_ratio(
            ridge_ic, tlob_ic, n_bootstraps=2000, seed=42,
        )
        # Mean TLOB IC = 0.06; ratio = 0.15/0.06 = 2.5
        assert math.isclose(ratio, 2.5, rel_tol=0.01)
        # CI should bracket the point estimate
        assert lo < ratio < hi
        # CI should be non-trivial (variable TLOB → real bootstrap variance)
        assert hi - lo > 0.01

    def test_r16d_observed_ratio_replication(self):
        """Replicate R-16d observed Ratio=2.585 with realistic TLOB seed variance."""
        ridge_ic = 0.1473  # R-16d Ridge×Point×H60 exact value
        # Synthetic TLOB seeds with mean=0.057 (R-16d observed)
        np.random.seed(123)
        tlob_ic = np.random.normal(0.057, 0.015, size=10).astype(np.float64)
        ratio, lo, hi = _paired_bootstrap_ic_ratio(
            ridge_ic, tlob_ic, n_bootstraps=2000, seed=42,
        )
        # Ratio should be around 2.585; CI brackets it
        assert 1.5 < ratio < 4.0  # rough sanity bounds
        assert lo < ratio < hi

    def test_zero_tlob_returns_nan(self):
        """If TLOB mean is ~0, ratio computation returns NaN (div-by-zero guard)."""
        ridge_ic = 0.15
        tlob_ic = np.array([1e-15] * 10, dtype=np.float64)  # near-zero
        ratio, lo, hi = _paired_bootstrap_ic_ratio(
            ridge_ic, tlob_ic, n_bootstraps=100, seed=42,
        )
        assert math.isnan(ratio)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_non_finite_ridge_raises(self):
        """Non-finite Ridge IC raises ValueError."""
        tlob_ic = np.array([0.06] * 10, dtype=np.float64)
        with pytest.raises(ValueError, match="non-finite ridge_ic"):
            _paired_bootstrap_ic_ratio(float("nan"), tlob_ic)

    def test_empty_tlob_raises(self):
        """Empty TLOB IC array raises ValueError."""
        with pytest.raises(ValueError, match="empty tlob_ic_per_seed"):
            _paired_bootstrap_ic_ratio(0.15, np.array([], dtype=np.float64))

    def test_nan_in_tlob_raises(self):
        """Non-finite values in TLOB IC raise ValueError via assert_finite_array."""
        tlob_ic = np.array([0.06, 0.05, np.nan, 0.07, 0.06], dtype=np.float64)
        with pytest.raises(ValueError):
            _paired_bootstrap_ic_ratio(0.15, tlob_ic)


class TestClassifyVerdictR16e:
    """Parametric tests for _classify_verdict_r16e logic.

    Updated 2026-05-14 (#PY-208 closure): tests now reflect the manifest-aligned
    H1 gates:
      - h1_ci_ok           = manifest H1(a): pooled CI > 0
      - h1_mean_across_8_ok = manifest H1(b): mean across 8 thresholds > 0
      - h1_ic_floor_ok     = manifest H1(c): per-seed test_ic CI > 0.05
      - h1_ci_borderline   = INDETERMINATE clause prerequisite
    """

    @pytest.mark.parametrize("h1_ci,h1_mean8,h1_ic,h4,expected_verdict,expected_exit", [
        # All H1 + H4 pass → GO
        (True, True, True, True, "GO", 0),
        # Any H1 fail + H4 pass + not borderline → REFUTE
        (False, True, True, True, "REFUTE", 1),
        (True, False, True, True, "REFUTE", 1),
        (True, True, False, True, "REFUTE", 1),
        # H4 fails → ABORT (regardless of H1)
        (True, True, True, False, "ABORT", 2),
        (False, False, False, False, "ABORT", 2),
        # All H1 fail + H4 fail → ABORT (H4 dominates)
        (False, False, False, False, "ABORT", 2),
    ])
    def test_classify_verdict_parametric_non_borderline(
        self, h1_ci: bool, h1_mean8: bool, h1_ic: bool, h4: bool,
        expected_verdict: str, expected_exit: int,
    ):
        """Parametric over manifest-aligned gates with borderline=False default."""
        verdict, exit_code = _classify_verdict_r16e(
            h1_ci_ok=h1_ci,
            h1_mean_across_8_ok=h1_mean8,
            h1_ic_floor_ok=h1_ic,
            h4_invariant_ok=h4,
        )
        assert verdict == expected_verdict
        assert exit_code == expected_exit

    def test_h4_fail_dominates_all_h1(self):
        """H4 fail → ABORT regardless of any H1 state (ship-blocker semantics)."""
        for h1_combo in [(True, True, True), (False, False, False), (True, False, True)]:
            verdict, _ = _classify_verdict_r16e(
                h1_ci_ok=h1_combo[0],
                h1_mean_across_8_ok=h1_combo[1],
                h1_ic_floor_ok=h1_combo[2],
                h4_invariant_ok=False,
            )
            assert verdict == "ABORT", \
                f"Expected ABORT when H4 fails, got {verdict} with H1={h1_combo}"

    def test_indeterminate_clause_triggers_when_ci_borderline_and_mean_passes(self):
        """Manifest line 157-158: H1(a) borderline AND H1(b) > 0 → INDETERMINATE.

        Closes the analyzer drift surfaced by #PY-208 (R-16e empirical case:
        H1(a) FAIL with CI=(-0.000468, +0.000313) — borderline within ±1% —
        AND H1(b) PASS at mean-across-8=+0.2025% → INDETERMINATE per manifest).
        """
        verdict, exit_code = _classify_verdict_r16e(
            h1_ci_ok=False,            # CI FAIL (lower < 0)
            h1_mean_across_8_ok=True,  # H1(b) PASS
            h1_ic_floor_ok=True,       # H1(c) PASS
            h4_invariant_ok=True,      # H4 PASS
            h1_ci_borderline=True,     # CI within ±1% margin
        )
        assert verdict == "INDETERMINATE", \
            "Expected INDETERMINATE per manifest line 157-158 when CI fails but borderline + H1(b) > 0"
        assert exit_code == 1  # 1 = REFUTE or INDETERMINATE (per analyzer docstring)

    def test_indeterminate_clause_NOT_triggered_when_not_borderline(self):
        """If CI fails NON-borderline (>1% margin), REFUTE not INDETERMINATE."""
        verdict, _ = _classify_verdict_r16e(
            h1_ci_ok=False,
            h1_mean_across_8_ok=True,
            h1_ic_floor_ok=True,
            h4_invariant_ok=True,
            h1_ci_borderline=False,  # NOT borderline
        )
        assert verdict == "REFUTE", \
            "Expected REFUTE when CI fails non-borderline (manifest clause requires borderline)"

    def test_indeterminate_clause_NOT_triggered_when_mean_across_8_fails(self):
        """INDETERMINATE clause requires H1(b) PASS — fails if mean across 8 ≤ 0."""
        verdict, _ = _classify_verdict_r16e(
            h1_ci_ok=False,
            h1_mean_across_8_ok=False,  # H1(b) FAIL
            h1_ic_floor_ok=True,
            h4_invariant_ok=True,
            h1_ci_borderline=True,  # CI borderline but H1(b) fails
        )
        assert verdict == "REFUTE", \
            "Expected REFUTE when H1(b) fails — INDETERMINATE clause requires BOTH borderline AND H1(b) PASS"

    def test_indeterminate_clause_subsumed_by_GO_when_ci_passes(self):
        """If CI passes (not borderline), normal GO path applies — borderline is irrelevant."""
        verdict, _ = _classify_verdict_r16e(
            h1_ci_ok=True,             # CI > 0 (GO eligibility)
            h1_mean_across_8_ok=True,
            h1_ic_floor_ok=True,
            h4_invariant_ok=True,
            h1_ci_borderline=True,  # borderline flag is irrelevant when CI passes
        )
        assert verdict == "GO", \
            "Expected GO when all H1 pass — borderline flag should be ignored"

    def test_indeterminate_clause_does_not_require_h1c_pass(self):
        """INDETERMINATE clause only requires H1(a) borderline + H1(b) PASS — H1(c) is silent.

        Manifest line 157-158 ("H1(a) borderline AND H1(b) > 0 → INDETERMINATE") is silent
        on H1(c). The analyzer's interpretation: INDETERMINATE means "low power, need
        more data". An IC floor failure doesn't preclude that (more seeds might lift IC
        above floor in R-16e-extended N=20). Deliberate design choice; locked here to
        prevent silent regression.

        Renamed 2026-05-14 mid-impl gate per Agent 1 review: original name said
        "still_refutes" but assertion is INDETERMINATE (name was inverted relative
        to behavior). This name matches the actual behavior.
        """
        verdict, _ = _classify_verdict_r16e(
            h1_ci_ok=False,
            h1_mean_across_8_ok=True,
            h1_ic_floor_ok=False,  # H1(c) FAIL — IC noise floor not met
            h4_invariant_ok=True,
            h1_ci_borderline=True,
        )
        assert verdict == "INDETERMINATE", \
            "Per current spec, INDETERMINATE only requires h1_ci borderline + h1_mean_across_8"


class TestExtractTestIc:
    """Test test_ic extraction from training records."""

    def test_valid_test_ic(self):
        record = {"training_metrics": {"test_ic": 0.147}}
        assert _extract_test_ic(record) == 0.147

    def test_missing_metrics_returns_none(self):
        assert _extract_test_ic({}) is None
        assert _extract_test_ic({"training_metrics": {}}) is None

    def test_nan_test_ic_returns_none(self):
        record = {"training_metrics": {"test_ic": float("nan")}}
        assert _extract_test_ic(record) is None

    def test_inf_test_ic_returns_none(self):
        record = {"training_metrics": {"test_ic": float("inf")}}
        assert _extract_test_ic(record) is None


class TestDataclassInvariants:
    """Frozen-dataclass + fingerprint-stability invariants."""

    def test_r16e_cell_result_frozen(self):
        cell = R16eCellResult(
            arm_label="test_arm",
            threshold_label="deep_itm_1.4bps",
            n_seeds=10, n_total_trades=500,
            mean_opt_ret=0.001, ci_low=-0.001, ci_high=0.003,
            n_nonfinite_replaced=0, block_length_used=10,
            drop_top5_per_seed=(0.0005, 0.0006), drop_top5_mean=0.00055,
            test_ic_per_seed=(0.06, 0.07), test_ic_mean=0.065,
            test_ic_ci_low=0.055, test_ic_ci_high=0.075,
            h4_bit_exact=None,
        )
        # Frozen — mutation must raise
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            cell.mean_opt_ret = 0.999

    def test_r16e_ratio_result_frozen(self):
        rr = R16eRatioResult(
            return_type="point_return",
            ridge_ic=0.147,
            tlob_ic_per_seed=(0.05, 0.06, 0.07),
            tlob_ic_mean=0.06,
            ratio_mean=2.45, ratio_ci_low=1.8, ratio_ci_high=3.1,
            ratio_floor_ok=True,
        )
        with pytest.raises(Exception):
            rr.ridge_ic = 999.0

    def test_outcome_frozen(self):
        outcome = R16eDecisionGateOutcome(
            verdict="GO",
            # H1 PRIMARY (manifest-aligned gates per #PY-208 fix)
            h1_ci_ok=True,
            h1_mean_across_8_ok=True,
            h1_ic_floor_ok=True,
            h1_ci_borderline=False,
            # H1 diagnostic (informational)
            h1_mean_ok=True,
            # H2 BASELINE
            h2_ratio_ok_point=True, h2_ratio_ok_smoothed=False,
            # H4 + H6
            h4_invariant_ok=True, h6_e8_confirmed=False,
            # Observed values
            h1_mean_observed=0.001,
            h1_mean_across_8_observed=0.002,
            h1_ci_low_observed=0.0002, h1_ci_high_observed=0.002,
            h1_ic_mean_observed=0.07, h1_ic_ci_low_observed=0.06,
            h2_ratio_mean_point=2.5, h2_ratio_ci_low_point=1.8,
            h2_ratio_mean_smoothed=1.1, h2_ratio_ci_low_smoothed=0.9,
            h6_smoothed_ridge_mean=-0.001, h6_smoothed_tlob_mean=0.002,
            h4_failed_cells=(),
            reasons=("test",), exit_code=0,
        )
        with pytest.raises(Exception):
            outcome.verdict = "ABORT"


class TestPublicSurface:
    """Lock the module's public surface (regression guard against accidental rename)."""

    def test_required_public_symbols_exist(self):
        required = {
            "CANONICAL_THRESHOLD_LABELS", "EXPECTED_GRID_POINTS", "EXPECTED_CELLS",
            "H1_TARGET_THRESHOLD", "H1_MEAN_FLOOR", "H1_IC_FLOOR", "H2_RATIO_FLOOR",
            "H1_PRIMARY_CELL", "H6_DIAGNOSTIC_CELLS",
            "R16eAnalysisError", "R16eIncompleteSweepError", "R16eH4InvariantError",
            "R16eCellResult", "R16eRatioResult", "R16eDecisionGateOutcome",
            "analyze_r16e_sweep", "render_verdict", "outcome_to_json_dict",
        }
        actual = set(dir(m))
        missing = required - actual
        assert not missing, f"Missing public symbols: {missing}"
