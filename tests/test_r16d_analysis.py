"""
Unit tests for hft_ops.ledger.r16d_analysis (Cycle 8 / Phase B.5, 2026-05-13).

Covers the R-16d-specific primitives + verdict classifier + dataclass invariants:
- Constants (12 cells, 4 arms, 3 horizons, baseline ratio 0.80)
- _verify_h5_horizon_distinct (NEW horizon-axis activation invariant)
- _extract_test_ic (fallback chain across ExperimentRecord schema evolution)
- _classify_verdict_r16d (5-gate matrix)
- R16dCellResult / R16dArmDecayResult / R16dDecisionGateOutcome dataclass roundtrip
- Reused R-16c helpers smoke (delegates via from-import per §0 reuse-first)

Per Phase B.5 design: 28 tests across 7 test classes covering R-16d-specific
logic; relies on test_r16c_analysis.py for shared helper coverage (no
duplication).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from hft_ops.ledger.r16d_analysis import (
    EXPECTED_ARMS,
    EXPECTED_CELLS,
    EXPECTED_GRID_POINTS,
    EXPECTED_HORIZON_VALUES,
    EXPECTED_MODEL_TYPES,
    EXPECTED_RETURN_TYPES,
    H1_PASS_THRESHOLD_ARMS,
    H1_REFUTE_THRESHOLD_ARMS,
    H1_TOTAL_ARMS,
    H2_BASELINE_RATIO,
    H2_PASS_THRESHOLD_CELLS,
    H2_TOTAL_CELLS,
    H4_PASS_THRESHOLD_ARMS,
    R16dAnalysisError,
    R16dArmDecayResult,
    R16dBaselineCellResult,
    R16dCellResult,
    R16dDecisionGateOutcome,
    R16dH5HorizonAxisError,
    R16dIncompleteSweepError,
    _classify_verdict_r16d,
    _extract_test_ic,
    _verify_h5_horizon_distinct,
    outcome_to_json_dict,
    render_verdict,
)


# =============================================================================
# R-16d-specific constants invariants (12 cells, 4 arms, 3 horizons)
# =============================================================================


class TestR16dConstants:
    """Lock pre-registered R-16d constants per manifest L139-167."""

    def test_expected_grid_points_is_12(self):
        """2 model × 2 return × 3 horizon = 12 per manifest L18."""
        assert EXPECTED_GRID_POINTS == 12

    def test_expected_cells_is_cartesian_12(self):
        """Cartesian product over model × return × horizon."""
        assert len(EXPECTED_CELLS) == 12
        # Every (model, return, horizon) triple is present
        for mt in EXPECTED_MODEL_TYPES:
            for rt in EXPECTED_RETURN_TYPES:
                for h in EXPECTED_HORIZON_VALUES:
                    assert (mt, rt, h) in EXPECTED_CELLS

    def test_expected_arms_excludes_horizon(self):
        """Arms are (model, return) — horizon is iterated within each arm."""
        assert len(EXPECTED_ARMS) == 4
        assert ("temporal_ridge", "point_return") in EXPECTED_ARMS
        assert ("temporal_ridge", "smoothed_return") in EXPECTED_ARMS
        assert ("tlob", "point_return") in EXPECTED_ARMS
        assert ("tlob", "smoothed_return") in EXPECTED_ARMS

    def test_expected_return_types_excludes_peak(self):
        """R-16c REFUTED peak; R-16d drops it per manifest L41-44."""
        assert "peak_return" not in EXPECTED_RETURN_TYPES
        assert EXPECTED_RETURN_TYPES == ("point_return", "smoothed_return")

    def test_expected_horizons_includes_h300(self):
        """H300 included per user-selected preview (NOT tradeable but tested)."""
        assert EXPECTED_HORIZON_VALUES == (10, 60, 300)

    def test_h1_thresholds(self):
        """H1 pass = ≥3/4 arms decay; refute = <2/4."""
        assert H1_PASS_THRESHOLD_ARMS == 3
        assert H1_REFUTE_THRESHOLD_ARMS == 2
        assert H1_TOTAL_ARMS == 4

    def test_h2_baseline_ratio_is_0_80(self):
        """Ridge captures ≥80% TLOB IC per CLAUDE.md '91%' finding."""
        assert H2_BASELINE_RATIO == 0.80

    def test_h2_total_cells_is_6(self):
        """2 return × 3 horizon = 6 cells for H2 baseline test."""
        assert H2_TOTAL_CELLS == 6
        assert H2_PASS_THRESHOLD_CELLS == 4  # ≥4/6 pass


# =============================================================================
# _verify_h5_horizon_distinct (NEW R-16d-specific invariant)
# =============================================================================


class TestVerifyH5HorizonDistinct:
    """R-16d-specific: Ridge × <return> across 3 horizons must produce DISTINCT
    predicted_returns.npy SHAs. Tests horizon-axis activation invariant.
    """

    def test_three_distinct_files_returns_true(self, tmp_path):
        """3 horizon dirs with different content → all-distinct → PASS."""
        horizon_dirs: Dict[int, Path] = {}
        for h, content in [(10, b"H10_predictions"), (60, b"H60_predictions"),
                           (300, b"H300_predictions")]:
            d = tmp_path / f"H{h}"
            d.mkdir()
            # Write distinct content as predicted_returns.npy (mock; just bytes)
            np.save(d / "predicted_returns.npy", np.frombuffer(content, dtype=np.uint8))
            horizon_dirs[h] = d
        ok, diag = _verify_h5_horizon_distinct("test_arm", horizon_dirs)
        assert ok is True
        assert diag is None

    def test_two_horizons_collide_returns_false_with_diagnostic(self, tmp_path):
        """H10 + H60 with SAME content → COLLISION → FAIL with diagnostic."""
        same_arr = np.array([1.0, 2.0, 3.0])
        horizon_dirs: Dict[int, Path] = {}
        for h in (10, 60, 300):
            d = tmp_path / f"H{h}"
            d.mkdir()
            if h == 300:
                np.save(d / "predicted_returns.npy", same_arr * 2)  # distinct
            else:
                np.save(d / "predicted_returns.npy", same_arr)  # collision
            horizon_dirs[h] = d
        ok, diag = _verify_h5_horizon_distinct("ridge__point", horizon_dirs)
        assert ok is False
        assert diag is not None
        # Diagnostic should cite COLLIDE + horizon-axis-cosmetic likelihood
        assert "COLLIDE" in diag or "collide" in diag.lower()
        assert "horizon" in diag.lower()

    def test_missing_predicted_returns_returns_false(self, tmp_path):
        """Missing predicted_returns.npy in any horizon dir → FAIL."""
        d10 = tmp_path / "H10"
        d10.mkdir()
        np.save(d10 / "predicted_returns.npy", np.array([1.0]))
        d60 = tmp_path / "H60"
        d60.mkdir()
        # No predicted_returns.npy in d60
        horizon_dirs = {10: d10, 60: d60}
        ok, diag = _verify_h5_horizon_distinct("test", horizon_dirs)
        assert ok is False
        assert "missing" in diag.lower()

    def test_empty_input_returns_true_no_op(self):
        """Edge case: empty dict → no-op PASS."""
        ok, diag = _verify_h5_horizon_distinct("empty_arm", {})
        assert ok is True
        assert "no-op" in (diag or "")


# =============================================================================
# _extract_test_ic (fallback chain across schema evolution)
# =============================================================================


class TestExtractTestIc:
    """Tests defensive multi-key fallback for test_ic extraction."""

    def test_canonical_training_metrics_path(self):
        """Standard Phase 7+ path: record.training_metrics.test_ic."""
        record = {"training_metrics": {"test_ic": 0.3289}}
        assert _extract_test_ic(record) == pytest.approx(0.3289)

    def test_legacy_metrics_path(self):
        """Pre-Phase-7 fallback: record.metrics.test_ic."""
        record = {"metrics": {"test_ic": 0.1473}}
        assert _extract_test_ic(record) == pytest.approx(0.1473)

    def test_captured_metrics_path(self):
        """Sweep-level fallback: record.captured_metrics.test_ic."""
        record = {"captured_metrics": {"test_ic": 0.0466}}
        assert _extract_test_ic(record) == pytest.approx(0.0466)

    def test_priority_order_canonical_wins(self):
        """When multiple sources have test_ic, training_metrics wins."""
        record = {
            "training_metrics": {"test_ic": 0.5},
            "metrics": {"test_ic": 0.3},
            "captured_metrics": {"test_ic": 0.1},
        }
        assert _extract_test_ic(record) == pytest.approx(0.5)

    def test_missing_returns_nan(self):
        """All sources missing → NaN (downstream treats as insufficient-data)."""
        record = {"some_other_field": "no test_ic"}
        assert math.isnan(_extract_test_ic(record))

    def test_non_numeric_value_falls_through(self):
        """Non-numeric in one source falls through to next."""
        record = {
            "training_metrics": {"test_ic": "not_a_number"},
            "metrics": {"test_ic": 0.42},
        }
        # First source has non-numeric; fall through to next
        assert _extract_test_ic(record) == pytest.approx(0.42)


# =============================================================================
# _classify_verdict_r16d (5-gate matrix)
# =============================================================================


class TestClassifyVerdictR16d:
    """Tests 5-gate verdict classification (H1/H2/H4/H5) → 4-way + exit code."""

    def test_all_pass_returns_go(self):
        verdict, code = _classify_verdict_r16d(
            h1_ok=True, h2_ok=True, h4_ok=True, h5_ok=True,
            h1_arms_passing=4,
        )
        assert verdict == "GO"
        assert code == 0

    def test_h5_fail_returns_abort_regardless(self):
        """H5 = ABORT exit 2; horizon axis cosmetic."""
        verdict, code = _classify_verdict_r16d(
            h1_ok=True, h2_ok=True, h4_ok=True, h5_ok=False,
            h1_arms_passing=4,
        )
        assert verdict == "ABORT"
        assert code == 2

    def test_h1_zero_arms_refute(self):
        """H1 = 0/4 arms decay → REFUTE."""
        verdict, code = _classify_verdict_r16d(
            h1_ok=False, h2_ok=False, h4_ok=False, h5_ok=True,
            h1_arms_passing=0,
        )
        assert verdict == "REFUTE"
        assert code == 1

    def test_h1_two_arms_indeterminate(self):
        """H1 = 2/4 (borderline) → INDETERMINATE."""
        verdict, code = _classify_verdict_r16d(
            h1_ok=False, h2_ok=True, h4_ok=True, h5_ok=True,
            h1_arms_passing=2,
        )
        assert verdict == "INDETERMINATE"
        assert code == 1

    def test_h1_three_arms_but_h2_fails_refute(self):
        """H1 passes (≥3) but H2 fails → REFUTE (baseline gate)."""
        verdict, code = _classify_verdict_r16d(
            h1_ok=True, h2_ok=False, h4_ok=True, h5_ok=True,
            h1_arms_passing=3,
        )
        assert verdict == "REFUTE"
        assert code == 1

    def test_h1_three_arms_but_h4_fails_refute(self):
        """H1 passes but H4 fails → REFUTE."""
        verdict, code = _classify_verdict_r16d(
            h1_ok=True, h2_ok=True, h4_ok=False, h5_ok=True,
            h1_arms_passing=3,
        )
        assert verdict == "REFUTE"
        assert code == 1


# =============================================================================
# Dataclass invariants
# =============================================================================


class TestR16dDataclasses:
    """Frozen dataclass + roundtrip stability."""

    def test_r16d_cell_result_is_frozen(self):
        c = R16dCellResult(
            arm_label="test", horizon_value=10, threshold_label="deep_itm_1.4bps",
            test_ic=0.1, n_total_trades=100, mean_opt_ret=0.01,
            ci_low=-0.01, ci_high=0.02, n_nonfinite_replaced=0,
            block_length_used=5, drop_top5_mean=0.005,
            h5_horizon_distinct=True, insufficient_data=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            c.arm_label = "mutated"  # frozen rejects

    def test_r16d_arm_decay_result_monotonic_field(self):
        d = R16dArmDecayResult(
            arm_label="tlob__smoothed",
            test_ic_H10=0.30, test_ic_H60=0.15, test_ic_H300=0.05,
            decay_H10_H60=True, decay_H60_H300=True, monotonic_decay=True,
        )
        assert d.monotonic_decay is True

    def test_r16d_decision_gate_outcome_serializable(self):
        outcome = R16dDecisionGateOutcome(
            verdict="GO",
            h1_arms_passing=4, h1_arms_total=4, h1_ok=True,
            h2_cells_passing=5, h2_cells_total=6, h2_ok=True,
            h3_arms_clearing_breakeven=3, h3_ok=True,
            h4_arms_clearing_floor=3, h4_ok=True,
            h5_ridge_arms_distinct=2, h5_ok=True,
            h5_failed_arms=(),
            arm_decays=(),
            baseline_cells=(),
            reasons=("H1 PASS", "H2 PASS"),
            exit_code=0,
        )
        d = outcome_to_json_dict(outcome, {})
        assert d["verdict"] == "GO"
        assert d["exit_code"] == 0
        assert d["gates"]["h1_ok"] is True


# =============================================================================
# render_verdict smoke (output structure)
# =============================================================================


class TestRenderVerdict:
    """Verdict rendering produces non-empty multi-line output."""

    def test_render_includes_verdict_banner(self):
        outcome = R16dDecisionGateOutcome(
            verdict="REFUTE",
            h1_arms_passing=1, h1_arms_total=4, h1_ok=False,
            h2_cells_passing=2, h2_cells_total=6, h2_ok=False,
            h3_arms_clearing_breakeven=1, h3_ok=False,
            h4_arms_clearing_floor=1, h4_ok=False,
            h5_ridge_arms_distinct=2, h5_ok=True,
            h5_failed_arms=(),
            arm_decays=(
                R16dArmDecayResult(
                    arm_label="tlob__point_return",
                    test_ic_H10=0.01, test_ic_H60=0.02, test_ic_H300=0.005,
                    decay_H10_H60=False, decay_H60_H300=True, monotonic_decay=False,
                ),
            ),
            baseline_cells=(
                R16dBaselineCellResult(
                    return_type="point_return", horizon_value=10,
                    test_ic_ridge=0.3, test_ic_tlob=0.37, ratio=0.81, h2_pass=True,
                ),
            ),
            reasons=("H1 REFUTE: only 1/4 arms decay",),
            exit_code=1,
        )
        rendered = render_verdict(outcome, {})
        assert "R-16d" in rendered
        assert "REFUTE" in rendered
        assert "H1 PRIMARY" in rendered
        assert "H2 BASELINE" in rendered
        assert "tlob__point_return" in rendered
