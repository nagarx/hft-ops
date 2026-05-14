"""
R-16e EXTENDED multi-seed IC + H60-HOLD backtest analysis (Phase 2 deliverable, 2026-05-14).

Pre-registered statistical analysis of the cycle9_r16e_multi_seed_h60_point
sweep produced by
``hft-ops/experiments/sweeps/cycle9_r16e_multi_seed_h60_point.yaml``.
Tests whether R-16d's Ridge×Point×H60 IC=0.147, Ratio=2.585 vs TLOB
finding validates to a tradeable edge at MATCHED H60-hold (closes the
R-16d H10-hold-only gap surfaced by Wave 4 Phase 2.a 3-agent pre-impl gate).

Cycle scope:
  - 3-axis grid: model_type {temporal_ridge, tlob} × return_type
    {point_return, smoothed_return} × train_seed {42..51} = 40 records
  - Backtest config: --hold-events 60 (NEW; R-16d used H10-hold uniformly)
  - Phase Y composability expectations (verified empirical R-16c precedent):
    22 unique experiment_provenance_hash post-dedup; 2 distinct
    compatibility_fingerprint (1 per return_type); 2 distinct
    model_config_hash (Ridge vs TLOB).

Pre-registered hypothesis gates per hft-rules §13 (committed BEFORE sweep
run):

  H1 PRIMARY (three-conjunctive at deep_itm_1.4bps, Ridge × Point × H60-hold):
    (a) mean OptRet > 0% AND
    (b) pooled per-trade bootstrap CI lower bound > 0 AND
    (c) per-seed test_ic CI lower bound > 0.05 (rejects noise floor)
    ANY ONE failing → REFUTE.

  H2 BASELINE (Ridge vs TLOB IC ratio at H60, paired-by-seed bootstrap):
    Ridge IC / TLOB_mean_IC > H2_RATIO_FLOOR (default 1.5). R-16d
    single-seed Ratio = 2.585. Multi-seed verification.

  H4 ARCHITECTURAL INVARIANT (Phase A.3 REDESIGN lock):
    Ridge × {point_return, smoothed_return} × 10 seeds produces 10
    bit-exact-identical predicted_returns.npy files per
    (model, return_type) cell. ABORT if any within-cell SHA differs.

  H6 LABEL-EXECUTION DIAGNOSTIC (E8 root-cause validation at H60-hold):
    Both smoothed × {Ridge, TLOB} arms produce mean OptRet ≤ 0% at
    H60-hold → E8 is STRUCTURAL (label-side), NOT hold-mismatch artifact.

Decision gate (committed BEFORE running per hft-rules §13):
  - GO: H1(a) AND H1(b) AND H1(c) at deep_itm_1.4bps AND H4 OK
  - REFUTE: H4 OK AND (any H1 conjunct fails)
  - INDETERMINATE: H4 OK AND borderline H1 → trigger R-16e-extended N=20
  - ABORT: H4 fails — Ridge non-determinism is architectural ship-blocker

Exit codes (mirror r16c_analysis):
  0 = GO; 1 = REFUTE or INDETERMINATE; 2 = ABORT; 3 = INCOMPLETE_SWEEP

**Architecture notes** (per Phase 2.a 3-agent design gate 2026-05-13 late):
  - Sibling to ``r16c_analysis.py`` in ``hft_ops.ledger`` subpackage.
  - FROM-IMPORTs 14 symbols from ``r16c_analysis``; ~80% code reuse.
  - NEW: ``_paired_bootstrap_ic_ratio`` primitive (Ridge_IC / TLOB_IC paired
    by seed) — NEEDED because Ridge bit-exact across seeds means Ridge IC
    is a single value while TLOB IC has 10-seed distribution.
  - NEW: H6 label-execution diagnostic on both return_type arms.
  - All atomic SSoT discipline preserved (hash_file, assert_finite_array).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from hft_contracts.provenance import hash_file
from hft_metrics._sanitize import assert_finite_array

from hft_ops.ledger.signal_dir import resolve_signal_dir
from hft_ops.paths import PipelinePaths

# =============================================================================
# FROM-IMPORT shared symbols from r16c_analysis.py (~80% reuse per Phase 2.a)
# =============================================================================
from hft_ops.ledger.r16c_analysis import (
    # Constants
    CANONICAL_THRESHOLD_LABELS,
    H4_MEAN_FLOOR,
    H1_DROP_TOP_K,
    N_BOOTSTRAP_DEFAULT,
    N_BLOCKS_MIN,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_INITIAL_CAPITAL,
    # Exception classes (subclassed below for distinct exit-code observability)
    R16cAnalysisError,
    R16cIncompleteSweepError,
    R16cH5InvariantError,
    # Helpers (single-array bootstrap + per-seed drop-top-K + path resolver + pnls loader)
    _pooled_block_bootstrap_mean_ci,
    _drop_top_k_by_abs_per_seed,
    _resolve_backtest_pnls_dir,
    _load_per_trade_pnls,
    _verify_h5_bit_exact,
)


# =============================================================================
# Pre-registered constants (LOCKED BEFORE sweep run per hft-rules §13)
# =============================================================================

# H1 PRIMARY target threshold + floors (Ridge × Point × H60-hold at deep_itm)
H1_TARGET_THRESHOLD: str = "deep_itm_1.4bps"
H1_MEAN_FLOOR: float = 0.0      # mean OptRet > 0% (looser than R-16c's +1%
                                 # since H1c IC floor is the gating constraint
                                 # for R-16e tradeability validation)
H1_IC_FLOOR: float = 0.05        # per-seed test_ic CI lower bound > 0.05
                                 # (rejects noise floor per CLAUDE.md
                                 # "predictive IC > 0.05 → tradeable signal")

# H2 BASELINE ratio floor — R-16d single-seed produced Ratio=2.585; multi-seed
# CI lower bound > 1.5 is the H2 acceptance gate
H2_RATIO_FLOOR: float = 1.5

# R-16e grid expectations (per cycle9_r16e manifest: 2 model × 2 return × 10 seed = 40)
EXPECTED_GRID_POINTS: int = 40

# 4 cells: (model_type, return_type) tuples. NOTE: differs from R-16c's
# {point_return, peak_return} — R-16e tests {point_return, smoothed_return}
# for E8 label-execution diagnostic.
EXPECTED_CELLS: Tuple[Tuple[str, str], ...] = (
    ("temporal_ridge", "point_return"),
    ("temporal_ridge", "smoothed_return"),
    ("tlob", "point_return"),
    ("tlob", "smoothed_return"),
)

# Subset for H1 PRIMARY focus arm (Ridge × Point × H60-hold)
H1_PRIMARY_CELL: Tuple[str, str] = ("temporal_ridge", "point_return")

# H6 E8 diagnostic arms (smoothed × {Ridge, TLOB})
H6_DIAGNOSTIC_CELLS: Tuple[Tuple[str, str], ...] = (
    ("temporal_ridge", "smoothed_return"),
    ("tlob", "smoothed_return"),
)


# =============================================================================
# Exception subclasses (distinct exit codes; inherit r16c hierarchy)
# =============================================================================


class R16eAnalysisError(R16cAnalysisError):
    """Base class for R-16e analysis failures. Inherits R16cAnalysisError
    so existing exit-code dispatch logic continues to work."""


class R16eIncompleteSweepError(R16cIncompleteSweepError, R16eAnalysisError):
    """Missing records or per-trade .npy files → exit 3."""


class R16eH4InvariantError(R16cH5InvariantError, R16eAnalysisError):
    """Ridge × N-seed bit-exact invariant violated → ABORT exit 2.

    Phase A.3 REDESIGN architectural lock: Ridge(alpha=...) is RNG-free.
    If 10 ridge × point or 10 ridge × smoothed records produce DIFFERENT
    predicted_returns.npy SHA-256 hashes, sklearn determinism claim is
    refuted. Fix REDESIGN before retrying R-16e.

    NOTE: numbered H4 in R-16e manifest (was H5 in R-16c manifest); same
    architectural concept, different number in hypothesis ordering.
    """


# =============================================================================
# Result dataclasses (frozen — fingerprint-stable per atomic_io conventions)
# =============================================================================


@dataclass(frozen=True)
class R16eCellResult:
    """Per-cell statistics for one (arm × threshold) combination at H60-hold.

    Cell = {(model_type, return_type) × threshold_label}. Stats are pooled
    across N seeds (10 for TLOB; 1 effective for Ridge since H4 implies
    identity). Extends R16cCellResult with test_ic-per-seed for H1(c)
    IC-floor verification.
    """
    arm_label: str               # e.g., "temporal_ridge__point_return"
    threshold_label: str         # e.g., "deep_itm_1.4bps"
    n_seeds: int                 # observed N seeds with valid records (≤10)
    n_total_trades: int          # pooled across N seeds at this threshold
    mean_opt_ret: float          # pooled per-trade mean (FRACTION units)
    ci_low: float                # 95% block-bootstrap CI lower bound
    ci_high: float               # 95% block-bootstrap CI upper bound
    n_nonfinite_replaced: int    # Cluster Z v0.1.9+ observability counter
    block_length_used: int       # actually-applied block_length (post-floor)
    drop_top5_per_seed: Tuple[float, ...]  # per-seed mean-after-drop-top-5
    drop_top5_mean: float        # mean across seeds (sensitivity diagnostic)
    test_ic_per_seed: Tuple[float, ...]  # per-seed test_ic (H1c IC floor)
    test_ic_mean: float          # mean test_ic across seeds (H1c statistic)
    test_ic_ci_low: float        # bootstrap CI on test_ic (H1c IC floor gate)
    test_ic_ci_high: float
    h4_bit_exact: Optional[bool]  # Ridge cells only; None for TLOB
    insufficient_data: bool = False


@dataclass(frozen=True)
class R16eRatioResult:
    """Ridge/TLOB IC ratio paired bootstrap result for H2 BASELINE gate.

    Pairs Ridge IC (single value due to bit-exact across seeds) with TLOB
    IC per seed (10 distinct values). Bootstraps the per-seed ratio
    Ridge_IC / TLOB_IC_seed_i to get CI on the ratio.

    NOTE: this differs from the R-16c "F7 falsification" paired bootstrap
    because Ridge has NO seed variance; only TLOB does. Effectively this
    is a 1-vs-N comparison.
    """
    return_type: str             # "point_return" or "smoothed_return"
    ridge_ic: float              # single value — Ridge bit-exact across seeds
    tlob_ic_per_seed: Tuple[float, ...]  # 10 distinct values
    tlob_ic_mean: float          # mean across 10 seeds
    ratio_mean: float            # ridge_ic / tlob_ic_mean
    ratio_ci_low: float          # bootstrap CI on ratio
    ratio_ci_high: float
    ratio_floor_ok: bool         # ratio_ci_low > H2_RATIO_FLOOR


@dataclass(frozen=True)
class R16eDecisionGateOutcome:
    """5-way verdict on R-16e sweep (mirrors R-16c outcome structure).

    H1 PRIMARY three-conjunctive at Ridge×Point×H60-hold deep_itm_1.4bps:
      (a) mean OptRet > 0%
      (b) pooled CI lower bound > 0
      (c) per-seed test_ic CI lower bound > H1_IC_FLOOR (0.05)
    H2 BASELINE: Ridge/TLOB IC ratio CI lower bound > 1.5
    H4 INVARIANT: Ridge bit-exact within each (model, return_type) cell
    H6 LABEL-EXEC DIAGNOSTIC: smoothed × {Ridge, TLOB} mean OptRet ≤ 0%
       confirms E8 is structural (informational, not GO/REFUTE gating)
    """
    verdict: Literal["GO", "REFUTE", "INDETERMINATE", "ABORT"]
    # H1 PRIMARY three-conjunctive
    h1_mean_ok: bool
    h1_ci_ok: bool
    h1_ic_floor_ok: bool          # per-seed test_ic CI lower bound > 0.05
    # H2 BASELINE
    h2_ratio_ok_point: bool       # Ridge/TLOB ratio CI > 1.5 at point_return
    h2_ratio_ok_smoothed: bool    # same at smoothed_return (informational)
    # H4 INVARIANT
    h4_invariant_ok: bool
    # H6 LABEL-EXEC DIAGNOSTIC (informational only)
    h6_e8_confirmed: bool         # both smoothed arms mean OptRet ≤ 0%
    # Diagnostic context (humans need to see WHY)
    h1_mean_observed: float
    h1_ci_low_observed: float
    h1_ci_high_observed: float
    h1_ic_mean_observed: float
    h1_ic_ci_low_observed: float
    h2_ratio_mean_point: float
    h2_ratio_ci_low_point: float
    h2_ratio_mean_smoothed: float
    h2_ratio_ci_low_smoothed: float
    h6_smoothed_ridge_mean: float
    h6_smoothed_tlob_mean: float
    h4_failed_cells: Tuple[str, ...]
    reasons: Tuple[str, ...]
    exit_code: int


# =============================================================================
# NEW R-16e primitive: paired bootstrap on Ridge/TLOB IC ratio
# =============================================================================


def _paired_bootstrap_ic_ratio(
    ridge_ic: float,
    tlob_ic_per_seed: np.ndarray,
    *,
    n_bootstraps: int = N_BOOTSTRAP_DEFAULT,
    ci: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI on Ridge IC / TLOB IC ratio.

    Ridge is bit-exact across seeds (Phase A.3 REDESIGN invariant) so its IC
    is a single deterministic value. TLOB has 10-seed variance. Bootstrap
    samples seeds WITH REPLACEMENT from the TLOB seed pool; ratio is
    computed as ``ridge_ic / mean(tlob_ic_sample)`` per bootstrap iteration.

    Args:
        ridge_ic: Single deterministic Ridge IC value (one per return_type).
        tlob_ic_per_seed: 10-element array of TLOB test_ic per seed.
        n_bootstraps: Number of bootstrap iterations. Default 10000.
        ci: Confidence interval level. Default 0.95.
        seed: RNG seed for reproducibility.

    Returns:
        ``(ratio_mean, ratio_ci_low, ratio_ci_high)`` — point estimate from
        full TLOB sample + percentile-based CI bounds.

    Raises:
        ValueError: tlob_ic_per_seed empty or contains non-finite values.
    """
    assert_finite_array(tlob_ic_per_seed, name="tlob_ic_per_seed")
    if len(tlob_ic_per_seed) == 0:
        raise ValueError("_paired_bootstrap_ic_ratio: empty tlob_ic_per_seed.")
    if not np.isfinite(ridge_ic):
        raise ValueError(f"_paired_bootstrap_ic_ratio: non-finite ridge_ic={ridge_ic}.")

    # Point estimate uses full TLOB sample mean
    tlob_ic_mean_full = float(np.mean(tlob_ic_per_seed))
    if abs(tlob_ic_mean_full) < 1e-12:
        # Avoid division by zero — return NaN ratio per §8 (don't silently
        # produce huge numbers)
        return float("nan"), float("nan"), float("nan")
    ratio_point = ridge_ic / tlob_ic_mean_full

    # Bootstrap distribution: sample seeds with replacement, compute ratio
    rng = np.random.default_rng(seed)
    n_seeds = len(tlob_ic_per_seed)
    boot_ratios = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        sample_idx = rng.integers(0, n_seeds, size=n_seeds)
        tlob_mean_boot = float(np.mean(tlob_ic_per_seed[sample_idx]))
        if abs(tlob_mean_boot) < 1e-12:
            boot_ratios[i] = ratio_point  # avoid div-by-zero in bootstrap
        else:
            boot_ratios[i] = ridge_ic / tlob_mean_boot

    # Percentile-based CI
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_ratios, 100 * alpha))
    ci_high = float(np.percentile(boot_ratios, 100 * (1 - alpha)))
    return ratio_point, ci_low, ci_high


# =============================================================================
# Helpers (extract test_ic + classify verdict)
# =============================================================================


def _extract_test_ic(record: Dict[str, Any]) -> Optional[float]:
    """Extract test_ic from a training record's training_metrics.

    Returns None if test_ic is missing or non-finite. Per Cluster Z (2026-05-11)
    spearman_ic returns NaN on too-few-samples; consumer must tolerate.
    """
    tm = record.get("training_metrics") or {}
    val = tm.get("test_ic")
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def _classify_verdict_r16e(
    *,
    h1_mean_ok: bool,
    h1_ci_ok: bool,
    h1_ic_floor_ok: bool,
    h4_invariant_ok: bool,
) -> Tuple[str, int]:
    """Classify R-16e decision-gate verdict.

    Per pre-registered decision logic in cycle9_r16e manifest:
    - GO: all H1 + H4
    - ABORT: H4 fails (architectural ship-blocker)
    - REFUTE: H4 OK AND any H1 fails
    """
    if not h4_invariant_ok:
        return "ABORT", 2
    if h1_mean_ok and h1_ci_ok and h1_ic_floor_ok:
        return "GO", 0
    return "REFUTE", 1


# =============================================================================
# Main analyzer
# =============================================================================


def analyze_r16e_sweep(
    sweep_id: str,
    ledger: Any,
    paths: PipelinePaths,
    *,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    allow_partial: bool = False,
    min_grid_points: int = EXPECTED_GRID_POINTS,
) -> Tuple[
    Dict[Tuple[str, str, str], R16eCellResult],
    Dict[str, R16eRatioResult],
    R16eDecisionGateOutcome,
]:
    """Analyze an R-16e sweep + render verdict.

    Args:
        sweep_id: Sweep aggregate identifier (e.g., cycle9_r16e_*_<timestamp>).
        ledger: ExperimentLedger instance.
        paths: PipelinePaths for output-dir resolution.
        n_bootstrap: Bootstrap iterations per cell. Default 10000.
        bootstrap_seed: RNG seed for reproducibility.
        initial_capital: Per-trade pnls DOLLAR→FRACTION conversion factor.
        allow_partial: Permit < EXPECTED_GRID_POINTS records (with warning).
        min_grid_points: Lower bound on accepted grid count if allow_partial.

    Returns:
        ``(cell_results, ratio_results, outcome)``.

    Raises:
        R16eIncompleteSweepError: missing records → exit 3.
        R16eH4InvariantError: Ridge non-determinism → exit 2 ABORT.
    """
    # =========================================================================
    # Load + group records by sweep_id
    # =========================================================================
    records = ledger.filter(sweep_id=sweep_id)
    grid_records = [
        r for r in records
        if r.get("record_type") not in ("sweep_aggregate", "sweep_failure")
    ]
    effective_threshold = min_grid_points if allow_partial else EXPECTED_GRID_POINTS
    if len(grid_records) < effective_threshold:
        raise R16eIncompleteSweepError(
            f"analyze_r16e_sweep: expected {effective_threshold} grid records; "
            f"got {len(grid_records)} for sweep_id={sweep_id}. Re-run failed "
            f"arms or pass allow_partial=True for explicit partial analysis."
        )
    if allow_partial and len(grid_records) < EXPECTED_GRID_POINTS:
        import warnings as _warnings
        _warnings.warn(
            f"PARTIAL R-16e analysis: {len(grid_records)}/{EXPECTED_GRID_POINTS} "
            f"grid records (allow_partial=True). Bootstrap CI may widen.",
            UserWarning,
            stacklevel=2,
        )

    # Group records into cells {(model_type, return_type): [records...]}
    cells_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in grid_records:
        axis_values = r.get("axis_values") or {}
        mt = axis_values.get("model_type")
        rt = axis_values.get("return_type")
        if mt is None or rt is None:
            tc = r.get("training_config") or {}
            mt = mt or tc.get("model", {}).get("model_type")
            data_labels = tc.get("data", {}).get("labels", {})
            rt = rt or data_labels.get("return_type")
        if mt is None or rt is None:
            raise R16eIncompleteSweepError(
                f"analyze_r16e_sweep: record {r.get('experiment_id', '<unknown>')} "
                f"missing model_type/return_type in axis_values + training_config."
            )
        cells_records.setdefault((mt, rt), []).append(r)

    missing_cells = set(EXPECTED_CELLS) - set(cells_records.keys())
    if missing_cells:
        raise R16eIncompleteSweepError(
            f"analyze_r16e_sweep: missing cells {sorted(missing_cells)}."
        )

    # =========================================================================
    # H4 INVARIANT (Ridge cells only)
    # =========================================================================
    h4_failed_cells: List[str] = []
    h4_passed_cell_keys: set = set()
    signal_dir_cache: Dict[str, Path] = {}

    def _cached_resolve_signal_dir(r: Dict[str, Any]) -> Path:
        exp_id = str(r.get("experiment_id", ""))
        if exp_id not in signal_dir_cache:
            signal_dir_cache[exp_id] = resolve_signal_dir(r, ledger, paths)
        return signal_dir_cache[exp_id]

    for cell_key in EXPECTED_CELLS:
        model_type, return_type = cell_key
        if model_type != "temporal_ridge":
            continue
        cell_records = cells_records[cell_key]
        sig_dirs = [_cached_resolve_signal_dir(r) for r in cell_records]
        all_equal, _diag = _verify_h5_bit_exact(
            arm_label=f"{model_type}__{return_type}",
            signal_dirs=sig_dirs,
        )
        if all_equal:
            h4_passed_cell_keys.add(cell_key)
        else:
            h4_failed_cells.append(f"{model_type}__{return_type}")

    h4_invariant_ok = not h4_failed_cells

    # =========================================================================
    # Per-cell × per-threshold analysis (mirror r16c structure)
    # =========================================================================
    cell_results: Dict[Tuple[str, str, str], R16eCellResult] = {}
    pnls_dir_resolved: Optional[Path] = None

    for cell_key in EXPECTED_CELLS:
        model_type, return_type = cell_key
        cell_records = cells_records[cell_key]
        if pnls_dir_resolved is None and cell_records:
            pnls_dir_resolved = _resolve_backtest_pnls_dir(
                cell_records[0], ledger, paths,
            )

        # Single-seed pooling for Ridge h4-passed cells (per R-16c Agent 1 fix)
        if cell_key in h4_passed_cell_keys:
            records_for_pooling = cell_records[:1]
        else:
            records_for_pooling = cell_records

        # test_ic per seed (always use all 10 records for IC analysis)
        test_ic_per_seed_list: List[float] = []
        for r in cell_records:
            ic = _extract_test_ic(r)
            if ic is not None:
                test_ic_per_seed_list.append(ic)
        test_ic_array = np.array(test_ic_per_seed_list, dtype=np.float64)

        # Bootstrap CI on test_ic
        if len(test_ic_array) >= 2:
            (
                test_ic_mean,
                test_ic_ci_lo,
                test_ic_ci_hi,
                _ic_nf,
                _ic_bl,
            ) = _pooled_block_bootstrap_mean_ci(
                test_ic_array, n_bootstraps=n_bootstrap, seed=bootstrap_seed,
            )
        elif len(test_ic_array) == 1:
            # Ridge single-record case — point estimate only
            test_ic_mean = float(test_ic_array[0])
            test_ic_ci_lo = test_ic_ci_hi = test_ic_mean
        else:
            test_ic_mean = test_ic_ci_lo = test_ic_ci_hi = float("nan")

        for threshold_label in CANONICAL_THRESHOLD_LABELS:
            per_seed_arrays: List[np.ndarray] = []
            for r in records_for_pooling:
                run_name = r.get("name") or r.get("experiment_id", "")
                record_status = r.get("status")
                arr = _load_per_trade_pnls(
                    pnls_dir_resolved,
                    run_name=str(run_name),
                    threshold_label=threshold_label,
                    record_status=record_status,
                    initial_capital=initial_capital,
                )
                if arr is not None and len(arr) > 0:
                    per_seed_arrays.append(arr)

            if not per_seed_arrays:
                cell_results[(model_type, return_type, threshold_label)] = R16eCellResult(
                    arm_label=f"{model_type}__{return_type}",
                    threshold_label=threshold_label,
                    n_seeds=0, n_total_trades=0,
                    mean_opt_ret=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"),
                    n_nonfinite_replaced=0, block_length_used=0,
                    drop_top5_per_seed=tuple(),
                    drop_top5_mean=float("nan"),
                    test_ic_per_seed=tuple(test_ic_per_seed_list),
                    test_ic_mean=test_ic_mean,
                    test_ic_ci_low=test_ic_ci_lo,
                    test_ic_ci_high=test_ic_ci_hi,
                    h4_bit_exact=(None if model_type != "temporal_ridge" else h4_invariant_ok),
                    insufficient_data=True,
                )
                continue

            pooled = np.concatenate(per_seed_arrays)
            insufficient = len(pooled) < max(5, 2 * N_BLOCKS_MIN)
            if insufficient:
                mean_pooled = float(np.mean(pooled))
                ci_lo = ci_hi = mean_pooled
                n_nonfinite = 0
                bl_used = 1
            else:
                mean_pooled, ci_lo, ci_hi, n_nonfinite, bl_used = _pooled_block_bootstrap_mean_ci(
                    pooled, n_bootstraps=n_bootstrap, seed=bootstrap_seed,
                )

            per_seed_drops, drop_top5_mean = _drop_top_k_by_abs_per_seed(
                per_seed_arrays, k=H1_DROP_TOP_K,
            )

            cell_results[(model_type, return_type, threshold_label)] = R16eCellResult(
                arm_label=f"{model_type}__{return_type}",
                threshold_label=threshold_label,
                n_seeds=len(per_seed_arrays),
                n_total_trades=int(len(pooled)),
                mean_opt_ret=mean_pooled,
                ci_low=ci_lo, ci_high=ci_hi,
                n_nonfinite_replaced=n_nonfinite,
                block_length_used=bl_used,
                drop_top5_per_seed=per_seed_drops,
                drop_top5_mean=drop_top5_mean,
                test_ic_per_seed=tuple(test_ic_per_seed_list),
                test_ic_mean=test_ic_mean,
                test_ic_ci_low=test_ic_ci_lo,
                test_ic_ci_high=test_ic_ci_hi,
                h4_bit_exact=(None if model_type != "temporal_ridge" else h4_invariant_ok),
                insufficient_data=insufficient,
            )

    # =========================================================================
    # H2 BASELINE: Ridge vs TLOB IC ratio (paired bootstrap per return_type)
    # =========================================================================
    ratio_results: Dict[str, R16eRatioResult] = {}
    for rt in ("point_return", "smoothed_return"):
        ridge_records = cells_records.get(("temporal_ridge", rt), [])
        tlob_records = cells_records.get(("tlob", rt), [])
        ridge_ic_values = [_extract_test_ic(r) for r in ridge_records]
        ridge_ic_values = [v for v in ridge_ic_values if v is not None]
        tlob_ic_values = [_extract_test_ic(r) for r in tlob_records]
        tlob_ic_values = [v for v in tlob_ic_values if v is not None]

        if not ridge_ic_values or len(tlob_ic_values) < 2:
            ratio_results[rt] = R16eRatioResult(
                return_type=rt,
                ridge_ic=float("nan") if not ridge_ic_values else float(ridge_ic_values[0]),
                tlob_ic_per_seed=tuple(tlob_ic_values),
                tlob_ic_mean=float("nan"),
                ratio_mean=float("nan"),
                ratio_ci_low=float("nan"),
                ratio_ci_high=float("nan"),
                ratio_floor_ok=False,
            )
            continue

        # Ridge is bit-exact → use first value as the deterministic Ridge IC
        ridge_ic = float(ridge_ic_values[0])
        tlob_arr = np.array(tlob_ic_values, dtype=np.float64)
        tlob_ic_mean = float(np.mean(tlob_arr))
        ratio_mean, ratio_lo, ratio_hi = _paired_bootstrap_ic_ratio(
            ridge_ic, tlob_arr,
            n_bootstraps=n_bootstrap, seed=bootstrap_seed,
        )
        ratio_results[rt] = R16eRatioResult(
            return_type=rt,
            ridge_ic=ridge_ic,
            tlob_ic_per_seed=tuple(tlob_ic_values),
            tlob_ic_mean=tlob_ic_mean,
            ratio_mean=ratio_mean,
            ratio_ci_low=ratio_lo,
            ratio_ci_high=ratio_hi,
            ratio_floor_ok=(np.isfinite(ratio_lo) and ratio_lo > H2_RATIO_FLOOR),
        )

    # =========================================================================
    # H1 PRIMARY: Ridge × Point × H60-hold at deep_itm_1.4bps
    # =========================================================================
    primary_cell_key = (*H1_PRIMARY_CELL, H1_TARGET_THRESHOLD)
    primary_cell = cell_results.get(primary_cell_key)
    if primary_cell is None or primary_cell.insufficient_data:
        h1_mean_ok = h1_ci_ok = h1_ic_floor_ok = False
        h1_mean_obs = h1_ci_low_obs = h1_ci_high_obs = float("nan")
        h1_ic_mean_obs = h1_ic_ci_low_obs = float("nan")
    else:
        h1_mean_obs = primary_cell.mean_opt_ret
        h1_ci_low_obs = primary_cell.ci_low
        h1_ci_high_obs = primary_cell.ci_high
        h1_ic_mean_obs = primary_cell.test_ic_mean
        h1_ic_ci_low_obs = primary_cell.test_ic_ci_low
        h1_mean_ok = h1_mean_obs > H1_MEAN_FLOOR
        h1_ci_ok = h1_ci_low_obs > 0.0
        h1_ic_floor_ok = h1_ic_ci_low_obs > H1_IC_FLOOR

    # =========================================================================
    # H6 LABEL-EXECUTION DIAGNOSTIC (informational)
    # =========================================================================
    h6_means: Dict[str, float] = {}
    for (mt, rt) in H6_DIAGNOSTIC_CELLS:
        cell = cell_results.get((mt, rt, H1_TARGET_THRESHOLD))
        if cell is not None and not cell.insufficient_data:
            h6_means[mt] = cell.mean_opt_ret
        else:
            h6_means[mt] = float("nan")
    h6_smoothed_ridge_mean = h6_means.get("temporal_ridge", float("nan"))
    h6_smoothed_tlob_mean = h6_means.get("tlob", float("nan"))
    # E8 confirmed if BOTH smoothed arms produce mean ≤ 0% at H60-hold
    h6_e8_confirmed = (
        np.isfinite(h6_smoothed_ridge_mean)
        and np.isfinite(h6_smoothed_tlob_mean)
        and h6_smoothed_ridge_mean <= 0.0
        and h6_smoothed_tlob_mean <= 0.0
    )

    # =========================================================================
    # Verdict classification + reasons
    # =========================================================================
    verdict, exit_code = _classify_verdict_r16e(
        h1_mean_ok=h1_mean_ok,
        h1_ci_ok=h1_ci_ok,
        h1_ic_floor_ok=h1_ic_floor_ok,
        h4_invariant_ok=h4_invariant_ok,
    )

    reasons: List[str] = []
    if not h4_invariant_ok:
        reasons.append(
            f"H4 FAILED (ABORT): Ridge bit-exact invariant violated in cells "
            f"{h4_failed_cells}. Phase A.3 REDESIGN ship-blocker."
        )
    if h1_mean_ok:
        reasons.append(f"H1a PASS: mean OptRet={h1_mean_obs:.6f} > +{H1_MEAN_FLOOR:.4f}")
    else:
        reasons.append(f"H1a FAIL: mean OptRet={h1_mean_obs:.6f} ≤ +{H1_MEAN_FLOOR:.4f}")
    if h1_ci_ok:
        reasons.append(f"H1b PASS: CI=({h1_ci_low_obs:.6f}, {h1_ci_high_obs:.6f}); lower > 0")
    else:
        reasons.append(f"H1b FAIL: CI=({h1_ci_low_obs:.6f}, {h1_ci_high_obs:.6f}); lower ≤ 0")
    if h1_ic_floor_ok:
        reasons.append(f"H1c PASS: test_ic CI lower={h1_ic_ci_low_obs:.4f} > {H1_IC_FLOOR:.2f}")
    else:
        reasons.append(f"H1c FAIL: test_ic CI lower={h1_ic_ci_low_obs:.4f} ≤ {H1_IC_FLOOR:.2f}")
    # H2 informational reasons
    rr_point = ratio_results.get("point_return")
    rr_smoothed = ratio_results.get("smoothed_return")
    h2_ratio_ok_point = rr_point.ratio_floor_ok if rr_point else False
    h2_ratio_ok_smoothed = rr_smoothed.ratio_floor_ok if rr_smoothed else False
    if rr_point:
        reasons.append(
            f"H2 RATIO (point_return): {rr_point.ratio_mean:.3f}× "
            f"[CI {rr_point.ratio_ci_low:.3f}, {rr_point.ratio_ci_high:.3f}] "
            f"{'PASS' if h2_ratio_ok_point else 'FAIL'} (floor {H2_RATIO_FLOOR})"
        )
    if rr_smoothed:
        reasons.append(
            f"H2 RATIO (smoothed_return): {rr_smoothed.ratio_mean:.3f}× "
            f"[CI {rr_smoothed.ratio_ci_low:.3f}, {rr_smoothed.ratio_ci_high:.3f}] "
            f"{'PASS' if h2_ratio_ok_smoothed else 'FAIL'} (informational)"
        )
    if h6_e8_confirmed:
        reasons.append(
            f"H6 E8 CONFIRMED: smoothed × Ridge mean={h6_smoothed_ridge_mean:.6f} ≤ 0, "
            f"smoothed × TLOB mean={h6_smoothed_tlob_mean:.6f} ≤ 0. "
            f"Label-execution mismatch is STRUCTURAL (label-side), NOT hold-mismatch artifact."
        )
    else:
        reasons.append(
            f"H6 E8 NOT CONFIRMED: smoothed × Ridge={h6_smoothed_ridge_mean:.6f}, "
            f"smoothed × TLOB={h6_smoothed_tlob_mean:.6f}. At least one smoothed arm "
            f"produces non-negative mean at H60-hold."
        )

    outcome = R16eDecisionGateOutcome(
        verdict=verdict,
        h1_mean_ok=h1_mean_ok,
        h1_ci_ok=h1_ci_ok,
        h1_ic_floor_ok=h1_ic_floor_ok,
        h2_ratio_ok_point=h2_ratio_ok_point,
        h2_ratio_ok_smoothed=h2_ratio_ok_smoothed,
        h4_invariant_ok=h4_invariant_ok,
        h6_e8_confirmed=h6_e8_confirmed,
        h1_mean_observed=h1_mean_obs,
        h1_ci_low_observed=h1_ci_low_obs,
        h1_ci_high_observed=h1_ci_high_obs,
        h1_ic_mean_observed=h1_ic_mean_obs,
        h1_ic_ci_low_observed=h1_ic_ci_low_obs,
        h2_ratio_mean_point=rr_point.ratio_mean if rr_point else float("nan"),
        h2_ratio_ci_low_point=rr_point.ratio_ci_low if rr_point else float("nan"),
        h2_ratio_mean_smoothed=rr_smoothed.ratio_mean if rr_smoothed else float("nan"),
        h2_ratio_ci_low_smoothed=rr_smoothed.ratio_ci_low if rr_smoothed else float("nan"),
        h6_smoothed_ridge_mean=h6_smoothed_ridge_mean,
        h6_smoothed_tlob_mean=h6_smoothed_tlob_mean,
        h4_failed_cells=tuple(h4_failed_cells),
        reasons=tuple(reasons),
        exit_code=exit_code,
    )
    return cell_results, ratio_results, outcome


# =============================================================================
# Verdict rendering (human-readable + JSON)
# =============================================================================


def render_verdict(
    outcome: R16eDecisionGateOutcome,
    cell_results: Dict[Tuple[str, str, str], R16eCellResult],
    ratio_results: Dict[str, R16eRatioResult],
) -> str:
    """Render a human-readable verdict block.

    Includes: verdict header, all 4 hypothesis gates with PASS/FAIL + observed
    values, H6 E8 diagnostic block, per-cell summary table for the H1 PRIMARY
    threshold across all 4 cells.
    """
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"R-16e EXTENDED VERDICT: {outcome.verdict} (exit_code={outcome.exit_code})")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Pre-registered hypothesis gates:")
    for reason in outcome.reasons:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append("Per-cell summary at H1_TARGET_THRESHOLD = " + H1_TARGET_THRESHOLD + ":")
    lines.append("-" * 78)
    lines.append(f"{'Cell':<40} {'n_seeds':>8} {'n_trades':>10} {'mean':>12} {'CI':>20}")
    lines.append("-" * 78)
    for (mt, rt) in EXPECTED_CELLS:
        cell = cell_results.get((mt, rt, H1_TARGET_THRESHOLD))
        if cell is None:
            continue
        ci_str = f"({cell.ci_low:.5f},{cell.ci_high:.5f})"
        lines.append(
            f"{mt}__{rt:<25} {cell.n_seeds:>8} {cell.n_total_trades:>10} "
            f"{cell.mean_opt_ret:>12.6f} {ci_str:>20}"
        )
    lines.append("")
    lines.append("H2 Ridge/TLOB IC ratio (paired bootstrap):")
    for rt, rr in ratio_results.items():
        lines.append(
            f"  {rt}: Ridge_IC={rr.ridge_ic:.4f}, TLOB_IC_mean={rr.tlob_ic_mean:.4f}, "
            f"Ratio={rr.ratio_mean:.3f} CI=({rr.ratio_ci_low:.3f},{rr.ratio_ci_high:.3f}) "
            f"{'PASS' if rr.ratio_floor_ok else 'FAIL'} (floor {H2_RATIO_FLOOR})"
        )
    lines.append("")
    lines.append("H6 E8 LABEL-EXECUTION DIAGNOSTIC (informational):")
    lines.append(f"  Smoothed × Ridge mean OptRet @ H60-hold: {outcome.h6_smoothed_ridge_mean:.6f}")
    lines.append(f"  Smoothed × TLOB mean OptRet @ H60-hold:  {outcome.h6_smoothed_tlob_mean:.6f}")
    lines.append(f"  E8 confirmed (both ≤ 0): {outcome.h6_e8_confirmed}")
    lines.append("=" * 78)
    return "\n".join(lines)


def outcome_to_json_dict(outcome: R16eDecisionGateOutcome) -> Dict[str, Any]:
    """Serialize outcome to a JSON-compatible dict for archival."""
    return {
        "verdict": outcome.verdict,
        "exit_code": outcome.exit_code,
        "h1": {
            "mean_ok": outcome.h1_mean_ok,
            "ci_ok": outcome.h1_ci_ok,
            "ic_floor_ok": outcome.h1_ic_floor_ok,
            "mean_observed": outcome.h1_mean_observed,
            "ci_low_observed": outcome.h1_ci_low_observed,
            "ci_high_observed": outcome.h1_ci_high_observed,
            "ic_mean_observed": outcome.h1_ic_mean_observed,
            "ic_ci_low_observed": outcome.h1_ic_ci_low_observed,
        },
        "h2": {
            "ratio_ok_point": outcome.h2_ratio_ok_point,
            "ratio_ok_smoothed": outcome.h2_ratio_ok_smoothed,
            "ratio_mean_point": outcome.h2_ratio_mean_point,
            "ratio_ci_low_point": outcome.h2_ratio_ci_low_point,
            "ratio_mean_smoothed": outcome.h2_ratio_mean_smoothed,
            "ratio_ci_low_smoothed": outcome.h2_ratio_ci_low_smoothed,
        },
        "h4": {
            "invariant_ok": outcome.h4_invariant_ok,
            "failed_cells": list(outcome.h4_failed_cells),
        },
        "h6": {
            "e8_confirmed": outcome.h6_e8_confirmed,
            "smoothed_ridge_mean": outcome.h6_smoothed_ridge_mean,
            "smoothed_tlob_mean": outcome.h6_smoothed_tlob_mean,
        },
        "reasons": list(outcome.reasons),
    }
