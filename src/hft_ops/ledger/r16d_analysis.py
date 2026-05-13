"""
R-16d horizon-axis sweep analysis (Cycle 8, Phase B.3 deliverable, 2026-05-13).

Pre-registered statistical analysis of the cycle8_r16d_horizon_axis sweep
produced by ``hft-ops/experiments/sweeps/cycle8_r16d_horizon_axis.yaml``.
Tests horizon-decay hypothesis (H1) on v3p0 corpus + baseline (H2) + cost
(H3) + negative control (H4) + Ridge architectural invariant (H5) +
label-execution alignment diagnostic (H6).

- **H1 PRIMARY (Horizon Decay)**:
  For each (model_type × return_type) arm (4 arms total):
    test_ic(H10) > test_ic(H60) AND test_ic(H60) > test_ic(H300)
  GO: ≥3/4 arms exhibit strict monotonic decay
  REFUTE: <2/4 arms exhibit decay
  INDETERMINATE: 2/4 arms exhibit decay

- **H2 BASELINE GATE (per hft-rules §13)**:
  For each (return_type × horizon) cell (6 cells):
    test_ic(temporal_ridge) ≥ 0.80 × test_ic(tlob)
  PASS-THRESHOLD: ≥4/6 cells satisfy ratio.

- **H3 COST GATE**: Auto-applied by backtester (median |pred| > 1.4 bps).
  PASS-THRESHOLD: ≥2/4 arms (per horizon) clear breakeven.

- **H4 NEGATIVE CONTROL**: For each arm at H10: mean OptRet across 8
  thresholds > -0.5% (mirrors R-16c convention).
  PASS-THRESHOLD: ≥2/4 arms clear floor at H10.

- **H5 ARCHITECTURAL INVARIANT (R-16d-specific)**:
  For each Ridge × (return_type) arm, the 3 horizon-cells' SHA-256 of
  predicted_returns.npy must be ALL DISTINCT. Tests that horizon axis
  is architecturally active (NOT cosmetically collapsed like the closed
  #PY-87/#PY-88 shadow-precedence bug class).
  FAIL = ABORT (horizon axis is silently cosmetic — ship-blocker).

- **H6 LABEL-EXECUTION DIAGNOSTIC (E8 closure test)**:
  point_return_ic / smoothed_return_ic ratio per (model, horizon) cell.
  Informational — does NOT gate verdict.

**Decision gate** (committed BEFORE running per hft-rules §13):

- GO: H1 ≥3/4 + H2 ≥4/6 + H4 ≥2/4 at H10 + H5 OK
- REFUTE: H1 <2/4 arms (closes horizon-decay-as-alpha for v3p0)
- INDETERMINATE: H1 = 2/4 arms (trigger R-16d-extended N=4 seeds × 12 cells)
- ABORT: H5 fails — horizon axis cosmetic

**Exit codes** (per R-16c precedent):
- 0 = GO
- 1 = REFUTE or INDETERMINATE (legitimate research outcomes)
- 2 = ABORT (horizon axis cosmetic — architectural ship-blocker)
- 3 = INCOMPLETE_SWEEP (missing records or per-trade .npy files)

**Architectural pattern** (per Phase B.3 ground-truth design 2026-05-13):
- FROM-IMPORT strategy (NOT subclass — frozen dataclass shape rejects
  horizon field). Imports SHARED HELPERS via ``from hft_ops.ledger.
  r16c_analysis import ...`` per hft-rules §0 reuse-first.
- 11 shared symbols reused as-is (4 constants + 4 exception classes +
  4 helper functions, with 1 overlap):
  ``CANONICAL_THRESHOLD_LABELS``, ``H4_MEAN_FLOOR``, ``H1_DROP_TOP_K``,
  ``N_BOOTSTRAP_DEFAULT``, ``N_BLOCKS_MIN``, ``DEFAULT_BOOTSTRAP_SEED``,
  ``DEFAULT_INITIAL_CAPITAL``, ``R16cAnalysisError``,
  ``R16cIncompleteSweepError``, ``R16cH5InvariantError``,
  ``_pooled_block_bootstrap_mean_ci``, ``_drop_top_k_by_abs_per_seed``,
  ``_resolve_backtest_pnls_dir``, ``_load_per_trade_pnls``.
- NEW for R-16d:
  - ``EXPECTED_GRID_POINTS=12``, ``EXPECTED_CELLS`` 3-tuple keyed by
    ``(model_type, return_type, horizon_value)``.
  - ``R16dCellResult`` adds ``horizon_value`` + ``test_ic`` fields.
  - ``R16dArmDecayResult`` per-arm horizon-decay diagnostic.
  - ``R16dDecisionGateOutcome`` 5-gate verdict (H1/H2/H3/H4/H5).
  - ``_verify_h5_horizon_distinct`` — NEW invariant: Ridge cells at
    different horizons MUST produce DISTINCT predicted_returns.npy SHAs
    (horizon axis IS architecturally active, NOT cosmetic).

5 fail-loud invariants (mirror R-16c discipline per hft-rules §8):

1. FILE-COUNT: expected_records × 8_thresholds .npy files
2. FINITENESS: assert_finite_array(opt_trade_pnls) per cell
3. N-SIZE-FLOOR: n_pooled >= max(5, 2*block_length); else INSUFFICIENT_DATA
4. H5-HORIZON-DISTINCT: Ridge × <return> across {H10, H60, H300} → ALL
   DISTINCT SHAs (NEW R-16d invariant)
5. THRESHOLD-CANON: exact match of 8 canonical threshold labels
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from hft_contracts.provenance import hash_file

from hft_ops.ledger.signal_dir import resolve_signal_dir
from hft_ops.paths import PipelinePaths

# Reuse R-16c helpers per hft-rules §0 reuse-first (Strategy: COPY-and-import)
from hft_ops.ledger.r16c_analysis import (
    CANONICAL_THRESHOLD_LABELS,
    H4_MEAN_FLOOR,
    H1_DROP_TOP_K,
    N_BOOTSTRAP_DEFAULT,
    N_BLOCKS_MIN,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_INITIAL_CAPITAL,
    R16cAnalysisError,
    R16cIncompleteSweepError,
    R16cH5InvariantError,
    _pooled_block_bootstrap_mean_ci,
    _drop_top_k_by_abs_per_seed,
    _resolve_backtest_pnls_dir,
    _load_per_trade_pnls,
    # NOTE: _verify_h5_bit_exact NOT imported — R-16d is single-seed so the
    # R-16c within-cell bit-exact check is N/A. R-16d uses
    # `_verify_h5_horizon_distinct` (NEW, defined below) which tests the
    # horizon-axis-activation invariant instead. If a future R-16d-extended
    # adds multi-seed, import `_verify_h5_bit_exact` and apply per cell.
)


# =============================================================================
# Pre-registered constants (R-16d specific; locked BEFORE sweep run per §13)
# =============================================================================

# Cell grid: 2 model × 2 return × 3 horizon = 12 grid points
EXPECTED_GRID_POINTS: int = 12

EXPECTED_HORIZON_VALUES: Tuple[int, ...] = (10, 60, 300)
EXPECTED_RETURN_TYPES: Tuple[str, ...] = ("point_return", "smoothed_return")
EXPECTED_MODEL_TYPES: Tuple[str, ...] = ("temporal_ridge", "tlob")

# 12 cells: (model_type, return_type, horizon_value) Cartesian product
EXPECTED_CELLS: Tuple[Tuple[str, str, int], ...] = tuple(
    (mt, rt, h)
    for mt in EXPECTED_MODEL_TYPES
    for rt in EXPECTED_RETURN_TYPES
    for h in EXPECTED_HORIZON_VALUES
)

# 4 arms: (model_type, return_type) — H1 decay test iterates over arms
EXPECTED_ARMS: Tuple[Tuple[str, str], ...] = tuple(
    (mt, rt) for mt in EXPECTED_MODEL_TYPES for rt in EXPECTED_RETURN_TYPES
)

# H1 horizon-decay pre-registered thresholds (per manifest L139-146)
H1_PASS_THRESHOLD_ARMS: int = 3   # ≥3/4 arms must decay
H1_REFUTE_THRESHOLD_ARMS: int = 2  # <2/4 arms = REFUTE
H1_TOTAL_ARMS: int = 4

# H2 baseline gate (per manifest L150-156)
H2_BASELINE_RATIO: float = 0.80    # Ridge must capture ≥80% TLOB IC
H2_PASS_THRESHOLD_CELLS: int = 4   # ≥4/6 cells satisfy ratio
H2_TOTAL_CELLS: int = 6            # 2 return × 3 horizon

# H3 cost gate (auto-applied; analyzer reports informational)
H3_BREAKEVEN_BPS: float = 1.4      # Deep ITM per OpraCalibratedCosts
H3_PASS_THRESHOLD_ARMS: int = 2    # ≥2/4 arms clear

# H4 negative control (per manifest L165-167)
H4_PASS_THRESHOLD_ARMS: int = 2    # ≥2/4 arms at H10 clear floor

# Aliased exception classes (delegate to R-16c via subclass for distinct names
# in tracebacks; same exit-code semantics)
class R16dAnalysisError(R16cAnalysisError):
    """Base class for R-16d analysis failures.

    Subclass exit codes:
    - R16dIncompleteSweepError → exit code 3
    - R16dH5HorizonAxisError → exit code 2 (ABORT — horizon axis cosmetic)
    """


class R16dIncompleteSweepError(R16cIncompleteSweepError, R16dAnalysisError):
    """Missing records / per-trade files → exit 3."""


class R16dH5HorizonAxisError(R16cH5InvariantError, R16dAnalysisError):
    """Ridge × <return> across {H10, H60, H300} produced duplicate SHAs.

    Indicates horizon axis is COSMETICALLY collapsed (silent shadow-precedence
    bug class — same root cause as closed #PY-87/#PY-88). Fix the horizon
    override path before retrying R-16d.
    """


# =============================================================================
# Result dataclasses (frozen — fingerprint-stable)
# =============================================================================


@dataclass(frozen=True)
class R16dCellResult:
    """Per-cell statistics for one (arm × horizon × threshold) combination.

    Cell = {(model_type, return_type, horizon_value) × threshold_label}.
    Test IC sourced from record.training_metrics. Pooled trade pnls from
    backtester producer-side dump.
    """
    arm_label: str               # e.g., "temporal_ridge__point_return"
    horizon_value: int            # 10 / 60 / 300
    threshold_label: str          # e.g., "deep_itm_1.4bps"
    test_ic: float                # from record.training_metrics.test_ic
    n_total_trades: int           # backtest at this threshold
    mean_opt_ret: float           # pooled bootstrap mean (FRACTION of capital)
    ci_low: float                 # 95% CI lower bound
    ci_high: float                # 95% CI upper bound
    n_nonfinite_replaced: int     # bootstrap observability counter
    block_length_used: int        # post-floor block_length
    drop_top5_mean: float         # H4 robustness statistic
    h5_horizon_distinct: Optional[bool]  # Ridge only; None for TLOB or non-cell-level
    insufficient_data: bool = False


@dataclass(frozen=True)
class R16dArmDecayResult:
    """Per-arm (model × return) horizon-decay test result.

    For each arm, compute test_ic at each of {H10, H60, H300} + check
    strict monotonic decay (H10 > H60 > H300).
    """
    arm_label: str
    test_ic_H10: float
    test_ic_H60: float
    test_ic_H300: float
    decay_H10_H60: bool           # test_ic(H10) > test_ic(H60)
    decay_H60_H300: bool          # test_ic(H60) > test_ic(H300)
    monotonic_decay: bool         # both strictly decay


@dataclass(frozen=True)
class R16dBaselineCellResult:
    """Per-(return_type × horizon_value) cell H2 baseline test.

    Tests Ridge IC ≥ 0.80 × TLOB IC. Refutes if TLOB ic is materially
    larger than Ridge in this cell.
    """
    return_type: str
    horizon_value: int
    test_ic_ridge: float
    test_ic_tlob: float
    ratio: float                  # ridge / tlob (NaN if tlob ≤ 0)
    h2_pass: bool                 # ratio ≥ 0.80


@dataclass(frozen=True)
class R16dDecisionGateOutcome:
    """5-gate verdict on R-16d horizon-axis sweep."""
    verdict: Literal["GO", "REFUTE", "INDETERMINATE", "ABORT"]
    # H1 PRIMARY (horizon decay)
    h1_arms_passing: int          # 0-4
    h1_arms_total: int            # 4
    h1_ok: bool                   # ≥3/4 arms decay
    # H2 BASELINE
    h2_cells_passing: int         # 0-6
    h2_cells_total: int           # 6
    h2_ok: bool                   # ≥4/6 cells
    # H3 COST (informational; backtester auto-applies)
    h3_arms_clearing_breakeven: int
    h3_ok: bool
    # H4 NEGATIVE CONTROL (at H10)
    h4_arms_clearing_floor: int
    h4_ok: bool
    # H5 ARCHITECTURAL (Ridge horizon-axis distinct SHAs)
    h5_ridge_arms_distinct: int   # of 2 Ridge arms
    h5_ok: bool
    h5_failed_arms: Tuple[str, ...]
    # Per-arm decay diagnostics
    arm_decays: Tuple[R16dArmDecayResult, ...]
    # Per-(return × horizon) baseline diagnostics
    baseline_cells: Tuple[R16dBaselineCellResult, ...]
    reasons: Tuple[str, ...]
    exit_code: int


# =============================================================================
# H5 horizon-axis distinct verification (R-16d-specific invariant)
# =============================================================================


def _verify_h5_horizon_distinct(
    arm_label: str,
    horizon_signal_dirs: Dict[int, Path],
) -> Tuple[bool, Optional[str]]:
    """Verify that a Ridge arm's 3 horizon cells produce DISTINCT
    predicted_returns.npy SHA-256.

    R-16d-specific invariant: tests that horizon axis IS architecturally
    active. If 2 or 3 horizons produce the SAME SHA, the horizon axis has
    collapsed cosmetically (same shadow-precedence bug class as the closed
    #PY-87/#PY-88).

    Args:
        arm_label: e.g., "temporal_ridge__point_return"
        horizon_signal_dirs: Dict[horizon_value → signal_dir]. Expected size 3
            for {H10, H60, H300}.

    Returns:
        ``(all_distinct, diagnostic_message)``. On failure, message
        enumerates the SHA prefixes per horizon + which horizons collide.
    """
    if not horizon_signal_dirs:
        return True, "no-op (0 horizon signal dirs)"
    shas: Dict[int, str] = {}
    for horizon, sig_dir in sorted(horizon_signal_dirs.items()):
        pred_path = sig_dir / "predicted_returns.npy"
        if not pred_path.exists():
            return False, f"{arm_label}: missing predicted_returns.npy for H{horizon}"
        shas[horizon] = hash_file(pred_path, missing_ok=False)
    distinct_shas = set(shas.values())
    if len(distinct_shas) == len(shas):
        return True, None
    # Build diagnostic: enumerate which horizons share SHAs
    sha_to_horizons: Dict[str, List[int]] = {}
    for h, sh in shas.items():
        sha_to_horizons.setdefault(sh, []).append(h)
    diag_parts = [
        f"{arm_label}: only {len(distinct_shas)} distinct SHAs across "
        f"{len(shas)} horizons (expected ALL distinct — horizon axis is "
        f"architecturally active when each horizon produces different labels)",
    ]
    for sh, horizons in sorted(sha_to_horizons.items()):
        horizon_str = ",".join(f"H{h}" for h in sorted(horizons))
        if len(horizons) > 1:
            diag_parts.append(f"  COLLIDE {horizon_str}: {sh[:16]}...")
        else:
            diag_parts.append(f"  {horizon_str}: {sh[:16]}...")
    diag_parts.append(
        f"  Likely cause: horizon axis override silently failed; "
        f"check sweep.axes[horizon] overrides include all 3 fields "
        f"(data.horizon_idx + data.labels.primary_horizon_idx + horizon_value)."
    )
    return False, "\n".join(diag_parts)


# =============================================================================
# Test IC extraction (from training records)
# =============================================================================


def _extract_test_ic(record: Dict[str, Any]) -> float:
    """Extract test_ic scalar from a training record.

    Defensive fallback across multiple known locations (different stages
    write to slightly different keys per ExperimentRecord schema evolution):
    1. record.training_metrics.test_ic (canonical post Phase 7 Stage 7.4)
    2. record.metrics.test_ic (legacy)
    3. record.captured_metrics.test_ic (sweep-level)

    Returns NaN if all locations missing — caller treats NaN as
    insufficient-data per N_SIZE_FLOOR semantics.
    """
    candidates = [
        record.get("training_metrics", {}),
        record.get("metrics", {}),
        record.get("captured_metrics", {}),
    ]
    for source in candidates:
        if isinstance(source, dict):
            val = source.get("test_ic")
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return float("nan")


# =============================================================================
# Verdict classification (5-gate matrix for R-16d)
# =============================================================================


def _classify_verdict_r16d(
    *,
    h1_ok: bool,
    h2_ok: bool,
    h4_ok: bool,
    h5_ok: bool,
    h1_arms_passing: int,
) -> Tuple[Literal["GO", "REFUTE", "INDETERMINATE", "ABORT"], int]:
    """Map 5 gates → 4-way verdict + exit code.

    Per manifest L173-188 decision-gate table:
    - H5 fails → ABORT (exit 2): horizon axis cosmetic.
    - H1 ≥3/4 AND H2 ≥4/6 AND H4 ≥2/4 → GO (exit 0).
    - H1 <2/4 → REFUTE (exit 1): closes horizon-decay-as-alpha for v3p0.
    - H1 = 2/4 AND other gates borderline → INDETERMINATE (exit 1):
      trigger R-16d-extended N=4 seeds × 12 cells for power.
    - Otherwise → REFUTE (exit 1).
    """
    if not h5_ok:
        return "ABORT", 2
    if h1_ok and h2_ok and h4_ok:
        return "GO", 0
    # H1 < 2/4 = clear REFUTE
    if h1_arms_passing < H1_REFUTE_THRESHOLD_ARMS:
        return "REFUTE", 1
    # H1 = 2/4 (borderline) → INDETERMINATE
    if h1_arms_passing == H1_REFUTE_THRESHOLD_ARMS:
        return "INDETERMINATE", 1
    # H1 ≥3/4 but H2 or H4 failed → REFUTE (decay observed but gates not cleared)
    return "REFUTE", 1


# =============================================================================
# Main entry: analyze_r16d_sweep
# =============================================================================


def analyze_r16d_sweep(
    sweep_id: str,
    ledger: Any,
    paths: PipelinePaths,
    *,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    allow_partial: bool = False,
    min_grid_points: int = EXPECTED_GRID_POINTS,
) -> Tuple[Dict[Tuple[str, str, int, str], R16dCellResult], R16dDecisionGateOutcome]:
    """Analyze an R-16d horizon-axis sweep + render verdict.

    Args:
        sweep_id: Sweep aggregate identifier (e.g.,
            ``cycle8_r16d_horizon_axis_20260513T200000``).
        ledger: ``hft_ops.ledger.ledger.ExperimentLedger`` instance.
        paths: ``PipelinePaths`` for output-dir resolution.
        n_bootstrap: Bootstrap iterations per cell. Default 10000.
        bootstrap_seed: RNG seed for reproducibility. Default 42.
        initial_capital: DOLLAR→FRACTION normalization (default 100000.0).
        allow_partial: If True, permit fewer than EXPECTED_GRID_POINTS
            (e.g., cross-sweep dedup). Default False (strict).
        min_grid_points: Floor when allow_partial=True. Default 12.

    Returns:
        ``(cell_results, outcome)`` — dict keyed by
        ``(model_type, return_type, horizon_value, threshold_label)`` ×
        ``R16dDecisionGateOutcome`` with 4-way verdict.

    Raises:
        R16dIncompleteSweepError: missing records / .npy files. Exit code 3.
        R16dH5HorizonAxisError: horizon axis cosmetic. Exit code 2.
    """
    # Filter ledger by sweep_id; skip parent sweep_aggregate / sweep_failure
    records = ledger.filter(sweep_id=sweep_id)
    grid_records = [
        r for r in records
        if r.get("record_type") not in ("sweep_aggregate", "sweep_failure")
    ]
    effective_threshold = min_grid_points if allow_partial else EXPECTED_GRID_POINTS
    if len(grid_records) < effective_threshold:
        raise R16dIncompleteSweepError(
            f"analyze_r16d_sweep: expected {effective_threshold} grid records; "
            f"got {len(grid_records)} for sweep_id={sweep_id}. Re-run failed "
            f"arms or accept PARTIAL banner via separate explicit invocation. "
            f"Per hft-rules §8 fail-loud."
        )
    if allow_partial and len(grid_records) < EXPECTED_GRID_POINTS:
        import warnings as _warnings
        _warnings.warn(
            f"PARTIAL R-16d analysis: {len(grid_records)}/{EXPECTED_GRID_POINTS} "
            f"grid records (allow_partial=True). Reduced sample-per-cell may "
            f"widen bootstrap CI. Document rationale (typical cause: cross-sweep "
            f"dedup against earlier sweep records).",
            UserWarning,
            stacklevel=2,
        )

    # Group records into cells {(model_type, return_type, horizon_value): record}
    # Single-seed → 1 record per cell (vs R-16c's 10 records per cell).
    cells_records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for r in grid_records:
        axis_values = r.get("axis_values") or {}
        mt = axis_values.get("model_type")
        rt = axis_values.get("return_type")
        # horizon axis may surface as `horizon` (axis name) or `horizon_value`
        horizon = axis_values.get("horizon") or axis_values.get("horizon_value")
        # Fall back to training_config introspection
        if mt is None or rt is None or horizon is None:
            tc = r.get("training_config") or {}
            mt = mt or tc.get("model", {}).get("model_type")
            data_labels = tc.get("data", {}).get("labels", {})
            rt = rt or data_labels.get("return_type")
            # horizon_value derives from primary_horizon_idx of horizons list
            data_block = tc.get("data", {})
            phi = data_labels.get("primary_horizon_idx", data_block.get("horizon_idx"))
            horizons_list = data_labels.get("horizons", [])
            if horizon is None and phi is not None and horizons_list:
                try:
                    horizon = int(horizons_list[int(phi)])
                except (IndexError, TypeError, ValueError):
                    pass
        if mt is None or rt is None or horizon is None:
            raise R16dIncompleteSweepError(
                f"analyze_r16d_sweep: record {r.get('experiment_id', '<unknown>')} "
                f"missing model_type/return_type/horizon_value in axis_values + "
                f"training_config (rt={rt!r}, mt={mt!r}, horizon={horizon!r})."
            )
        # Coerce horizon to int (axis values may be strings or labels like 'H10').
        # Sweep stores axis LABEL (e.g., 'H10', 'H60', 'H300') in axis_values["horizon"],
        # not the override value. Strip leading 'H' if present then coerce.
        if isinstance(horizon, str) and horizon.upper().startswith("H"):
            horizon_stripped = horizon[1:]
        else:
            horizon_stripped = horizon
        try:
            horizon = int(horizon_stripped)
        except (TypeError, ValueError):
            raise R16dIncompleteSweepError(
                f"analyze_r16d_sweep: record {r.get('experiment_id', '<unknown>')} "
                f"horizon value {horizon!r} is not coercible to int (tried "
                f"H-prefix strip → {horizon_stripped!r})."
            )
        key = (str(mt), str(rt), horizon)
        if key in cells_records:
            # Duplicate (model, return, horizon) suggests sweep dedup misbehavior
            raise R16dIncompleteSweepError(
                f"analyze_r16d_sweep: duplicate cell {key}; check seed-axis "
                f"is not active (R-16d is single-seed by design)."
            )
        cells_records[key] = r

    # Verify all 12 cells present (when not allow_partial)
    missing_cells = set(EXPECTED_CELLS) - set(cells_records.keys())
    if missing_cells and not allow_partial:
        raise R16dIncompleteSweepError(
            f"analyze_r16d_sweep: missing cells {sorted(missing_cells)} in sweep."
        )

    # =========================================================================
    # H5 HORIZON-DISTINCT (Ridge arms only; NEW R-16d-specific invariant)
    # =========================================================================
    h5_failed_arms: List[str] = []
    h5_diagnostics: List[str] = []
    h5_distinct_count = 0

    # Cache signal_dir resolutions
    signal_dir_cache: Dict[str, Path] = {}

    def _cached_resolve_signal_dir(r: Dict[str, Any]) -> Path:
        exp_id = str(r.get("experiment_id", ""))
        if exp_id not in signal_dir_cache:
            signal_dir_cache[exp_id] = resolve_signal_dir(r, ledger, paths)
        return signal_dir_cache[exp_id]

    for return_type in EXPECTED_RETURN_TYPES:
        # Ridge arm: collect signal dirs across 3 horizons
        arm_horizon_dirs: Dict[int, Path] = {}
        for horizon in EXPECTED_HORIZON_VALUES:
            cell_key = ("temporal_ridge", return_type, horizon)
            if cell_key in cells_records:
                arm_horizon_dirs[horizon] = _cached_resolve_signal_dir(
                    cells_records[cell_key]
                )
        arm_label = f"temporal_ridge__{return_type}"
        all_distinct, diag = _verify_h5_horizon_distinct(
            arm_label=arm_label,
            horizon_signal_dirs=arm_horizon_dirs,
        )
        if all_distinct:
            h5_distinct_count += 1
        else:
            h5_failed_arms.append(arm_label)
            if diag is not None:
                h5_diagnostics.append(diag)

    h5_ok = not h5_failed_arms  # NEW: ALL Ridge arms must have distinct SHAs

    # =========================================================================
    # Per-cell extraction: test_ic + per-trade pnls + bootstrap
    # =========================================================================
    cell_results: Dict[Tuple[str, str, int, str], R16dCellResult] = {}
    pnls_dir_resolved: Optional[Path] = None
    # Pre-fetch test_ic per cell (used by H1 + H2)
    cell_test_ic: Dict[Tuple[str, str, int], float] = {}

    for cell_key in EXPECTED_CELLS:
        if cell_key not in cells_records:
            continue
        record = cells_records[cell_key]
        model_type, return_type, horizon_value = cell_key
        cell_test_ic[cell_key] = _extract_test_ic(record)

        # Resolve per-trade pnls dir lazily
        if pnls_dir_resolved is None:
            pnls_dir_resolved = _resolve_backtest_pnls_dir(record, ledger, paths)

        run_name = str(record.get("name") or record.get("experiment_id", ""))
        record_status = record.get("status")
        # Is this Ridge cell in h5-passed-arm scope (single seed, trivially yes)?
        ridge_h5_pass = (
            model_type == "temporal_ridge"
            and f"{model_type}__{return_type}" not in h5_failed_arms
        )

        for threshold_label in CANONICAL_THRESHOLD_LABELS:
            arr = _load_per_trade_pnls(
                pnls_dir_resolved,
                run_name=run_name,
                threshold_label=threshold_label,
                record_status=record_status,
                initial_capital=initial_capital,
            )
            # Single-seed: 1 array per cell (vs R-16c's 10 per cell)
            per_seed_arrays: List[np.ndarray] = []
            if arr is not None and len(arr) > 0:
                per_seed_arrays.append(arr)
            # N_SIZE_FLOOR check
            if not per_seed_arrays:
                cell_results[(*cell_key, threshold_label)] = R16dCellResult(
                    arm_label=f"{model_type}__{return_type}",
                    horizon_value=horizon_value,
                    threshold_label=threshold_label,
                    test_ic=cell_test_ic[cell_key],
                    n_total_trades=0,
                    mean_opt_ret=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"),
                    n_nonfinite_replaced=0, block_length_used=0,
                    drop_top5_mean=float("nan"),
                    h5_horizon_distinct=(ridge_h5_pass if model_type == "temporal_ridge" else None),
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
                    pooled,
                    n_bootstraps=n_bootstrap,
                    seed=bootstrap_seed,
                )
            _, drop_top5_mean = _drop_top_k_by_abs_per_seed(
                per_seed_arrays, k=H1_DROP_TOP_K,
            )
            cell_results[(*cell_key, threshold_label)] = R16dCellResult(
                arm_label=f"{model_type}__{return_type}",
                horizon_value=horizon_value,
                threshold_label=threshold_label,
                test_ic=cell_test_ic[cell_key],
                n_total_trades=int(len(pooled)),
                mean_opt_ret=mean_pooled,
                ci_low=ci_lo, ci_high=ci_hi,
                n_nonfinite_replaced=n_nonfinite,
                block_length_used=bl_used,
                drop_top5_mean=drop_top5_mean,
                h5_horizon_distinct=(ridge_h5_pass if model_type == "temporal_ridge" else None),
                insufficient_data=insufficient,
            )

    # =========================================================================
    # H1 PRIMARY: per-arm horizon-decay test
    # =========================================================================
    arm_decays: List[R16dArmDecayResult] = []
    h1_arms_passing = 0
    for arm_key in EXPECTED_ARMS:
        model_type, return_type = arm_key
        ic_H10 = cell_test_ic.get((model_type, return_type, 10), float("nan"))
        ic_H60 = cell_test_ic.get((model_type, return_type, 60), float("nan"))
        ic_H300 = cell_test_ic.get((model_type, return_type, 300), float("nan"))
        decay_10_60 = bool(np.isfinite(ic_H10) and np.isfinite(ic_H60) and ic_H10 > ic_H60)
        decay_60_300 = bool(np.isfinite(ic_H60) and np.isfinite(ic_H300) and ic_H60 > ic_H300)
        monotonic = decay_10_60 and decay_60_300
        if monotonic:
            h1_arms_passing += 1
        arm_decays.append(R16dArmDecayResult(
            arm_label=f"{model_type}__{return_type}",
            test_ic_H10=ic_H10,
            test_ic_H60=ic_H60,
            test_ic_H300=ic_H300,
            decay_H10_H60=decay_10_60,
            decay_H60_H300=decay_60_300,
            monotonic_decay=monotonic,
        ))
    h1_ok = h1_arms_passing >= H1_PASS_THRESHOLD_ARMS

    # =========================================================================
    # H2 BASELINE: Ridge ≥ 0.80 × TLOB per (return_type × horizon)
    # =========================================================================
    baseline_cells: List[R16dBaselineCellResult] = []
    h2_cells_passing = 0
    for return_type in EXPECTED_RETURN_TYPES:
        for horizon in EXPECTED_HORIZON_VALUES:
            ic_ridge = cell_test_ic.get(("temporal_ridge", return_type, horizon), float("nan"))
            ic_tlob = cell_test_ic.get(("tlob", return_type, horizon), float("nan"))
            # Ratio: ridge / tlob. Guard against TLOB IC ≤ 0 (rare; ratio undefined).
            # When tlob ic is non-positive, ratio is NaN and H2 marked as FAIL for this cell.
            if np.isfinite(ic_ridge) and np.isfinite(ic_tlob) and ic_tlob > 1e-6:
                ratio = ic_ridge / ic_tlob
                h2_pass = ratio >= H2_BASELINE_RATIO
            else:
                ratio = float("nan")
                h2_pass = False
            if h2_pass:
                h2_cells_passing += 1
            baseline_cells.append(R16dBaselineCellResult(
                return_type=return_type,
                horizon_value=horizon,
                test_ic_ridge=ic_ridge,
                test_ic_tlob=ic_tlob,
                ratio=ratio,
                h2_pass=h2_pass,
            ))
    h2_ok = h2_cells_passing >= H2_PASS_THRESHOLD_CELLS

    # =========================================================================
    # H3 COST GATE: median |prediction| > 1.4 bps (informational from backtester)
    # =========================================================================
    # Backtester auto-applies via OpraCalibratedCosts. We infer "cleared" from
    # presence of trades at deep_itm_1.4bps threshold (≥1 trade implies clear).
    h3_arms_clearing = 0
    for arm_key in EXPECTED_ARMS:
        # Check H10 cell at deep_itm_1.4bps (cost-aware tradeable horizon)
        cell = cell_results.get((*arm_key, 10, "deep_itm_1.4bps"))
        if cell is not None and not cell.insufficient_data and cell.n_total_trades > 0:
            h3_arms_clearing += 1
    h3_ok = h3_arms_clearing >= H3_PASS_THRESHOLD_ARMS

    # =========================================================================
    # H4 NEGATIVE CONTROL: per-arm mean OptRet across 8 thresholds at H10
    # =========================================================================
    h4_arms_clearing = 0
    arm_h4_means: Dict[Tuple[str, str], float] = {}
    for arm_key in EXPECTED_ARMS:
        model_type, return_type = arm_key
        h4_means: List[float] = []
        for tl in CANONICAL_THRESHOLD_LABELS:
            c = cell_results.get((model_type, return_type, 10, tl))
            if c is not None and not c.insufficient_data:
                h4_means.append(c.mean_opt_ret)
        if h4_means:
            mean_across = float(np.mean(h4_means))
            arm_h4_means[arm_key] = mean_across
            if mean_across > H4_MEAN_FLOOR:
                h4_arms_clearing += 1
    h4_ok = h4_arms_clearing >= H4_PASS_THRESHOLD_ARMS

    # =========================================================================
    # Verdict classification
    # =========================================================================
    verdict, exit_code = _classify_verdict_r16d(
        h1_ok=h1_ok,
        h2_ok=h2_ok,
        h4_ok=h4_ok,
        h5_ok=h5_ok,
        h1_arms_passing=h1_arms_passing,
    )

    # Reasons
    reasons: List[str] = []
    if not h5_ok:
        reasons.append(
            f"H5 FAILED: horizon axis cosmetic in arms {h5_failed_arms}. "
            f"Likely shadow-precedence bug (same class as closed #PY-87/#PY-88). "
            f"Check sweep.axes[horizon] override paths."
        )
    if h1_ok:
        reasons.append(
            f"H1 PASS: {h1_arms_passing}/{H1_TOTAL_ARMS} arms exhibit monotonic "
            f"horizon-decay (≥{H1_PASS_THRESHOLD_ARMS} required)."
        )
    elif h1_arms_passing < H1_REFUTE_THRESHOLD_ARMS:
        reasons.append(
            f"H1 REFUTE: only {h1_arms_passing}/{H1_TOTAL_ARMS} arms decay "
            f"(<{H1_REFUTE_THRESHOLD_ARMS}). Closes horizon-decay-as-alpha for v3p0."
        )
    else:
        reasons.append(
            f"H1 INDETERMINATE: {h1_arms_passing}/{H1_TOTAL_ARMS} arms decay "
            f"(borderline). Trigger R-16d-extended N=4 seeds × 12 cells."
        )
    if h2_ok:
        reasons.append(
            f"H2 PASS: {h2_cells_passing}/{H2_TOTAL_CELLS} cells satisfy "
            f"Ridge ≥ {H2_BASELINE_RATIO} × TLOB."
        )
    else:
        reasons.append(
            f"H2 FAIL: only {h2_cells_passing}/{H2_TOTAL_CELLS} cells satisfy "
            f"baseline ratio (≥{H2_PASS_THRESHOLD_CELLS} required)."
        )
    if h4_ok:
        reasons.append(
            f"H4 PASS: {h4_arms_clearing}/{H1_TOTAL_ARMS} arms at H10 clear "
            f"mean OptRet floor {H4_MEAN_FLOOR}."
        )
    else:
        reasons.append(
            f"H4 FAIL: only {h4_arms_clearing}/{H1_TOTAL_ARMS} arms at H10 clear "
            f"floor (≥{H4_PASS_THRESHOLD_ARMS} required)."
        )
    if h3_ok:
        reasons.append(
            f"H3 PASS: {h3_arms_clearing}/{H1_TOTAL_ARMS} arms at H10 cleared "
            f"backtest cost gate."
        )
    else:
        reasons.append(
            f"H3 informational: {h3_arms_clearing}/{H1_TOTAL_ARMS} arms cleared "
            f"H10 cost gate (not blocking)."
        )

    outcome = R16dDecisionGateOutcome(
        verdict=verdict,
        h1_arms_passing=h1_arms_passing,
        h1_arms_total=H1_TOTAL_ARMS,
        h1_ok=h1_ok,
        h2_cells_passing=h2_cells_passing,
        h2_cells_total=H2_TOTAL_CELLS,
        h2_ok=h2_ok,
        h3_arms_clearing_breakeven=h3_arms_clearing,
        h3_ok=h3_ok,
        h4_arms_clearing_floor=h4_arms_clearing,
        h4_ok=h4_ok,
        h5_ridge_arms_distinct=h5_distinct_count,
        h5_ok=h5_ok,
        h5_failed_arms=tuple(h5_failed_arms),
        arm_decays=tuple(arm_decays),
        baseline_cells=tuple(baseline_cells),
        reasons=tuple(reasons),
        exit_code=exit_code,
    )
    return cell_results, outcome


# =============================================================================
# Verdict rendering (human-readable + JSON)
# =============================================================================


def render_verdict(
    outcome: R16dDecisionGateOutcome,
    cell_results: Dict[Tuple[str, str, int, str], R16dCellResult],
) -> str:
    """Human-readable verdict + per-arm decay summary + per-cell H2 table."""
    lines = [
        "=" * 78,
        f"  R-16d Horizon-Axis Sweep Analysis — VERDICT: {outcome.verdict}",
        "=" * 78,
        "",
        f"Exit code: {outcome.exit_code}",
        "",
        "Gate outcomes:",
        f"  H1 PRIMARY (horizon decay):       "
        f"{'PASS' if outcome.h1_ok else 'FAIL'}  "
        f"({outcome.h1_arms_passing}/{outcome.h1_arms_total} arms decay)",
        f"  H2 BASELINE (Ridge ≥ 0.80 TLOB):  "
        f"{'PASS' if outcome.h2_ok else 'FAIL'}  "
        f"({outcome.h2_cells_passing}/{outcome.h2_cells_total} cells)",
        f"  H3 COST (median |pred| > 1.4 bps):"
        f"{'PASS' if outcome.h3_ok else 'INFO'}  "
        f"({outcome.h3_arms_clearing_breakeven}/{outcome.h1_arms_total} arms)",
        f"  H4 NEGATIVE CONTROL (mean > -0.5%):"
        f"{'PASS' if outcome.h4_ok else 'FAIL'}  "
        f"({outcome.h4_arms_clearing_floor}/{outcome.h1_arms_total} arms at H10)",
        f"  H5 ARCH (Ridge × horizon distinct):"
        f"{'PASS' if outcome.h5_ok else 'ABORT'}  "
        f"({outcome.h5_ridge_arms_distinct}/2 Ridge arms)",
        "",
        "Reasons:",
    ]
    for r in outcome.reasons:
        lines.append(f"  - {r}")
    lines += [
        "",
        "Per-arm horizon-decay (H1 PRIMARY):",
        f"  {'arm':<35} {'IC(H10)':>10} {'IC(H60)':>10} {'IC(H300)':>10} {'decay':>10}",
    ]
    for d in outcome.arm_decays:
        decay_str = "✓ monotonic" if d.monotonic_decay else "✗"
        lines.append(
            f"  {d.arm_label:<35} "
            f"{d.test_ic_H10:>+10.4f} {d.test_ic_H60:>+10.4f} "
            f"{d.test_ic_H300:>+10.4f} {decay_str:>10}"
        )
    lines += [
        "",
        "Per-cell baseline (H2 — Ridge / TLOB ratio):",
        f"  {'return_type':<18} {'horizon':>8} {'IC(Ridge)':>11} {'IC(TLOB)':>11} {'ratio':>8} {'pass':>6}",
    ]
    for b in outcome.baseline_cells:
        pass_str = "✓" if b.h2_pass else "✗"
        ratio_str = f"{b.ratio:.3f}" if np.isfinite(b.ratio) else "NaN"
        lines.append(
            f"  {b.return_type:<18} H{b.horizon_value:<7} "
            f"{b.test_ic_ridge:>+11.4f} {b.test_ic_tlob:>+11.4f} "
            f"{ratio_str:>8} {pass_str:>6}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def outcome_to_json_dict(
    outcome: R16dDecisionGateOutcome,
    cell_results: Dict[Tuple[str, str, int, str], R16dCellResult],
) -> Dict[str, Any]:
    """Serialize verdict + cell results for ``--json`` output."""
    return {
        "verdict": outcome.verdict,
        "exit_code": outcome.exit_code,
        "gates": {
            "h1_ok": outcome.h1_ok,
            "h2_ok": outcome.h2_ok,
            "h3_ok": outcome.h3_ok,
            "h4_ok": outcome.h4_ok,
            "h5_ok": outcome.h5_ok,
        },
        "observed": {
            "h1_arms_passing": outcome.h1_arms_passing,
            "h1_arms_total": outcome.h1_arms_total,
            "h2_cells_passing": outcome.h2_cells_passing,
            "h2_cells_total": outcome.h2_cells_total,
            "h3_arms_clearing_breakeven": outcome.h3_arms_clearing_breakeven,
            "h4_arms_clearing_floor": outcome.h4_arms_clearing_floor,
            "h5_ridge_arms_distinct": outcome.h5_ridge_arms_distinct,
        },
        "h5_failed_arms": list(outcome.h5_failed_arms),
        "reasons": list(outcome.reasons),
        "arm_decays": [
            {
                "arm_label": d.arm_label,
                "test_ic_H10": d.test_ic_H10,
                "test_ic_H60": d.test_ic_H60,
                "test_ic_H300": d.test_ic_H300,
                "decay_H10_H60": d.decay_H10_H60,
                "decay_H60_H300": d.decay_H60_H300,
                "monotonic_decay": d.monotonic_decay,
            }
            for d in outcome.arm_decays
        ],
        "baseline_cells": [
            {
                "return_type": b.return_type,
                "horizon_value": b.horizon_value,
                "test_ic_ridge": b.test_ic_ridge,
                "test_ic_tlob": b.test_ic_tlob,
                "ratio": b.ratio,
                "h2_pass": b.h2_pass,
            }
            for b in outcome.baseline_cells
        ],
        "cell_results": {
            f"{mt}__{rt}__H{h}__{tl}": {
                "test_ic": c.test_ic,
                "n_total_trades": c.n_total_trades,
                "mean_opt_ret": c.mean_opt_ret,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "n_nonfinite_replaced": c.n_nonfinite_replaced,
                "block_length_used": c.block_length_used,
                "drop_top5_mean": c.drop_top5_mean,
                "h5_horizon_distinct": c.h5_horizon_distinct,
                "insufficient_data": c.insufficient_data,
            }
            for (mt, rt, h, tl), c in cell_results.items()
        },
    }


__all__ = [
    "EXPECTED_GRID_POINTS",
    "EXPECTED_HORIZON_VALUES",
    "EXPECTED_RETURN_TYPES",
    "EXPECTED_MODEL_TYPES",
    "EXPECTED_CELLS",
    "EXPECTED_ARMS",
    "H1_PASS_THRESHOLD_ARMS",
    "H1_REFUTE_THRESHOLD_ARMS",
    "H1_TOTAL_ARMS",
    "H2_BASELINE_RATIO",
    "H2_PASS_THRESHOLD_CELLS",
    "H2_TOTAL_CELLS",
    "H3_BREAKEVEN_BPS",
    "H3_PASS_THRESHOLD_ARMS",
    "H4_PASS_THRESHOLD_ARMS",
    "R16dAnalysisError",
    "R16dIncompleteSweepError",
    "R16dH5HorizonAxisError",
    "R16dCellResult",
    "R16dArmDecayResult",
    "R16dBaselineCellResult",
    "R16dDecisionGateOutcome",
    "analyze_r16d_sweep",
    "render_verdict",
    "outcome_to_json_dict",
]
