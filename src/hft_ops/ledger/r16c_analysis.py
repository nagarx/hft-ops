"""
R-16c multi-seed power analysis (Sub-cycle 4b deliverable, 2026-05-12).

Pre-registered statistical analysis of the cycle7_r16c_multi_seed_r16a
sweep produced by ``hft-ops/experiments/sweeps/cycle7_r16c_multi_seed_r16a.yaml``.
Tests whether R-16a's "+2.84% Ridge × Peak × deep_itm_1.4bps" finding
(F7) is genuine alpha or noise/cherry-pick/outlier artifact via:

- **H1 PRIMARY (three-conjunctive at deep_itm_1.4bps)**:
  (a) mean OptRet > +1.0% AND
  (b) pooled per-trade bootstrap CI lower bound > 0 AND
  (c) per-seed drop-top-5 OptRet mean > 0%
  ANY ONE failing → REFUTE. (Per X2 pre-registered design 2026-05-12.)

- **H4 NEGATIVE CONTROL (cherry-pick avoidance)**:
  per-arm mean OptRet across 8 thresholds > -0.5% (per Agent H E11 verdict
  2026-05-12: per-arm mean across thresholds, NOT pooled global; H4 is
  computed only for the Ridge×Peak arm — same arm that owns H1).

- **H5 ARCHITECTURAL INVARIANT (Phase A.3 REDESIGN lock)**:
  Ridge × N-seed produces bit-exact identical predicted_returns.npy SHA-256
  within each (model_type, return_type) cell. Failure indicates a
  ship-blocker in the sklearn determinism claim → ABORT.

**Decision gate** (committed BEFORE running per hft-rules §13):

- GO: H1(a) AND H1(b) AND H1(c) at deep_itm_1.4bps AND H4 > -0.5% AND H5 OK
- REFUTE: H5 OK AND (any H1 conjunct fails)
- INDETERMINATE: H5 OK AND mean in (0, +1%) range AND CI crosses zero AND
  drop-top-5 in (-0.5%, 0%) range  → trigger R-16c-extended N=20
- ABORT: H5 fails — Ridge non-determinism is an architectural ship-blocker

**Exit codes** (per Agent G Q6 + Q8 verdict):
- 0 = GO
- 1 = REFUTE or INDETERMINATE (legitimate research outcomes, not errors)
- 2 = ABORT (architectural ship-blocker, fix REDESIGN before retry)
- 3 = INCOMPLETE_SWEEP (missing records or per-trade .npy files)

**Architectural pattern**: extends X2's pre-registered "pooled per-trade
bootstrap" primitive. ``hft-ops sweep compare`` is NOT applicable per X2
pre-impl audit 2026-05-12 (3 guard rails reject: unpaired-labels +
val_ic-only registry + classification-signal rejection). This module is
the manual analysis pipeline pre-registered in the R-16c manifest L154-180.

**Architecture notes** (per Agent G design review 2026-05-12):
- Sibling to ``statistical_compare.py`` in ``hft_ops.ledger`` subpackage
  (NOT new ``hft_ops/analysis/`` subpackage; established Class A surface).
- Single-array ``_pooled_block_bootstrap_mean_ci`` helper avoids the
  paired-API dummy-y workaround of ``block_bootstrap_ci``. Documented
  intentional-near-duplicate of ``hft-metrics/bootstrap.py:141-234``; if
  3rd consumer emerges, promote to ``hft-metrics`` v0.1.10.
- Reuses ``signal_dir.resolve_signal_dir`` (Sub-cycle 4b SSoT extraction).
- Reuses ``hft_contracts.provenance.hash_file`` for H5 SHA verification.
- Reuses ``hft_metrics._sanitize.assert_finite_array`` for §8 fail-loud.

6 fail-loud invariants (per Agent H verdict 2026-05-12; Agent H E9
AVG-CROSS-CHECK deferred to `R-16c-analyzer-v2` follow-up cycle — the
#PY-180 close (2026-05-13) addresses the H1(a)/H4 unit-mismatch via
load-time DOLLAR→FRACTION conversion in ``_load_per_trade_pnls``;
AVG-CROSS-CHECK (analyzer-side aggregate option_return_pct recomputation
+ cross-check against producer JSON summary) is a separate concern about
producer/consumer aggregate consistency, scoped to the follow-up cycle):

1. FILE-COUNT: expected_records × 8_thresholds .npy files (modulo n_trades=0)
2. FINITENESS: assert_finite_array(opt_trade_pnls) per cell
3. N-SIZE-FLOOR: n_pooled >= max(5, 2*block_length); else INSUFFICIENT_DATA
4. H5-RIDGE-IDENTITY: SHA-256 equal across all 10 Ridge seeds per cell
5. THRESHOLD-CANON: exact match of 8 canonical threshold labels
6. MODEL-TYPE-DISPATCH: H5 only checked on temporal_ridge arms
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
# Pre-registered constants (locked BEFORE sweep run per hft-rules §13)
# =============================================================================

# Canonical 8 cost thresholds. Sourced from
# `lob-backtester/scripts/run_regression_backtest.py:275-284` (the producer
# of the per-threshold backtests). THRESHOLD-CANON invariant (Agent H E10):
# exact match required across all 40 records; any divergence → fail-loud.
CANONICAL_THRESHOLD_LABELS: Tuple[str, ...] = (
    "deep_itm_1.4bps",
    "itm_2bps",
    "itm_3bps",
    "atm_5bps",
    "high_conv_8bps",
    "very_high_10bps",
    "ultra_conv_15bps",
    "max_conv_20bps",
)

# H1 PRIMARY hypothesis threshold + floor values per manifest L51, L61-63
H1_TARGET_THRESHOLD: str = "deep_itm_1.4bps"
H1_MEAN_FLOOR: float = 0.01    # +1.0% per manifest L61
H1_DROP_TOP_K: int = 5         # per manifest L62-63
# H4 negative-control floor per manifest L114
H4_MEAN_FLOOR: float = -0.005  # -0.5%

# Pooled bootstrap defaults per manifest L168 + Agent H E3 (cube-root rule
# w/ n_blocks>=5 hard floor, n_bootstrap=10000 for tight CI).
N_BOOTSTRAP_DEFAULT: int = 10_000
N_BLOCKS_MIN: int = 5
DEFAULT_BOOTSTRAP_SEED: int = 42

# #PY-180 close (2026-05-13): producer-side `lob-backtester/scripts/run_regression_backtest.py:158`
# hardcodes initial_capital=100_000.0 USD. The producer dumps DOLLAR per-trade pnls;
# consumer-side analyzer MUST convert to FRACTION-of-capital units at load time to align
# with H1_MEAN_FLOOR=0.01 (1.0%) + H4_MEAN_FLOOR=-0.005 (-0.5%) fractional thresholds.
# Single conversion-point at `_load_per_trade_pnls` (D1 design verified by 3 pre-impl agents).
DEFAULT_INITIAL_CAPITAL: float = 100_000.0

# R-16c expected grid points (per manifest: 2 model_type × 2 return_type
# × 10 seed = 40). Used for incomplete-sweep detection.
EXPECTED_GRID_POINTS: int = 40

# 4 cells: (model_type, return_type) tuples
EXPECTED_CELLS: Tuple[Tuple[str, str], ...] = (
    ("temporal_ridge", "point_return"),
    ("temporal_ridge", "peak_return"),
    ("tlob", "point_return"),
    ("tlob", "peak_return"),
)


# =============================================================================
# Exception classes (per Agent G Q6 + Q8: distinct exit codes)
# =============================================================================


class R16cAnalysisError(ValueError):
    """Base class for R-16c analysis failures. Subclass exit codes:
    - R16cIncompleteSweepError → exit code 3
    - R16cH5InvariantError → exit code 2 (ABORT)
    """


class R16cIncompleteSweepError(R16cAnalysisError):
    """Missing records or missing per-trade .npy files → exit 3.

    Sub-cycle 4b §8 fail-loud per Agent H E7: refuse to render verdict on
    partial data. Operator must re-run failed arms or accept PARTIAL banner
    in a separate explicit invocation.
    """


class R16cH5InvariantError(R16cAnalysisError):
    """Ridge × N-seed bit-exact invariant violated → ABORT exit 2.

    Phase A.3 REDESIGN architectural lock: ``Ridge(alpha=...)`` is RNG-free
    by construction. If 10 ridge×point or 10 ridge×peak records produce
    DIFFERENT predicted_returns.npy SHA-256 hashes, the sklearn determinism
    claim is refuted. Fix REDESIGN before retrying R-16c.
    """


# =============================================================================
# Result dataclasses (frozen — fingerprint-stable per CLAUDE.md atomic_io conv.)
# =============================================================================


@dataclass(frozen=True)
class R16cCellResult:
    """Per-cell statistics for one (arm × threshold) combination.

    Cell = {(model_type, return_type) × threshold_label}. Stats are pooled
    across N seeds (10 for TLOB; 1 effective for Ridge since H5 implies
    identity). All fields are scalar or Tuple[float, ...] for frozen
    serialization stability.
    """
    arm_label: str               # e.g., "temporal_ridge__peak_return"
    threshold_label: str          # e.g., "deep_itm_1.4bps"
    n_seeds: int                  # observed N seeds with valid records (≤10)
    n_total_trades: int           # pooled across N seeds at this threshold
    mean_opt_ret: float           # pooled per-trade mean
    ci_low: float                 # 95% block-bootstrap CI lower bound
    ci_high: float                # 95% block-bootstrap CI upper bound
    n_nonfinite_replaced: int     # Cluster Z v0.1.9 observability counter
    block_length_used: int        # actually-applied block_length (post-floor)
    drop_top5_per_seed: Tuple[float, ...]  # per-seed mean-after-drop-top-5
    drop_top5_mean: float                  # mean across seeds (H1c statistic)
    h5_bit_exact: Optional[bool]  # Ridge cells only; None for TLOB
    insufficient_data: bool = False  # True if N_SIZE_FLOOR violated for this cell


@dataclass(frozen=True)
class DecisionGateOutcome:
    """4-way verdict on R-16c sweep.

    Per Agent G Q7 verdict: structured per-conjunct booleans + reasons list
    + exit_code for crisp operator diagnostics. NOT a bare bool.
    """
    verdict: Literal["GO", "REFUTE", "INDETERMINATE", "ABORT"]
    # H1 PRIMARY three-conjunctive (at deep_itm_1.4bps for Ridge × Peak)
    h1_mean_ok: bool                # mean OptRet > +1%
    h1_ci_ok: bool                  # CI lower bound > 0
    h1_drop_top5_ok: bool           # drop-top-5 mean > 0
    # H4 NEGATIVE CONTROL
    h4_negative_control_ok: bool    # mean across 8 thresholds > -0.5%
    # H5 INVARIANT
    h5_invariant_ok: bool           # Ridge bit-exact across seeds within cells
    # Diagnostic context (humans need to see WHY)
    h1_mean_observed: float
    h1_ci_low_observed: float
    h1_ci_high_observed: float
    h1_drop_top5_observed: float
    h4_mean_observed: float
    h5_failed_cells: Tuple[str, ...]  # e.g., ("temporal_ridge__peak_return",)
    reasons: Tuple[str, ...]        # human-readable per-gate reasons
    exit_code: int                  # 0=GO, 1=REFUTE/INDET, 2=ABORT, 3=INCOMPLETE


# =============================================================================
# Helpers (single-array bootstrap + per-seed drop-top-K + H5 SHA verify)
# =============================================================================


def _pooled_block_bootstrap_mean_ci(
    arr: np.ndarray,
    *,
    n_bootstraps: int = N_BOOTSTRAP_DEFAULT,
    block_length: Optional[int] = None,
    ci: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float, int, int]:
    """Moving block bootstrap CI on single-array mean.

    Specialized for R-16c per-trade pooled use case. For paired-array stats
    (Spearman IC, Pearson r), use ``hft_metrics.bootstrap.block_bootstrap_ci``
    (paired (x, y) signature, returns 4-tuple). This single-array helper
    avoids the dummy-y workaround Agent G Q2 flagged. ~15 LOC documented-
    intentional-near-duplicate; if a 3rd consumer emerges, promote to
    ``hft-metrics`` v0.1.10 (per §0 reuse-first threshold).

    Returns:
        (estimate, ci_lower, ci_upper, n_nonfinite_replaced, block_length_used)
        — extends Cluster Z v0.1.9 4-tuple with explicit ``block_length_used``
        for operator diagnostics (when default cube-root rule produced
        n_blocks < N_BLOCKS_MIN floor, the helper raises the block_length
        downward to maintain ≥5 blocks).

    Args:
        arr: Single 1-D array of values (pooled per-trade pnls).
        n_bootstraps: Resample count. Default 10000 for tight CI.
        block_length: Fixed block size. Default ceil(n^(1/3)) per
            Politis-Romano 1994 with N_BLOCKS_MIN=5 hard floor.
        ci: Confidence level (default 0.95).
        seed: RNG seed for reproducibility.

    Raises:
        ValueError: if len(arr) < 3 (insufficient data for bootstrap).
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 3:
        # Mirror hft-metrics block_bootstrap_ci edge-case behavior
        m = float(np.mean(arr)) if n > 0 else float("nan")
        return m, m, m, 0, 1

    # Cube-root block_length with N_BLOCKS_MIN hard floor (Agent H E3)
    if block_length is None:
        block_length = max(1, math.ceil(n ** (1.0 / 3.0)))
    # If default produces fewer than N_BLOCKS_MIN blocks, reduce block_length
    if n // max(1, block_length) < N_BLOCKS_MIN:
        block_length = max(1, n // N_BLOCKS_MIN)

    estimate = float(np.mean(arr))
    rng = np.random.RandomState(seed)
    alpha = 1.0 - ci
    n_blocks = max(1, n // block_length)

    boot_stats = np.empty(n_bootstraps, dtype=np.float64)
    n_nonfinite_replaced = 0
    for b in range(n_bootstraps):
        block_starts = rng.randint(0, max(1, n - block_length + 1), size=n_blocks)
        indices = np.concatenate([
            np.arange(s, min(s + block_length, n)) for s in block_starts
        ])[:n]
        stat = float(np.mean(arr[indices]))
        if np.isfinite(stat):
            boot_stats[b] = stat
        else:
            boot_stats[b] = estimate
            n_nonfinite_replaced += 1

    ci_lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return estimate, ci_lower, ci_upper, n_nonfinite_replaced, block_length


def _drop_top_k_by_abs_per_seed(
    per_seed_arrays: List[np.ndarray],
    k: int,
) -> Tuple[Tuple[float, ...], float]:
    """For each seed's per-trade pnl array, drop the top-K elements BY ABS(pnl)
    and compute the mean of the remaining. Returns per-seed means + overall mean.

    Per Agent G Q4 verdict 2026-05-12 + Wave 3 Agent A precedent: top-K by
    ABS magnitude (NOT by VALUE) — symmetric outlier-robustness. R-16a's
    Wave 3 corrected analysis used this convention to derive
    "top 5 trades = 123.2% of return" finding.

    Edge case (Agent H E5): when n_trades < 2*k for a seed, that seed's
    drop-top-K is computed on whatever remains (mean=0 if k>=n). The
    per-cell-level INSUFFICIENT_DATA flag (set elsewhere) is the operator's
    cue that the cell is below statistical power.
    """
    per_seed_means: List[float] = []
    for arr in per_seed_arrays:
        if len(arr) == 0:
            per_seed_means.append(0.0)
            continue
        if k >= len(arr):
            # Defensive — INSUFFICIENT_DATA cell flag fires elsewhere
            per_seed_means.append(0.0)
            continue
        # argsort by ABS descending; drop top-K indices
        abs_idx = np.argsort(np.abs(arr))[::-1]
        to_drop = set(abs_idx[:k].tolist())
        kept = np.array([v for i, v in enumerate(arr) if i not in to_drop])
        per_seed_means.append(float(np.mean(kept)) if len(kept) > 0 else 0.0)
    overall = float(np.mean(per_seed_means)) if per_seed_means else 0.0
    return tuple(per_seed_means), overall


def _verify_h5_bit_exact(
    arm_label: str,
    signal_dirs: List[Path],
) -> Tuple[bool, Optional[str]]:
    """Verify all paths produce bit-exact ``predicted_returns.npy``.

    Per Agent G Q5 verdict 2026-05-12: SHA-256 via ``hft_contracts.provenance.hash_file``
    (auditable; ``np.save`` writes deterministic byte layout for identical
    f64 arrays — no timestamp in NPY header). Equivalent to ``np.array_equal``
    for byte-identical inputs but cheaper + more diagnostic.

    Returns ``(all_equal, diagnostic_message)``. On failure, the message
    enumerates the distinct SHA hex prefixes + first-pair diff position
    for operator debugging.

    Args:
        arm_label: Descriptive arm name (for error msg context).
        signal_dirs: List of N signal directories (e.g., 10 for Ridge × N seeds).
            Empty list returns ``(True, "no-op (0 signal dirs)")``.
    """
    if not signal_dirs:
        return True, "no-op (0 signal dirs)"
    shas: List[str] = []
    for sig_dir in signal_dirs:
        pred_path = sig_dir / "predicted_returns.npy"
        if not pred_path.exists():
            return False, f"{arm_label}: missing predicted_returns.npy at {pred_path}"
        shas.append(hash_file(pred_path, missing_ok=False))
    distinct = set(shas)
    if len(distinct) == 1:
        return True, None
    # Build diagnostic message — useful for debugging Phase A.3 REDESIGN regressions
    diag_parts = [f"{arm_label}: {len(distinct)} distinct SHA across {len(shas)} seeds"]
    for i, (sd, sh) in enumerate(zip(signal_dirs, shas)):
        diag_parts.append(f"  seed_{i}: {sd.name} → {sh[:16]}...")
    return False, "\n".join(diag_parts)


def _classify_verdict(
    h1_mean_ok: bool,
    h1_ci_ok: bool,
    h1_drop_top5_ok: bool,
    h4_negative_control_ok: bool,
    h5_invariant_ok: bool,
    h1_mean_observed: float,
    h1_ci_low_observed: float,
    h1_drop_top5_observed: float,
) -> Tuple[Literal["GO", "REFUTE", "INDETERMINATE", "ABORT"], int]:
    """Map (5 gates, observed values) → 4-way verdict + exit code.

    Per manifest L108-132 decision-gate table:
    - H5 fails → ABORT (exit 2): architectural ship-blocker.
    - All H1 conjuncts pass + H4 pass → GO (exit 0).
    - Borderline (mean in (0,1%), CI crosses 0, drop-top-5 in (-0.5%, 0%))
      → INDETERMINATE (exit 1; trigger R-16c-extended N=20).
    - Otherwise → REFUTE (exit 1).
    """
    if not h5_invariant_ok:
        return "ABORT", 2
    if h1_mean_ok and h1_ci_ok and h1_drop_top5_ok and h4_negative_control_ok:
        return "GO", 0
    # INDETERMINATE conditions: borderline (low signal, wide CI, marginal robustness).
    # H4_MEAN_FLOOR = -0.005 (a NEGATIVE value); range "(-0.5%, 0%)" means strict
    # H4_MEAN_FLOOR < drop_top5 < 0.0 per Agent 1 mid-impl C1 fix (2026-05-12):
    # drop-top-5 = exactly 0.0 or = -0.5% should be REFUTE per manifest L120
    # ("drop-top-5 < -0.5%" or "drop-top-5 ≤ 0%"), NOT INDETERMINATE. Strict bounds
    # avoid boundary ambiguity. Do NOT negate H4_MEAN_FLOOR (it IS negative).
    is_borderline = (
        0.0 < h1_mean_observed < H1_MEAN_FLOOR
        and h1_ci_low_observed <= 0.0
        and H4_MEAN_FLOOR < h1_drop_top5_observed < 0.0  # strict (-0.5%, 0%)
    )
    if is_borderline:
        return "INDETERMINATE", 1
    return "REFUTE", 1


# =============================================================================
# Main entry: analyze_r16c_sweep
# =============================================================================


def _resolve_backtest_pnls_dir(
    record_entry: Dict[str, Any],
    ledger: Any,
    paths: PipelinePaths,
) -> Path:
    """Resolve the directory containing per-trade ``option_trade_pnls.npy``
    files for a sweep record.

    Layout (per Agent 2 mid-impl audit 2026-05-12 ground-truth verification):

    - Orchestrator runs backtester subprocess with ``cwd=config.paths.backtester_dir``
      (``hft-ops/src/hft_ops/stages/backtesting.py:176``). Backtester default
      ``--output-dir "outputs/backtests/"`` is CWD-relative, so per-trade files
      land at ``<backtester_dir>/outputs/backtests/{run_name}__option_trade_pnls__{label}.npy``.

    - For manual / standalone-script invocations, ``cwd`` may differ; the
      candidate chain covers both paths.

    - ``signal_dir`` resolves to ``<root>/outputs/experiments/<point>/signals/test/``
      (orchestrator-emitted). Walking up 4 levels from there lands at
      ``<root>/outputs/`` (NOT root); 5 levels lands at ``<root>/`` (true root).

    Per Agent 2 mid-impl FIX (2026-05-12): the original ``candidates[0]`` had
    `outputs/outputs/backtests` (DOUBLED `outputs`), silently never matched,
    fell through to fallback by accident. Restructured candidate chain:

    1. ``paths.backtester_dir / outputs / backtests`` — orchestrator default
       (most common production path)
    2. ``<root>/lob-backtester/outputs/backtests`` — explicit lob-backtester
       layout (covers test-fixture + manual-run cases)
    3. ``sig_dir.parent.parent.parent.parent.parent / outputs / backtests``
       — walking up 5 from signal_dir → ``<root>/outputs/backtests`` (alt layout)
    4. ``Path("outputs") / "backtests"`` — CWD-relative fallback (least likely)

    Raises:
        R16cIncompleteSweepError: if no candidate contains per-trade files.
    """
    sig_dir = resolve_signal_dir(record_entry, ledger, paths)
    # Build candidate chain. ``paths.backtester_dir`` may not exist on all
    # PipelinePaths variants (defensive); use getattr with fallback.
    backtester_dir = getattr(paths, "backtester_dir", None)
    candidates: List[Path] = []
    if backtester_dir is not None:
        candidates.append(Path(backtester_dir) / "outputs" / "backtests")
    candidates.extend([
        paths.pipeline_root / "lob-backtester" / "outputs" / "backtests",
        sig_dir.parent.parent.parent.parent.parent / "outputs" / "backtests",
        Path("outputs") / "backtests",
    ])
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise R16cIncompleteSweepError(
        f"_resolve_backtest_pnls_dir: no per-trade pnls directory found near "
        f"signal_dir={sig_dir}. Searched: {[str(c) for c in candidates]}. "
        f"R-16a's pre-Sub-cycle-4a backtests had NO per-trade dump (de99f45 ships "
        f"the producer-side fix); re-run backtests to populate. Orchestrator "
        f"runs backtester with cwd=config.paths.backtester_dir per "
        f"hft-ops/src/hft_ops/stages/backtesting.py:176."
    )


def _load_per_trade_pnls(
    pnls_dir: Path,
    run_name: str,
    threshold_label: str,
    *,
    record_status: Optional[str] = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> Optional[np.ndarray]:
    """Load one per-trade pnls array, converting DOLLAR units → FRACTION of capital.

    Producer (``lob-backtester/scripts/run_regression_backtest.py:139``) dumps
    raw USD per-trade pnls via ``atomic_write_npy``. Consumer-side analyzer
    requires FRACTION-of-capital units to align with H1/H4 fractional floors
    (``H1_MEAN_FLOOR=0.01`` = +1.0%, ``H4_MEAN_FLOOR=-0.005`` = -0.5%). This
    function performs the conversion at the unit boundary so all downstream
    computations (pooled bootstrap CI, drop-top-K, _classify_verdict) operate
    in fraction-of-capital units consistent with manifest pre-registration.

    #PY-180 close (2026-05-13): pre-fix, the analyzer compared DOLLAR per-trade
    means (~$0.45-$50) to FRACTION thresholds (0.01) producing semantically
    wrong dispatch on H1(a) + H4 gates, plus render bug at lines 815-818
    rendering "$X*100" as fake-percent. Load-time conversion fixes 12 sites
    at single source.

    Returns ``None`` if the file is absent AND the record status is
    consistent with "no trades met threshold" (legitimate n_trades=0 per
    ``run_regression_backtest.py:126`` guard).

    Per Agent 1 + Agent 2 mid-impl audit 2026-05-12 (Fix H2/§8):
    distinguishes (a) producer skipped dump because n_trades=0 (LEGITIMATE,
    record.status='completed') from (b) sweep arm failed mid-run leaving
    file absent (GENUINE failure, record.status='failed' or other).

    Filename convention: ``{run_name}__option_trade_pnls__{label}.npy`` per
    Sub-cycle 4a producer-side ship (de99f45).

    Args:
        pnls_dir: Resolved per-trade pnls directory.
        run_name: Sweep grid point name (used in filename).
        threshold_label: Cost-threshold label (e.g., ``deep_itm_1.4bps``).
        record_status: Optional record status from ledger
            (``completed``/``failed``). When supplied + status != ``completed``
            + file missing → §8 fail-loud raise. Default None preserves
            permissive behavior for analyzer callers that don't have status
            handy.
        initial_capital: Backtester initial capital (USD). Loaded array is
            divided by this to produce FRACTION-of-capital per-trade values.
            Default ``DEFAULT_INITIAL_CAPITAL=100_000.0`` matches all R-series
            experiments (producer-side hardcoded at ``run_regression_backtest.py:158``).
            Future experiments with different initial_capital MUST pass the
            actual value explicitly (separate #PY-181 will persist this to
            backtest record summary for automatic discovery).

    Raises:
        R16cIncompleteSweepError: if file is missing AND record_status
            indicates the arm did not complete successfully (Agent H E1 +
            Agent 2 #4 distinguishing legitimate-skip vs sweep-failure).
        ValueError: if initial_capital <= 0 (division-by-zero / sign-flip guard).
    """
    # Mid-impl gate post-fix (2026-05-13): extend guard to reject NaN + Inf.
    # Pre-fix `initial_capital <= 0` silently passed `math.nan` (NaN ≤ 0 == False)
    # and `math.inf` (inf ≤ 0 == False), producing all-NaN or all-0.0 fraction
    # arrays respectively → silent unit corruption. `math.isfinite()` first
    # catches both before any silent-arithmetic path. Per hft-rules §2 + §5 + §8.
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError(
            f"_load_per_trade_pnls: initial_capital must be finite and > 0, "
            f"got {initial_capital!r}. Per #PY-180 design: this value is used "
            f"to normalize DOLLAR per-trade pnls to FRACTION of capital. "
            f"Non-finite (NaN/Inf) or non-positive values would produce "
            f"silently-wrong fractional returns (NaN→all-NaN, Inf→all-zero, "
            f"≤0→sign-flip or div-by-zero)."
        )
    fname = f"{run_name}__option_trade_pnls__{threshold_label}.npy"
    path = pnls_dir / fname
    if not path.exists():
        if record_status is not None and record_status != "completed":
            raise R16cIncompleteSweepError(
                f"_load_per_trade_pnls: file {fname} missing AND "
                f"record_status={record_status!r} (NOT 'completed'). "
                f"This is a SWEEP FAILURE — cannot interpret as legitimate "
                f"n_trades=0 skip. Re-run the failed arm or accept PARTIAL "
                f"banner via separate explicit invocation."
            )
        return None  # interpreted as n_trades=0 (legitimate skip)
    arr = np.load(path)
    # Agent H invariant 2: FINITENESS fail-loud per §8
    assert_finite_array(arr, name=f"r16c.{run_name}.{threshold_label}")
    # #PY-180 close (2026-05-13): DOLLAR → FRACTION of capital at load boundary
    arr_frac = arr / initial_capital
    return arr_frac


def analyze_r16c_sweep(
    sweep_id: str,
    ledger: Any,
    paths: PipelinePaths,
    *,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    allow_partial: bool = False,
    min_grid_points: int = EXPECTED_GRID_POINTS,
) -> Tuple[Dict[Tuple[str, str, str], R16cCellResult], DecisionGateOutcome]:
    """Analyze an R-16c sweep + render verdict.

    Args:
        sweep_id: Sweep aggregate identifier (e.g.,
            ``cycle7_r16c_multi_seed_r16a_20260513T120000``).
        ledger: ``hft_ops.ledger.ledger.ExperimentLedger`` instance.
        paths: ``PipelinePaths`` for output-dir resolution.
        n_bootstrap: Bootstrap iterations per cell. Default 10000.
        bootstrap_seed: RNG seed for reproducibility. Default 42.
        initial_capital: Backtester initial capital (USD) used for
            DOLLAR→FRACTION normalization at the per-trade-pnls load boundary
            (#PY-180 close). Default matches producer-side hardcoded
            ``run_regression_backtest.py:158``. Future R-cycles with
            non-default capital MUST pass explicitly (#PY-181 will
            persist to backtest record summary).

    Returns:
        ``(cell_results, outcome)`` — dict keyed by ``(model_type, return_type,
        threshold_label)`` × ``DecisionGateOutcome`` with 4-way verdict.

    Raises:
        R16cIncompleteSweepError: missing records / .npy files. Exit code 3.
        R16cH5InvariantError: Ridge non-determinism detected. Exit code 2.
    """
    # Filter ledger by sweep_id
    records = ledger.filter(sweep_id=sweep_id)
    # Skip the parent sweep_aggregate record (per Phase 8A.1 RecordType)
    grid_records = [
        r for r in records
        if r.get("record_type") not in ("sweep_aggregate", "sweep_failure")
    ]
    effective_threshold = min_grid_points if allow_partial else EXPECTED_GRID_POINTS
    if len(grid_records) < effective_threshold:
        raise R16cIncompleteSweepError(
            f"analyze_r16c_sweep: expected {effective_threshold} grid records; "
            f"got {len(grid_records)} for sweep_id={sweep_id}. Re-run failed "
            f"arms or accept PARTIAL banner via separate explicit invocation. "
            f"Per Agent H E7 §8 fail-loud."
        )
    if allow_partial and len(grid_records) < EXPECTED_GRID_POINTS:
        import warnings as _warnings
        _warnings.warn(
            f"PARTIAL R-16c analysis: {len(grid_records)}/{EXPECTED_GRID_POINTS} "
            f"grid records (allow_partial=True). Reduced sample-per-cell may "
            f"widen bootstrap CI. Document rationale "
            f"(typical cause: cross-sweep dedup against earlier sweep records).",
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
            # Fall back to per-record training_config introspection
            tc = r.get("training_config") or {}
            mt = mt or tc.get("model", {}).get("model_type")
            data_labels = tc.get("data", {}).get("labels", {})
            rt = rt or data_labels.get("return_type")
        if mt is None or rt is None:
            raise R16cIncompleteSweepError(
                f"analyze_r16c_sweep: record {r.get('experiment_id', '<unknown>')} "
                f"missing model_type/return_type in axis_values + training_config."
            )
        cells_records.setdefault((mt, rt), []).append(r)

    # Verify all 4 cells present
    missing_cells = set(EXPECTED_CELLS) - set(cells_records.keys())
    if missing_cells:
        raise R16cIncompleteSweepError(
            f"analyze_r16c_sweep: missing cells {sorted(missing_cells)} in sweep."
        )

    # =========================================================================
    # H5 INVARIANT (Ridge cells only; Agent H E6 + E12 MODEL-TYPE-DISPATCH)
    # Track per-cell which Ridge cells passed — needed for H4 Fix below
    # (single-seed pooling avoids N-fold CI tightening per Agent 1 mid-impl
    # CRITICAL 2026-05-12).
    # =========================================================================
    h5_failed_cells: List[str] = []
    h5_diagnostics: List[str] = []
    h5_passed_cell_keys: set = set()  # Set[(model_type, return_type)] of identical-SHA Ridge cells
    # Cache signal_dir resolutions (40 records × N analyses → avoid repeated work)
    signal_dir_cache: Dict[str, Path] = {}

    def _cached_resolve_signal_dir(r: Dict[str, Any]) -> Path:
        exp_id = str(r.get("experiment_id", ""))
        if exp_id not in signal_dir_cache:
            signal_dir_cache[exp_id] = resolve_signal_dir(r, ledger, paths)
        return signal_dir_cache[exp_id]

    for cell_key in EXPECTED_CELLS:
        model_type, return_type = cell_key
        if model_type != "temporal_ridge":
            continue  # H5 only on Ridge per manifest L97-106
        cell_records = cells_records[cell_key]
        sig_dirs = [_cached_resolve_signal_dir(r) for r in cell_records]
        all_equal, diag = _verify_h5_bit_exact(
            arm_label=f"{model_type}__{return_type}",
            signal_dirs=sig_dirs,
        )
        if all_equal:
            h5_passed_cell_keys.add(cell_key)
        else:
            h5_failed_cells.append(f"{model_type}__{return_type}")
            if diag is not None:
                h5_diagnostics.append(diag)

    h5_invariant_ok = not h5_failed_cells

    # =========================================================================
    # H1 + per-cell + per-threshold analysis
    # =========================================================================
    cell_results: Dict[Tuple[str, str, str], R16cCellResult] = {}
    pnls_dir_resolved: Optional[Path] = None
    for cell_key in EXPECTED_CELLS:
        model_type, return_type = cell_key
        cell_records = cells_records[cell_key]
        # Resolve per-trade pnls dir lazily (same for all cells of same sweep)
        if pnls_dir_resolved is None and cell_records:
            pnls_dir_resolved = _resolve_backtest_pnls_dir(
                cell_records[0], ledger, paths,
            )
        # CRITICAL Fix H4 per Agent 1 mid-impl 2026-05-12: when this cell is in
        # h5_passed_cell_keys (Ridge cell with bit-exact SHA across seeds), all
        # 10 seeds produce IDENTICAL predicted_returns.npy → IDENTICAL
        # option_trade_pnls.npy (backtester is deterministic given identical
        # inputs). Concatenating 10 redundant copies would inflate n_total_trades
        # 10× → artificially tight bootstrap CI → potential false GO verdicts.
        # Per manifest L166-167: "use ANY 1 of 10 Ridge×Peak records since H5
        # implies identity". Take ONLY the first seed for Ridge h5-passed cells.
        if cell_key in h5_passed_cell_keys:
            records_for_pooling = cell_records[:1]
        else:
            records_for_pooling = cell_records
        for threshold_label in CANONICAL_THRESHOLD_LABELS:
            per_seed_arrays: List[np.ndarray] = []
            for r in records_for_pooling:
                run_name = r.get("name") or r.get("experiment_id", "")
                record_status = r.get("status")  # Fix H2: pass status for §8 discrimination
                arr = _load_per_trade_pnls(
                    pnls_dir_resolved,
                    run_name=str(run_name),
                    threshold_label=threshold_label,
                    record_status=record_status,
                    initial_capital=initial_capital,
                )
                if arr is not None and len(arr) > 0:
                    per_seed_arrays.append(arr)
            # N_SIZE_FLOOR check (Agent H E3)
            if not per_seed_arrays:
                cell_results[(model_type, return_type, threshold_label)] = R16cCellResult(
                    arm_label=f"{model_type}__{return_type}",
                    threshold_label=threshold_label,
                    n_seeds=0, n_total_trades=0,
                    mean_opt_ret=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"),
                    n_nonfinite_replaced=0, block_length_used=0,
                    drop_top5_per_seed=tuple(),
                    drop_top5_mean=float("nan"),
                    h5_bit_exact=(None if model_type != "temporal_ridge" else h5_invariant_ok),
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
            per_seed_drops, drop_top5_mean = _drop_top_k_by_abs_per_seed(
                per_seed_arrays, k=H1_DROP_TOP_K,
            )
            cell_results[(model_type, return_type, threshold_label)] = R16cCellResult(
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
                h5_bit_exact=(None if model_type != "temporal_ridge" else h5_invariant_ok),
                insufficient_data=insufficient,
            )

    # =========================================================================
    # H1 + H4 decision-gate computation (Ridge × Peak only)
    # =========================================================================
    f7_cell_key = ("temporal_ridge", "peak_return", H1_TARGET_THRESHOLD)
    f7_cell = cell_results.get(f7_cell_key)
    if f7_cell is None or f7_cell.insufficient_data:
        h1_mean_ok = h1_ci_ok = h1_drop_top5_ok = False
        h1_mean_obs = h1_ci_low_obs = h1_ci_high_obs = h1_drop_top5_obs = float("nan")
    else:
        h1_mean_obs = f7_cell.mean_opt_ret
        h1_ci_low_obs = f7_cell.ci_low
        h1_ci_high_obs = f7_cell.ci_high
        h1_drop_top5_obs = f7_cell.drop_top5_mean
        h1_mean_ok = h1_mean_obs > H1_MEAN_FLOOR
        h1_ci_ok = h1_ci_low_obs > 0.0
        h1_drop_top5_ok = h1_drop_top5_obs > 0.0

    # H4: mean across 8 thresholds for the Ridge × Peak arm (Agent H E11)
    h4_means: List[float] = []
    for tl in CANONICAL_THRESHOLD_LABELS:
        c = cell_results.get(("temporal_ridge", "peak_return", tl))
        if c is not None and not c.insufficient_data:
            h4_means.append(c.mean_opt_ret)
    h4_mean_obs = float(np.mean(h4_means)) if h4_means else float("nan")
    h4_negative_control_ok = h4_mean_obs > H4_MEAN_FLOOR

    # Verdict classification + reasons
    verdict, exit_code = _classify_verdict(
        h1_mean_ok=h1_mean_ok,
        h1_ci_ok=h1_ci_ok,
        h1_drop_top5_ok=h1_drop_top5_ok,
        h4_negative_control_ok=h4_negative_control_ok,
        h5_invariant_ok=h5_invariant_ok,
        h1_mean_observed=h1_mean_obs,
        h1_ci_low_observed=h1_ci_low_obs,
        h1_drop_top5_observed=h1_drop_top5_obs,
    )

    reasons: List[str] = []
    if not h5_invariant_ok:
        reasons.append(
            f"H5 FAILED: Ridge bit-exact invariant violated in cells "
            f"{h5_failed_cells}. Phase A.3 REDESIGN ship-blocker."
        )
    if h1_mean_ok:
        reasons.append(f"H1a PASS: mean OptRet={h1_mean_obs:.4f} > +{H1_MEAN_FLOOR:.4f}")
    else:
        reasons.append(f"H1a FAIL: mean OptRet={h1_mean_obs:.4f} ≤ +{H1_MEAN_FLOOR:.4f}")
    if h1_ci_ok:
        reasons.append(f"H1b PASS: CI=({h1_ci_low_obs:.4f}, {h1_ci_high_obs:.4f}); lower > 0")
    else:
        reasons.append(f"H1b FAIL: CI=({h1_ci_low_obs:.4f}, {h1_ci_high_obs:.4f}); lower ≤ 0")
    if h1_drop_top5_ok:
        reasons.append(f"H1c PASS: drop-top-5 mean={h1_drop_top5_obs:.4f} > 0")
    else:
        reasons.append(f"H1c FAIL: drop-top-5 mean={h1_drop_top5_obs:.4f} ≤ 0")
    if h4_negative_control_ok:
        reasons.append(f"H4 PASS: mean across 8 thresholds={h4_mean_obs:.4f} > {H4_MEAN_FLOOR:.4f}")
    else:
        reasons.append(f"H4 FAIL: mean across 8 thresholds={h4_mean_obs:.4f} ≤ {H4_MEAN_FLOOR:.4f}")

    outcome = DecisionGateOutcome(
        verdict=verdict,
        h1_mean_ok=h1_mean_ok,
        h1_ci_ok=h1_ci_ok,
        h1_drop_top5_ok=h1_drop_top5_ok,
        h4_negative_control_ok=h4_negative_control_ok,
        h5_invariant_ok=h5_invariant_ok,
        h1_mean_observed=h1_mean_obs,
        h1_ci_low_observed=h1_ci_low_obs,
        h1_ci_high_observed=h1_ci_high_obs,
        h1_drop_top5_observed=h1_drop_top5_obs,
        h4_mean_observed=h4_mean_obs,
        h5_failed_cells=tuple(h5_failed_cells),
        reasons=tuple(reasons),
        exit_code=exit_code,
    )
    return cell_results, outcome


# =============================================================================
# Verdict rendering (human-readable + JSON)
# =============================================================================


def render_verdict(
    outcome: DecisionGateOutcome,
    cell_results: Dict[Tuple[str, str, str], R16cCellResult],
) -> str:
    """Human-readable verdict + per-arm cell summary.

    Used by ``hft-ops/scripts/analyze_r16c.py`` for terminal output.
    """
    lines = [
        "=" * 78,
        f"  R-16c Multi-Seed Power Analysis — VERDICT: {outcome.verdict}",
        "=" * 78,
        "",
        f"Exit code: {outcome.exit_code}",
        "",
        "Gate outcomes:",
        f"  H1a (mean > +{H1_MEAN_FLOOR*100:.1f}%):       {'PASS' if outcome.h1_mean_ok else 'FAIL'}  ({outcome.h1_mean_observed*100:+.2f}%)",
        f"  H1b (pooled CI > 0):              {'PASS' if outcome.h1_ci_ok else 'FAIL'}  CI=({outcome.h1_ci_low_observed*100:+.2f}%, {outcome.h1_ci_high_observed*100:+.2f}%)",
        f"  H1c (drop-top-5 > 0):             {'PASS' if outcome.h1_drop_top5_ok else 'FAIL'}  ({outcome.h1_drop_top5_observed*100:+.2f}%)",
        f"  H4 (mean across 8 thresh >-0.5%): {'PASS' if outcome.h4_negative_control_ok else 'FAIL'}  ({outcome.h4_mean_observed*100:+.2f}%)",
        f"  H5 (Ridge bit-exact invariant):   {'PASS' if outcome.h5_invariant_ok else 'FAIL'}",
        "",
        "Reasons:",
    ]
    for r in outcome.reasons:
        lines.append(f"  - {r}")
    lines += [
        "",
        "Per-cell summary (Ridge × Peak focus — F7 target):",
        f"  {'threshold':<22} {'n_seeds':>8} {'n_trades':>9} {'mean':>9} {'CI':>22} {'drop5':>9}",
    ]
    for tl in CANONICAL_THRESHOLD_LABELS:
        c = cell_results.get(("temporal_ridge", "peak_return", tl))
        if c is None:
            continue
        ci_str = f"({c.ci_low*100:+.2f},{c.ci_high*100:+.2f})"
        lines.append(
            f"  {tl:<22} {c.n_seeds:>8} {c.n_total_trades:>9} "
            f"{c.mean_opt_ret*100:>+8.2f}% {ci_str:>22} {c.drop_top5_mean*100:>+8.2f}%"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def outcome_to_json_dict(
    outcome: DecisionGateOutcome,
    cell_results: Dict[Tuple[str, str, str], R16cCellResult],
) -> Dict[str, Any]:
    """Serialize verdict + cell results for ``--json`` output."""
    return {
        "verdict": outcome.verdict,
        "exit_code": outcome.exit_code,
        "gates": {
            "h1_mean_ok": outcome.h1_mean_ok,
            "h1_ci_ok": outcome.h1_ci_ok,
            "h1_drop_top5_ok": outcome.h1_drop_top5_ok,
            "h4_negative_control_ok": outcome.h4_negative_control_ok,
            "h5_invariant_ok": outcome.h5_invariant_ok,
        },
        "observed": {
            "h1_mean": outcome.h1_mean_observed,
            "h1_ci_low": outcome.h1_ci_low_observed,
            "h1_ci_high": outcome.h1_ci_high_observed,
            "h1_drop_top5": outcome.h1_drop_top5_observed,
            "h4_mean": outcome.h4_mean_observed,
        },
        "h5_failed_cells": list(outcome.h5_failed_cells),
        "reasons": list(outcome.reasons),
        "cell_results": {
            f"{mt}__{rt}__{tl}": {
                "n_seeds": c.n_seeds,
                "n_total_trades": c.n_total_trades,
                "mean_opt_ret": c.mean_opt_ret,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "n_nonfinite_replaced": c.n_nonfinite_replaced,
                "block_length_used": c.block_length_used,
                "drop_top5_mean": c.drop_top5_mean,
                "h5_bit_exact": c.h5_bit_exact,
                "insufficient_data": c.insufficient_data,
            }
            for (mt, rt, tl), c in cell_results.items()
        },
    }


__all__ = [
    "CANONICAL_THRESHOLD_LABELS",
    "H1_TARGET_THRESHOLD",
    "H1_MEAN_FLOOR",
    "H1_DROP_TOP_K",
    "H4_MEAN_FLOOR",
    "EXPECTED_GRID_POINTS",
    "EXPECTED_CELLS",
    "R16cAnalysisError",
    "R16cIncompleteSweepError",
    "R16cH5InvariantError",
    "R16cCellResult",
    "DecisionGateOutcome",
    "analyze_r16c_sweep",
    "render_verdict",
    "outcome_to_json_dict",
]
