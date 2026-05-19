"""Analyze #PY-243 R-19 multi-seed validation cycle (cycle10_r19_multi_seed).

Pre-registered decision-gate analysis per pre-impl Agent B LOCKED 2026-05-16 LATE NIGHT
(post Wave 2 REFUTE OVERTURNED Phase 2 TB; #PY-243 R-19 multi-seed authorized).

PURPOSE: Validate R-19's single-seed +4.9pp PT precision lift (26.9% vs R-17a 22.0%
on TB v3p0 corpus) survives perturbation across N=5 seeds. Per Lesson #12 mandatory
(N≥3 with bootstrap CI) + R-16e Lesson #93 precedent (single-seed R-16d Ratio=2.585
collapsed under N=10 to mean=1.653). If R-19 lift was within seed variance, the
entire Phase 2 TB decision matrix rests on noise → CRITICAL to validate.

PRE-REGISTERED HYPOTHESES (LOCKED before script runs):

  H1 PRIMARY (lift survives perturbation):
    H1.a Point-estimate persistence: |mean(seeds) - 0.269| ≤ 0.020 (within ±2pp)
    H1.b R-17a CI separation BINDING: bootstrap CI lower > 0.220 at α=0.05
    H1.c Cost-floor separation INFORMATIONAL: bootstrap CI lower > 0.2681

  H2 BASELINE (architectural lift stability):
    H2.a mean > 0.220 (single-side over R-17a)
    H2.b std < 0.020 (seed-stability)

  H3 ARCHITECTURAL INVARIANT (Phase A.3 REDESIGN — TLOB is NOT RNG-free):
    H3.a Phase Y composer: N seeds → N DISTINCT experiment_provenance_hash
    H3.b Corpus identity: N seeds → 1 IDENTICAL compatibility_fingerprint
         = "dd21d079228096917c6db63227bc71d2f14534dbebb5a4a939eef19732791eaf"
    H3.c Model arch identity: N seeds → 1 IDENTICAL model_config_hash
         = "2dc7eeef5192db921ed348364fb4c76fbc5e3e917a69929791e016a99ee16a0e"
    H3.d Predicted-array divergence: ≥2 seeds → DIFFERENT predicted_returns.npy SHA-256

  M5 INTERIM-ANALYSIS GATE: σ(seeds) > 0.040 → verdict
    INDETERMINATE-DOWNGRADED-NEED-N10 (variance assumption falsified)

DECISION MATRIX (LOCKED PRE-RUN per Agent B):
  - GO: H1.a + H1.b + H2.a + H2.b + H3.a-d all PASS
  - GO-LIFTED-AND-RAISED: mean > 0.289 + H1.b + H2.a/b + H3 PASS
  - REFUTE: H1.b FAIL (CI lower ≤ 0.220) OR mean ≤ 0.220
  - INDETERMINATE: H1.a borderline + H1.b PASS
  - INDETERMINATE-DOWNGRADED-NEED-N10: σ > 0.040
  - ABORT: any H3 FAIL (architectural invariant violated)

Pre-registered pivot for REFUTE branch:
  (a) R-20 HMHP cascade (different architecture class)
  (b) R-18 cost-aware barrier sweep (with theta scaling + slippage realism)

Inputs:
- 5 signal dirs at lob-model-trainer/outputs/experiments/cycle10_r19_multi_seed__seed_NN/signals/test/
- 5 ledger records at hft-ops/ledger/runs/cycle10_r19_multi_seed_<ts>/cycle10_..._seed_NN.json
  (extracted via --sweep-id or --sweep-dir CLI flag)

Output:
- Human-readable verdict to stdout
- JSON verdict at hft-ops/ledger/r19_multi_seed_verdicts/cycle10_r19_multi_seed_<ts>.json

Cross-ref: pre-impl Agent A (readiness) + Agent B (gates) + Agent C (REFUTE) gate
2026-05-16 LATE NIGHT; PHASE_P_BACKLOG.md #PY-243; analyze_r17a_vs_r19_pt_precision.py
pattern reuse per hft-rules §0; hft_metrics.block_bootstrap_ci v0.1.11 SSoT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest


# ---------------------------------------------------------------------------
# Pre-registered constants (LOCKED before script runs per Agent B 2026-05-16)
# ---------------------------------------------------------------------------
ALPHA: float = 0.05
N_BOOTSTRAP: int = 10000
PT_CLASS_INDEX: int = 2  # TB encoding: 0=SL, 1=Timeout, 2=ProfitTarget

# R-19 baseline anchors (per EXPERIMENT_INDEX.md R-19 entry + BACKTEST_INDEX Round 19)
R19_PT_PRECISION_ANCHOR: float = 0.269
R19_COMPAT_FINGERPRINT: str = (
    "dd21d079228096917c6db63227bc71d2f14534dbebb5a4a939eef19732791eaf"
)
R19_MODEL_CONFIG_HASH: str = (
    "2dc7eeef5192db921ed348364fb4c76fbc5e3e917a69929791e016a99ee16a0e"
)

# R-17a baseline (per EXPERIMENT_INDEX.md R-17a entry)
R17A_PT_PRECISION_BASELINE: float = 0.220

# Pre-registered decision-gate thresholds
H1A_POINT_ESTIMATE_TOLERANCE: float = 0.020  # ±2pp of R-19 anchor
H1B_R17A_SEPARATION_THRESHOLD: float = R17A_PT_PRECISION_BASELINE
H1C_COST_FLOOR_THRESHOLD: float = 0.2681  # R-17a 0.220 + cost-economics floor 0.05 ÷ ~1.0
H2A_MEAN_FLOOR: float = R17A_PT_PRECISION_BASELINE
H2B_SIGMA_CEILING: float = 0.020
M5_VARIANCE_ABORT_SIGMA: float = 0.040
GO_LIFTED_THRESHOLD: float = 0.289  # R-19 anchor + ~+2pp


@dataclass(frozen=True)
class SeedResult:
    """Per-seed PT precision result + Wilson 95% CI + invariant fingerprints."""
    seed: int
    pt_predictions_count: int
    pt_correct_count: int
    pt_precision: float
    wilson_ci_low: float
    wilson_ci_high: float
    predicted_returns_sha256: str
    experiment_provenance_hash: Optional[str]
    compatibility_fingerprint: Optional[str]
    model_config_hash: Optional[str]


@dataclass(frozen=True)
class GateResult:
    """Pre-registered gate result with explicit threshold comparison."""
    label: str  # "H1.a" / "H1.b" / "H2.a" / etc.
    passed: bool
    observed: float
    threshold: float
    description: str


@dataclass(frozen=True)
class ArchitecturalInvariantResult:
    """H3.a-d architectural invariant check results."""
    h3a_distinct_eph: bool
    h3a_all_populated: bool                # Mid-impl H3-split: distinguishes None vs duplicate
    h3a_observed_eph_count: int            # count of populated (non-None) values
    h3a_observed_distinct_count: int       # count of unique values among populated
    h3a_expected_eph_count: int
    h3b_compat_identity: bool
    h3b_observed_compat: List[Optional[str]]
    h3b_expected_compat: str
    h3c_mch_identity: bool
    h3c_observed_mch: List[Optional[str]]
    h3c_expected_mch: str
    h3d_pred_divergence: bool
    h3d_observed_pred_shas: List[str]


@dataclass(frozen=True)
class Verdict:
    """Pre-registered decision-gate verdict per Agent B matrix."""
    label: str  # GO / GO-LIFTED-AND-RAISED / REFUTE / INDETERMINATE /
                # INDETERMINATE-DOWNGRADED-NEED-N10 / ABORT
    rationale: str
    next_action: str


def _compute_sha256(arr: np.ndarray) -> str:
    """SHA-256 of NumPy array (canonical bytes for cross-process comparison)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_seed_signal_dir(signal_dir: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load (predictions, labels, predicted_returns_sha256) for one seed."""
    predictions_path = signal_dir / "predictions.npy"
    labels_path = signal_dir / "labels.npy"
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing predictions.npy at {predictions_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels.npy at {labels_path}")

    predictions = np.load(predictions_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)

    if predictions.shape != labels.shape:
        raise ValueError(
            f"shape mismatch at {signal_dir}: "
            f"predictions={predictions.shape} vs labels={labels.shape}"
        )
    if predictions.ndim != 1:
        raise ValueError(f"expected 1-D predictions, got shape {predictions.shape}")

    pred_sha = _compute_sha256(predictions)
    return predictions.astype(np.int64), labels.astype(np.int64), pred_sha


def _scan_ledger_for_seed(
    ledger_records_dir: Path, sweep_id: str, seed: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Scan hft-ops ledger records for a seed; return (eph, compat_fp, mch).

    Per Phase Y composer (banner anti-drift):
      - experiment_provenance_hash: top-level field (Phase 8B INDEX_SCHEMA_VERSION ≥1.4.0)
      - compatibility_fingerprint: top-level field (Phase V.A.4)
      - model_config_hash: top-level field (#PY-94 closure via index_entry projection)
    """
    if not ledger_records_dir.exists():
        return None, None, None

    pattern = f"{sweep_id}*seed_{seed}*.json"
    matches = list(ledger_records_dir.glob(pattern))
    if not matches:
        # Try alternate pattern with cycle10 prefix
        pattern = f"cycle10_r19_multi_seed*seed_{seed}*.json"
        matches = list(ledger_records_dir.glob(pattern))
    if not matches:
        return None, None, None

    # If multiple matches, take the most recent
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    record_path = matches[0]
    try:
        with open(record_path) as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, None

    eph = record.get("experiment_provenance_hash")
    compat_fp = record.get("compatibility_fingerprint")
    mch = record.get("model_config_hash")

    # Fall back to nested locations if top-level missing
    if mch is None:
        mch = (record.get("training_config") or {}).get("model_config_hash")
    if compat_fp is None:
        compat_fp = (record.get("compatibility") or {}).get("fingerprint")

    return eph, compat_fp, mch


def compute_seed_result(
    seed: int,
    signal_dir: Path,
    ledger_records_dir: Path,
    sweep_id: str,
) -> SeedResult:
    """Per-seed Wilson 95% CI on PT precision + trust-column extraction."""
    predictions, labels, pred_sha = load_seed_signal_dir(signal_dir)
    pt_predicted = predictions == PT_CLASS_INDEX
    pt_predictions_count = int(pt_predicted.sum())
    if pt_predictions_count == 0:
        raise ValueError(
            f"seed {seed}: zero PT-class predictions; "
            f"H5 architectural invariant ALREADY VIOLATED (PT class collapse)"
        )
    pt_correct_count = int((pt_predicted & (labels == PT_CLASS_INDEX)).sum())
    precision = pt_correct_count / pt_predictions_count

    # Wilson 95% CI on per-seed PT precision
    bt = binomtest(k=pt_correct_count, n=pt_predictions_count)
    ci = bt.proportion_ci(confidence_level=1 - ALPHA, method="wilson")

    # Trust column extraction from hft-ops ledger
    eph, compat_fp, mch = _scan_ledger_for_seed(ledger_records_dir, sweep_id, seed)

    return SeedResult(
        seed=seed,
        pt_predictions_count=pt_predictions_count,
        pt_correct_count=pt_correct_count,
        pt_precision=precision,
        wilson_ci_low=float(ci.low),
        wilson_ci_high=float(ci.high),
        predicted_returns_sha256=pred_sha,
        experiment_provenance_hash=eph,
        compatibility_fingerprint=compat_fp,
        model_config_hash=mch,
    )


def compute_architectural_invariants(
    seeds: List[SeedResult],
    expected_compat_fp: str = R19_COMPAT_FINGERPRINT,
    expected_mch: str = R19_MODEL_CONFIG_HASH,
) -> ArchitecturalInvariantResult:
    """H3.a-d invariant checks per Agent B pre-registration.

    Mid-impl H3 split (per Agent A2 finding): H3.a now distinguishes
    "all populated" from "all distinct" for cleaner ABORT diagnostics.

    Mid-impl H5 dynamic anchor (per Agent A2 finding): expected_compat_fp +
    expected_mch can be overridden by caller (typically derived from R-19's
    stored signal_metadata.json) to make verdict robust to corpus reruns.
    """
    # H3.a — split into all_populated + all_distinct for diagnostics
    eph_values_raw = [s.experiment_provenance_hash for s in seeds]
    eph_values_populated = [v for v in eph_values_raw if v is not None]
    n_populated = len(eph_values_populated)
    n_distinct = len(set(eph_values_populated))
    h3a_all_populated = n_populated == len(seeds)
    h3a_all_distinct = n_distinct == n_populated and n_populated > 0
    h3a_passed = h3a_all_populated and h3a_all_distinct

    # H3.b corpus identity
    compat_values = [s.compatibility_fingerprint for s in seeds]
    distinct_compat = {v for v in compat_values if v is not None}
    h3b_passed = (
        len(distinct_compat) == 1
        and expected_compat_fp in distinct_compat
    )

    # H3.c model arch identity
    mch_values = [s.model_config_hash for s in seeds]
    distinct_mch = {v for v in mch_values if v is not None}
    h3c_passed = (
        len(distinct_mch) == 1
        and expected_mch in distinct_mch
    )

    # H3.d predicted-array divergence (RNG actually perturbs)
    pred_shas = [s.predicted_returns_sha256 for s in seeds]
    distinct_pred_shas = len(set(pred_shas))
    h3d_passed = distinct_pred_shas >= 2  # At least 2 different

    return ArchitecturalInvariantResult(
        h3a_distinct_eph=h3a_passed,
        h3a_all_populated=h3a_all_populated,
        h3a_observed_eph_count=n_populated,
        h3a_observed_distinct_count=n_distinct,
        h3a_expected_eph_count=len(seeds),
        h3b_compat_identity=h3b_passed,
        h3b_observed_compat=compat_values,
        h3b_expected_compat=expected_compat_fp,
        h3c_mch_identity=h3c_passed,
        h3c_observed_mch=mch_values,
        h3c_expected_mch=expected_mch,
        h3d_pred_divergence=h3d_passed,
        h3d_observed_pred_shas=pred_shas,
    )


def derive_r19_anchors(
    r19_signal_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Derive expected compat_fp + mch from R-19's stored signal_metadata.json.

    Mid-impl H5 fix per Agent A2: makes verdict robust to corpus reruns / base
    config drift between R-19 ship (2026-05-15) and multi-seed launch.

    Returns (compat_fp, mch). Falls back to hardcoded R19_COMPAT_FINGERPRINT +
    R19_MODEL_CONFIG_HASH if file missing OR fields absent (with stderr WARNING).
    """
    import sys as _sys

    if r19_signal_dir is None:
        r19_signal_dir = Path(
            "lob-model-trainer/outputs/experiments/r19_tlob_tb_v3p0_h30/signals/test"
        )
    metadata_path = r19_signal_dir / "signal_metadata.json"

    if not metadata_path.exists():
        print(
            f"WARNING: R-19 signal_metadata.json not found at {metadata_path}. "
            f"Falling back to hardcoded anchors: "
            f"compat_fp={R19_COMPAT_FINGERPRINT[:16]}..., "
            f"mch={R19_MODEL_CONFIG_HASH[:16]}...",
            file=_sys.stderr,
        )
        return R19_COMPAT_FINGERPRINT, R19_MODEL_CONFIG_HASH

    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"WARNING: failed to load R-19 signal_metadata.json: {exc}. "
            f"Falling back to hardcoded anchors.",
            file=_sys.stderr,
        )
        return R19_COMPAT_FINGERPRINT, R19_MODEL_CONFIG_HASH

    # Phase Y composer surfaces (per BACKTEST_INDEX banner)
    compat_fp = meta.get("compatibility_fingerprint")
    mch = meta.get("model_config_hash")
    if compat_fp is None:
        compat_fp = R19_COMPAT_FINGERPRINT
        print(
            f"WARNING: compat_fp absent from R-19 signal_metadata.json; "
            f"using hardcoded {compat_fp[:16]}...",
            file=_sys.stderr,
        )
    if mch is None:
        mch = R19_MODEL_CONFIG_HASH
        print(
            f"WARNING: model_config_hash absent from R-19 signal_metadata.json; "
            f"using hardcoded {mch[:16]}...",
            file=_sys.stderr,
        )
    return compat_fp, mch


def compute_bootstrap_ci(
    pt_precisions: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = 12345
) -> Tuple[float, float, int]:
    """Bootstrap CI on mean PT precision across seeds.

    Uses hft_metrics.block_bootstrap_ci v0.1.11 with paired=False (single-array
    mode; closes #PY-255 §0 reuse-first SSoT discipline). block_length=1 because
    seeds are INDEPENDENT (no temporal correlation across seeds).

    Verified signature (v0.1.11):
      block_bootstrap_ci(statistic_fn, x, y=None, *, paired=True,
                         n_bootstraps=1000, block_length=None, ci=0.95, seed=42)
      -> (estimate, ci_low, ci_high, n_nonfinite_replaced)

    Returns (ci_low, ci_high, n_nonfinite_replaced).
    """
    from hft_metrics.bootstrap import block_bootstrap_ci

    estimate, ci_low, ci_high, n_nonfinite = block_bootstrap_ci(
        statistic_fn=np.mean,
        x=pt_precisions,
        paired=False,           # single-array mode; M3 N=5 seeds independent
        n_bootstraps=n_bootstrap,
        block_length=1,         # seeds are INDEPENDENT, no temporal correlation
        ci=1 - ALPHA,           # 0.95 for α=0.05 (NOT alpha — param is `ci`)
        seed=seed,              # integer seed (NOT rng object)
    )
    return float(ci_low), float(ci_high), int(n_nonfinite)


def evaluate_gates(
    seeds: List[SeedResult],
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
) -> Dict[str, GateResult]:
    """Apply pre-registered H1.a/b/c + H2.a/b gates."""
    pt_precisions = np.array([s.pt_precision for s in seeds])
    mean_precision = float(pt_precisions.mean())
    std_precision = float(pt_precisions.std(ddof=1))  # sample std (Bessel)

    # H1.a Point-estimate persistence
    h1a_diff = abs(mean_precision - R19_PT_PRECISION_ANCHOR)
    h1a_passed = h1a_diff <= H1A_POINT_ESTIMATE_TOLERANCE
    h1a = GateResult(
        label="H1.a",
        passed=h1a_passed,
        observed=h1a_diff,
        threshold=H1A_POINT_ESTIMATE_TOLERANCE,
        description=(
            f"|mean - R-19 anchor 0.269| = {h1a_diff:.4f} "
            f"({'≤' if h1a_passed else '>'}) {H1A_POINT_ESTIMATE_TOLERANCE}"
        ),
    )

    # H1.b R-17a CI separation (BINDING)
    h1b_passed = bootstrap_ci_low > H1B_R17A_SEPARATION_THRESHOLD
    h1b = GateResult(
        label="H1.b",
        passed=h1b_passed,
        observed=bootstrap_ci_low,
        threshold=H1B_R17A_SEPARATION_THRESHOLD,
        description=(
            f"bootstrap CI lower {bootstrap_ci_low:.4f} "
            f"({'>' if h1b_passed else '≤'}) R-17a baseline {H1B_R17A_SEPARATION_THRESHOLD}"
        ),
    )

    # H1.c Cost-floor separation (INFORMATIONAL)
    h1c_passed = bootstrap_ci_low > H1C_COST_FLOOR_THRESHOLD
    h1c = GateResult(
        label="H1.c",
        passed=h1c_passed,
        observed=bootstrap_ci_low,
        threshold=H1C_COST_FLOOR_THRESHOLD,
        description=(
            f"bootstrap CI lower {bootstrap_ci_low:.4f} "
            f"({'>' if h1c_passed else '≤'}) cost-floor {H1C_COST_FLOOR_THRESHOLD} "
            f"(INFORMATIONAL)"
        ),
    )

    # H2.a Magnitude (single-side over R-17a)
    h2a_passed = mean_precision > H2A_MEAN_FLOOR
    h2a = GateResult(
        label="H2.a",
        passed=h2a_passed,
        observed=mean_precision,
        threshold=H2A_MEAN_FLOOR,
        description=(
            f"mean {mean_precision:.4f} "
            f"({'>' if h2a_passed else '≤'}) R-17a floor {H2A_MEAN_FLOOR}"
        ),
    )

    # H2.b Stability (σ)
    h2b_passed = std_precision < H2B_SIGMA_CEILING
    h2b = GateResult(
        label="H2.b",
        passed=h2b_passed,
        observed=std_precision,
        threshold=H2B_SIGMA_CEILING,
        description=(
            f"std {std_precision:.4f} "
            f"({'<' if h2b_passed else '≥'}) σ ceiling {H2B_SIGMA_CEILING}"
        ),
    )

    return {
        "H1.a": h1a,
        "H1.b": h1b,
        "H1.c": h1c,
        "H2.a": h2a,
        "H2.b": h2b,
    }


def classify_verdict(
    seeds: List[SeedResult],
    gates: Dict[str, GateResult],
    arch_invariants: ArchitecturalInvariantResult,
    bootstrap_ci_low: float,
) -> Verdict:
    """Apply pre-registered decision matrix per Agent B 2026-05-16 LATE NIGHT."""
    pt_precisions = np.array([s.pt_precision for s in seeds])
    mean_precision = float(pt_precisions.mean())
    std_precision = float(pt_precisions.std(ddof=1))

    # ABORT gate first — architectural invariant violation
    if not (
        arch_invariants.h3a_distinct_eph
        and arch_invariants.h3b_compat_identity
        and arch_invariants.h3c_mch_identity
        and arch_invariants.h3d_pred_divergence
    ):
        violations = []
        if not arch_invariants.h3a_distinct_eph:
            # Mid-impl H3 split: distinguish "all populated" vs "all distinct"
            if not arch_invariants.h3a_all_populated:
                violations.append(
                    f"H3.a (epH coverage): only {arch_invariants.h3a_observed_eph_count}/"
                    f"{arch_invariants.h3a_expected_eph_count} seeds have populated "
                    f"experiment_provenance_hash. Phase Y composer may have silently failed "
                    f"on some seeds (likely missing trust column inputs)."
                )
            else:
                violations.append(
                    f"H3.a (epH distinctness): {arch_invariants.h3a_observed_distinct_count} "
                    f"distinct epH among {arch_invariants.h3a_observed_eph_count} populated. "
                    f"RNG state may be silently swallowed (TLOB ran as if RNG-free); "
                    f"check DESIGN-1 A.1+A.2 RNG capture."
                )
        if not arch_invariants.h3b_compat_identity:
            violations.append(
                f"H3.b (compat identity): observed {set(arch_invariants.h3b_observed_compat)}, "
                f"expected {arch_invariants.h3b_expected_compat!r}"
            )
        if not arch_invariants.h3c_mch_identity:
            violations.append(
                f"H3.c (mch identity): observed {set(arch_invariants.h3c_observed_mch)}, "
                f"expected {arch_invariants.h3c_expected_mch!r}"
            )
        if not arch_invariants.h3d_pred_divergence:
            violations.append(
                f"H3.d (pred divergence): all predictions identical (RNG silently swallowed?)"
            )
        return Verdict(
            label="ABORT",
            rationale=f"H3 architectural invariant FAILED: {'; '.join(violations)}",
            next_action=(
                "Investigate cycle: config drift / RNG capture failure / Phase Y "
                "composer regression. Do NOT issue scientific verdict until root-caused."
            ),
        )

    # M5 INTERIM-ANALYSIS GATE — variance too high
    if std_precision > M5_VARIANCE_ABORT_SIGMA:
        return Verdict(
            label="INDETERMINATE-DOWNGRADED-NEED-N10",
            rationale=(
                f"M5 gate FAILED: σ(PT_precision) = {std_precision:.4f} > "
                f"{M5_VARIANCE_ABORT_SIGMA}. Variance assumption σ≈2% FALSIFIED. "
                f"Current N=5 power insufficient for confident verdict."
            ),
            next_action=(
                "Manual follow-up cycle: add seeds 48-52 for N=10 "
                "(cycle11_r19_multi_seed_extension.yaml)."
            ),
        )

    # GO-LIFTED-AND-RAISED — mean above R-19 anchor by >2pp
    if (
        mean_precision > GO_LIFTED_THRESHOLD
        and gates["H1.b"].passed
        and gates["H2.a"].passed
        and gates["H2.b"].passed
    ):
        return Verdict(
            label="GO-LIFTED-AND-RAISED",
            rationale=(
                f"mean PT_precision {mean_precision:.4f} > {GO_LIFTED_THRESHOLD} "
                f"(R-19 anchor + 2pp). R-19 was conservative single-seed. "
                f"All H1.b/H2.a/H2.b/H3 gates PASS."
            ),
            next_action=(
                "Document new central estimate. Re-evaluate cost-economics break-even "
                "(35.7% PT precision required for tradeable alpha at 40/20 barriers). "
                "If mean crosses 35.7%, cost-floor FLIPS — major re-prioritization."
            ),
        )

    # GO — validated architectural lift
    if (
        gates["H1.a"].passed
        and gates["H1.b"].passed
        and gates["H2.a"].passed
        and gates["H2.b"].passed
    ):
        return Verdict(
            label="GO",
            rationale=(
                f"All pre-registered gates PASS: H1.a (mean ±2pp of R-19), "
                f"H1.b (CI lower > R-17a baseline), H2.a (mean > 0.220), "
                f"H2.b (σ < 0.020), H3 architectural invariants. "
                f"R-19's +4.9pp lift VALIDATED as architectural (not seed noise)."
            ),
            next_action=(
                "R-17a Lesson #95 REFUTATION holds. Phase 2 TB design matrix can rest "
                "on R-19 anchor. Next-cycle authorization for Phase 2 TB with informed prior."
            ),
        )

    # REFUTE — H1.b FAIL (CI lower ≤ R-17a) OR mean ≤ R-17a
    if not gates["H1.b"].passed or mean_precision <= R17A_PT_PRECISION_BASELINE:
        return Verdict(
            label="REFUTE",
            rationale=(
                f"H1.b FAILED: bootstrap CI lower {bootstrap_ci_low:.4f} ≤ "
                f"R-17a baseline {R17A_PT_PRECISION_BASELINE}, OR mean "
                f"{mean_precision:.4f} ≤ R-17a baseline. R-19's +4.9pp lift was "
                f"within seed variance; FALSIFIED."
            ),
            next_action=(
                "Close R-17a Lesson #95 REFUTATION as PREMATURE. Phase 2 TB matrix "
                "INVALIDATED (rests on falsified premise). Pivot to: "
                "(a) R-20 HMHP cascade (different architecture class) OR "
                "(b) R-18 cost-aware barrier sweep (with theta scaling + slippage realism)."
            ),
        )

    # INDETERMINATE — directional but magnitude uncertain
    return Verdict(
        label="INDETERMINATE",
        rationale=(
            f"H1.a borderline (mean {mean_precision:.4f} not within ±2pp of R-19 anchor "
            f"0.269) BUT H1.b PASSES (CI lower {bootstrap_ci_low:.4f} > 0.220). "
            f"Lift directionally real but magnitude uncertain."
        ),
        next_action=(
            "Report mean ± bootstrap CI. Manual follow-up cycle decision: "
            "extend to N=10 OR accept partial-lift claim with current CI."
        ),
    )


def assemble_verdict_json(
    seeds: List[SeedResult],
    gates: Dict[str, GateResult],
    arch_invariants: ArchitecturalInvariantResult,
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
    n_nonfinite_replaced: int,
    verdict: Verdict,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Assemble verdict JSON for ledger storage."""
    pt_precisions = np.array([s.pt_precision for s in seeds])
    return {
        "schema_version": "1.0",
        "analysis_kind": "r19_multi_seed_validation",
        "cycle": "cycle10_r19_multi_seed",
        "backlog_ref": "#PY-243",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registered_constants": {
            "alpha": ALPHA,
            "n_bootstrap": N_BOOTSTRAP,
            "pt_class_index": PT_CLASS_INDEX,
            "r19_pt_precision_anchor": R19_PT_PRECISION_ANCHOR,
            "r19_compat_fingerprint": R19_COMPAT_FINGERPRINT,
            "r19_model_config_hash": R19_MODEL_CONFIG_HASH,
            "r17a_pt_precision_baseline": R17A_PT_PRECISION_BASELINE,
            "h1a_tolerance": H1A_POINT_ESTIMATE_TOLERANCE,
            "h1b_threshold": H1B_R17A_SEPARATION_THRESHOLD,
            "h1c_threshold": H1C_COST_FLOOR_THRESHOLD,
            "h2a_floor": H2A_MEAN_FLOOR,
            "h2b_sigma_ceiling": H2B_SIGMA_CEILING,
            "m5_variance_abort_sigma": M5_VARIANCE_ABORT_SIGMA,
            "go_lifted_threshold": GO_LIFTED_THRESHOLD,
        },
        "inputs": {
            "sweep_id": args.sweep_id,
            "signal_root": str(args.signal_root),
            "ledger_records_dir": str(args.ledger_records_dir),
            "n_seeds": len(seeds),
        },
        "seeds": [
            {
                "seed": s.seed,
                "pt_predictions": s.pt_predictions_count,
                "pt_correct": s.pt_correct_count,
                "pt_precision": s.pt_precision,
                "wilson_ci_low": s.wilson_ci_low,
                "wilson_ci_high": s.wilson_ci_high,
                "predicted_returns_sha256": s.predicted_returns_sha256,
                "experiment_provenance_hash": s.experiment_provenance_hash,
                "compatibility_fingerprint": s.compatibility_fingerprint,
                "model_config_hash": s.model_config_hash,
            }
            for s in seeds
        ],
        "aggregate": {
            "mean_pt_precision": float(pt_precisions.mean()),
            "std_pt_precision": float(pt_precisions.std(ddof=1)),
            "min_pt_precision": float(pt_precisions.min()),
            "max_pt_precision": float(pt_precisions.max()),
            "bootstrap_ci_low": bootstrap_ci_low,
            "bootstrap_ci_high": bootstrap_ci_high,
            "bootstrap_n_nonfinite_replaced": n_nonfinite_replaced,
        },
        "gates": {
            label: {
                "passed": gate.passed,
                "observed": gate.observed,
                "threshold": gate.threshold,
                "description": gate.description,
            }
            for label, gate in gates.items()
        },
        "h3_architectural_invariants": {
            "h3a_distinct_eph": arch_invariants.h3a_distinct_eph,
            "h3a_all_populated": arch_invariants.h3a_all_populated,
            "h3a_observed_eph_count": arch_invariants.h3a_observed_eph_count,
            "h3a_observed_distinct_count": arch_invariants.h3a_observed_distinct_count,
            "h3a_expected_eph_count": arch_invariants.h3a_expected_eph_count,
            "h3b_compat_identity": arch_invariants.h3b_compat_identity,
            "h3b_observed_compat": arch_invariants.h3b_observed_compat,
            "h3b_expected_compat": arch_invariants.h3b_expected_compat,
            "h3c_mch_identity": arch_invariants.h3c_mch_identity,
            "h3c_observed_mch": arch_invariants.h3c_observed_mch,
            "h3c_expected_mch": arch_invariants.h3c_expected_mch,
            "h3d_pred_divergence": arch_invariants.h3d_pred_divergence,
            "h3d_distinct_pred_shas": len(set(arch_invariants.h3d_observed_pred_shas)),
        },
        "verdict": {
            "label": verdict.label,
            "rationale": verdict.rationale,
            "next_action": verdict.next_action,
        },
        "cross_ref": {
            "design_gate": "Pre-impl Wave 2026-05-16 LATE NIGHT (3 agents: readiness + gates + REFUTE)",
            "backlog_entries": ["#PY-243"],
            "consumes_ssot": "hft_metrics.block_bootstrap_ci v0.1.11 (paired=False; #PY-255 closure)",
            "pattern_reuse": "analyze_r17a_vs_r19_pt_precision.py (Wilson CI + atomic_write_json)",
        },
    }


def render_human_report(verdict_json: Dict[str, Any]) -> str:
    """Render verdict as human-readable text."""
    lines = []
    lines.append("=" * 78)
    lines.append("#PY-243 R-19 MULTI-SEED VALIDATION VERDICT")
    lines.append("=" * 78)
    lines.append("")

    pc = verdict_json["pre_registered_constants"]
    lines.append("PRE-REGISTERED CONSTANTS (Agent B LOCKED 2026-05-16 LATE NIGHT):")
    lines.append(f"  α                       = {pc['alpha']}")
    lines.append(f"  N_bootstrap             = {pc['n_bootstrap']}")
    lines.append(f"  R-19 PT precision anchor= {pc['r19_pt_precision_anchor']:.4f}")
    lines.append(f"  R-17a baseline          = {pc['r17a_pt_precision_baseline']:.4f}")
    lines.append(f"  H1.a tolerance          = ±{pc['h1a_tolerance']:.4f}")
    lines.append(f"  H1.b threshold (BINDING)= > {pc['h1b_threshold']:.4f}")
    lines.append(f"  H1.c threshold (INFO)   = > {pc['h1c_threshold']:.4f}")
    lines.append(f"  H2.a floor              = > {pc['h2a_floor']:.4f}")
    lines.append(f"  H2.b σ ceiling          = < {pc['h2b_sigma_ceiling']:.4f}")
    lines.append(f"  M5 variance abort       = > {pc['m5_variance_abort_sigma']:.4f}")
    lines.append(f"  GO-LIFTED threshold     = > {pc['go_lifted_threshold']:.4f}")
    lines.append("")

    lines.append(f"PER-SEED RESULTS (N={len(verdict_json['seeds'])}):")
    lines.append(
        "  seed │ PT pred │ PT correct │ PT precision │ Wilson 95% CI       │ pred SHA-256"
    )
    lines.append(
        "  ─────┼─────────┼────────────┼──────────────┼─────────────────────┼─────────────"
    )
    for s in verdict_json["seeds"]:
        lines.append(
            f"  {s['seed']:>4} │ {s['pt_predictions']:>7} │ {s['pt_correct']:>10} │"
            f" {s['pt_precision']:.4f}       │ [{s['wilson_ci_low']:.4f}, {s['wilson_ci_high']:.4f}] │"
            f" {s['predicted_returns_sha256'][:12]}..."
        )
    lines.append("")

    agg = verdict_json["aggregate"]
    lines.append("AGGREGATE STATISTICS:")
    lines.append(f"  mean PT precision       = {agg['mean_pt_precision']:.4f}")
    lines.append(f"  std PT precision        = {agg['std_pt_precision']:.4f}")
    lines.append(f"  [min, max]              = [{agg['min_pt_precision']:.4f}, {agg['max_pt_precision']:.4f}]")
    lines.append(
        f"  Bootstrap 95% CI on mean= [{agg['bootstrap_ci_low']:.4f}, {agg['bootstrap_ci_high']:.4f}]"
    )
    lines.append(f"  Bootstrap NaN replaced  = {agg['bootstrap_n_nonfinite_replaced']}")
    lines.append("")

    lines.append("PRE-REGISTERED GATES:")
    for label, gate in verdict_json["gates"].items():
        symbol = "✓" if gate["passed"] else "✗"
        lines.append(f"  {symbol} {label}: {gate['description']}")
    lines.append("")

    h3 = verdict_json["h3_architectural_invariants"]
    lines.append("H3 ARCHITECTURAL INVARIANTS:")
    h3a_pop_sym = '✓' if h3.get('h3a_all_populated', True) else '✗'
    h3a_dist_sym = '✓' if h3['h3a_distinct_eph'] else '✗'
    lines.append(
        f"  {h3a_pop_sym} H3.a populated: {h3['h3a_observed_eph_count']}/"
        f"{h3['h3a_expected_eph_count']} seeds have populated epH"
    )
    lines.append(
        f"  {h3a_dist_sym} H3.a distinct: {h3.get('h3a_observed_distinct_count', h3['h3a_observed_eph_count'])} "
        f"distinct epH (expect {h3['h3a_expected_eph_count']})"
    )
    lines.append(
        f"  {'✓' if h3['h3b_compat_identity'] else '✗'} H3.b: compat identity to R-19 anchor"
    )
    lines.append(
        f"  {'✓' if h3['h3c_mch_identity'] else '✗'} H3.c: mch identity to R-19 anchor"
    )
    lines.append(
        f"  {'✓' if h3['h3d_pred_divergence'] else '✗'} H3.d: pred SHA-256 divergence = "
        f"{h3['h3d_distinct_pred_shas']}/{verdict_json['inputs']['n_seeds']}"
    )
    lines.append("")

    v = verdict_json["verdict"]
    lines.append("VERDICT:")
    lines.append(f"  Label: {v['label']}")
    lines.append(f"  Rationale: {v['rationale']}")
    lines.append(f"  Next action: {v['next_action']}")
    lines.append("")

    cr = verdict_json["cross_ref"]
    lines.append("CROSS-REF:")
    lines.append(f"  Design gate:    {cr['design_gate']}")
    lines.append(f"  Backlog:        {', '.join(cr['backlog_entries'])}")
    lines.append(f"  SSoT consumed:  {cr['consumes_ssot']}")
    lines.append(f"  Pattern reuse:  {cr['pattern_reuse']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze #PY-243 R-19 multi-seed validation per Agent B LOCKED pre-registration."
    )
    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep ID (e.g., cycle10_r19_multi_seed_20260516T020000)",
    )
    parser.add_argument(
        "--signal-root",
        type=Path,
        default=Path("lob-model-trainer/outputs/experiments"),
        help="Root dir containing per-seed signal directories.",
    )
    parser.add_argument(
        "--ledger-records-dir",
        type=Path,
        default=Path("hft-ops/ledger/records"),
        help="hft-ops ledger records directory (for trust-column extraction).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[43, 44, 45, 46, 47],
        help="List of seed values to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hft-ops/ledger/r19_multi_seed_verdicts"),
        help="Directory to write verdict JSON.",
    )
    args = parser.parse_args()

    # Resolve paths to absolute for downstream cite-trail
    args.signal_root = args.signal_root.expanduser().absolute()
    args.ledger_records_dir = args.ledger_records_dir.expanduser().absolute()
    args.output_dir = args.output_dir.expanduser().absolute()

    # Compute per-seed results
    seeds: List[SeedResult] = []
    for seed in args.seeds:
        signal_dir = args.signal_root / f"cycle10_r19_multi_seed__seed_{seed}" / "signals" / "test"
        if not signal_dir.exists():
            raise FileNotFoundError(
                f"signal_dir not found for seed {seed}: {signal_dir}. "
                f"Has the sweep completed and exported signals for this seed?"
            )
        result = compute_seed_result(
            seed=seed,
            signal_dir=signal_dir,
            ledger_records_dir=args.ledger_records_dir,
            sweep_id=args.sweep_id,
        )
        seeds.append(result)

    # Compute bootstrap CI on per-seed PT precisions
    pt_precisions_array = np.array([s.pt_precision for s in seeds])
    bootstrap_ci_low, bootstrap_ci_high, n_nonfinite = compute_bootstrap_ci(
        pt_precisions_array
    )

    # H3 architectural invariants — mid-impl H5 dynamic anchor derivation
    r19_signal_dir = args.signal_root.parent.parent / "lob-model-trainer" / "outputs" / "experiments" / "r19_tlob_tb_v3p0_h30" / "signals" / "test"
    # The above heuristic-path-derivation assumes signal-root is at lob-model-trainer/outputs/experiments;
    # if invoked from monorepo root the path resolves correctly via .parent.parent climb.
    # Default fallback: pass None → derive_r19_anchors uses its own default path.
    if not r19_signal_dir.exists():
        # Fall back to default path resolution inside derive_r19_anchors
        expected_compat_fp, expected_mch = derive_r19_anchors()
    else:
        expected_compat_fp, expected_mch = derive_r19_anchors(r19_signal_dir)
    arch_invariants = compute_architectural_invariants(
        seeds,
        expected_compat_fp=expected_compat_fp,
        expected_mch=expected_mch,
    )

    # Evaluate H1/H2 gates
    gates = evaluate_gates(seeds, bootstrap_ci_low, bootstrap_ci_high)

    # Apply pre-registered decision matrix
    verdict = classify_verdict(seeds, gates, arch_invariants, bootstrap_ci_low)

    # Assemble verdict JSON
    verdict_json = assemble_verdict_json(
        seeds=seeds,
        gates=gates,
        arch_invariants=arch_invariants,
        bootstrap_ci_low=bootstrap_ci_low,
        bootstrap_ci_high=bootstrap_ci_high,
        n_nonfinite_replaced=n_nonfinite,
        verdict=verdict,
        args=args,
    )

    # Print human-readable
    print(render_human_report(verdict_json))

    # Write verdict JSON via atomic_write_json SSoT (closes #PY-73 atomic-write
    # discipline at this site)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    verdict_path = args.output_dir / f"{args.sweep_id}_verdict_{timestamp}.json"
    from hft_contracts.atomic_io import atomic_write_json
    atomic_write_json(verdict_path, verdict_json, sort_keys=True)
    print(f"\nVerdict JSON written to: {verdict_path}")

    # Exit-code convention per Agent B + R-16e precedent
    if verdict.label in ("GO", "GO-LIFTED-AND-RAISED"):
        exit_code = 0
    elif verdict.label == "ABORT":
        exit_code = 2
    else:
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
