"""Analyze R-20-HMHP-R multi-seed validation cycle (cycle12_r20_hmhp_r_multi_seed).

Pre-registered decision-gate analysis per cycle12_r20_hmhp_r_multi_seed.yaml
LOCKED PRE-RUN 2026-05-19 NIGHT (post Wave 1+2 adversarial REFUTE; user authorized
R-20-HMHP-R Multi-Seed via AskUserQuestion).

PURPOSE: Validate HMHP-R cascading-decoder architecture-axis lift survives
perturbation across N=5 seeds (43-47) per Lesson #12 mandatory multi-seed
rigor floor + R-16e Lesson #93 precedent. Tests:
  (a) absolute predictive floor: mean(test_h10_ic) > 0.30
  (b) paired-bootstrap CI lower > 0.10 (binding)
  (c) Phase Y composer SEED-INVARIANT invariant: 5 seeds → 1 IDENTICAL epH/compat_fp/mch
  (d) HMHP-R ConfirmationModule emission: agreement_ratio.npy non-degenerate per seed
  (e) PT vs TLOB Stage 2 baseline (0.3747) — informational comparison

PRE-REGISTERED HYPOTHESES (LOCKED before script runs):

  H1 PRIMARY — Architectural lift floor + TLOB comparison (multi-seed):
    H1.a Architectural floor: mean(test_h10_ic_seeds_43_47) > 0.30
    H1.b Paired-bootstrap CI lower (BINDING):
         block_bootstrap_ci(test_h10_ic_seeds_43_47, paired=False,
                            block_length=1, n_bootstraps=10000, α=0.05).ci_lower > 0.10
    H1.c PARTIAL-LIFT vs TLOB 0.3747: mean > 0.3247 (within 5pp)
    H1.d CLEAR-LIFT: mean > 0.3747

  H2 MULTI-HORIZON (HMHP-R cascading capability):
    H2.a mean(test_h60_ic_seeds_43_47) > 0.10 (Stage 6 anchor 0.1408)
    H2.b mean(test_h300_ic_seeds_43_47) > 0.05 (Stage 6 anchor 0.0820)
    H2.c std(test_h10_ic_seeds_43_47) < 0.040 (seed-variance stability)

  H3 ARCHITECTURAL INVARIANT (Phase Y composer SEED-INVARIANT):
    H3.a All 5 seeds populated compatibility_fingerprint
    H3.b All 5 seeds share IDENTICAL compatibility_fingerprint (empirical anchor)
    H3.c All 5 seeds populated experiment_provenance_hash
    H3.d All 5 seeds share IDENTICAL experiment_provenance_hash (seed-invariant composer)
    H3.e All 5 seeds populated model_config_hash (top OR nested) + IDENTICAL across seeds
    H3.f Predicted-array RNG sanity: ≥4-of-5 distinct sha256(predicted_returns.npy)

  H4 CONFIRMATION MODULE (HMHP-R unique architectural feature):
    H4.a agreement_ratio.npy emitted for ALL 5 seeds
    H4.b mean(agreement_ratio) ∈ [0.4, 0.9] for ALL 5 seeds
    H4.c std(agreement_ratio) > 0.05 for ALL 5 seeds

  H5 COST GATE — Informational (NOT binding per Lesson #106)

DECISION MATRIX (LOCKED PRE-RUN):
  GO-CLEAR-LIFT: H1.a + H1.b + H1.d + H2 + H3 + H4 PASS
  GO-COMPETITIVE: H1.a + H1.b + H1.c + H2 + H3 + H4 PASS, BUT H1.d FAIL
  PARTIAL-LIFT: H1.a + H1.b + H2.a/b + H3 + H4 PASS, BUT H1.c FAIL
  REFUTE: H1.a FAIL OR H1.b FAIL
  INDETERMINATE-DOWNGRADED-NEED-N10: H2.c FAIL AND H1.a PASS
  ABORT: any H3.a-e FAIL OR H4.a FAIL

Inputs:
  - Sweep ID (positional arg), e.g. "cycle12_r20_hmhp_r_multi_seed_20260519T2030"
  - 5 ledger records under hft-ops/ledger/runs/<sweep_id>/ OR hft-ops/ledger/records/
  - 5 signal dirs under outputs/experiments/cycle12_r20_hmhp_r_multi_seed__seed_NN/signals/test/

Output:
  - Human-readable verdict to stdout
  - JSON verdict at hft-ops/ledger/r20_multi_seed_verdicts/<sweep_id>_verdict_<ts>.json

Cross-ref: pre-impl 3-agent gate (Agent A/Y/Z) LOCKED 2026-05-19 NIGHT;
analyze_r19_multi_seed.py pattern reuse per hft-rules §0;
hft_metrics.block_bootstrap_ci v0.1.11 SSoT (paired=False single-array mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse hft_contracts atomic-write SSoT per hft-rules §0
from hft_contracts.atomic_io import atomic_write_json

# Reuse hft_metrics bootstrap SSoT per hft-rules §0
# (v0.1.11 paired=False single-array mode; #PY-255 closure 2026-05-15)
from hft_metrics.bootstrap import block_bootstrap_ci


# ---------------------------------------------------------------------------
# Pre-registered constants (LOCKED before script runs per cycle12 multi-seed
# manifest decision-gate section)
# ---------------------------------------------------------------------------

# Bootstrap settings (paired=False single-array mode; block_length=1 because
# seeds are INDEPENDENT; no temporal correlation across seeds)
ALPHA: float = 0.05
N_BOOTSTRAP: int = 10000
BLOCK_LENGTH: int = 1

# H1 PRIMARY thresholds
H1A_FLOOR: float = 0.30
H1B_CI_LOWER_FLOOR: float = 0.10  # paired-bootstrap CI lower bound BINDING
H1C_PARTIAL_LIFT: float = 0.3247  # TLOB Stage 2 - 5pp
H1D_CLEAR_LIFT: float = 0.3747    # TLOB Stage 2 baseline
TLOB_STAGE_2_BASELINE: float = 0.3747

# H2 MULTI-HORIZON thresholds
H2A_H60_FLOOR: float = 0.10
H2B_H300_FLOOR: float = 0.05
H2C_SIGMA_CEILING: float = 0.040

# H4 ConfirmationModule sanity
H4_MEAN_LOW: float = 0.4
H4_MEAN_HIGH: float = 0.9
H4_STD_MIN: float = 0.05

# H3.f predicted-array RNG sanity (≥4 distinct out of 5 seeds)
H3F_MIN_DISTINCT_PRED_SHAS: int = 4

# Stage 6 anchors (informational; NOT binding per cycle12 multi-seed Wave 2Y REFUTE)
STAGE_6_H10_ANCHOR: float = 0.3561
STAGE_6_H60_ANCHOR: float = 0.1408
STAGE_6_H300_ANCHOR: float = 0.0820

# Expected seed values
EXPECTED_SEEDS: Tuple[int, ...] = (43, 44, 45, 46, 47)


@dataclass(frozen=True)
class SeedResult:
    """Per-seed metrics + trust columns + agreement stats + RNG fingerprint."""

    seed: int
    test_h10_ic: float
    test_h60_ic: float
    test_h300_ic: float
    compat_fp: Optional[str]
    experiment_provenance_hash: Optional[str]
    model_config_hash: Optional[str]
    agreement_mean: Optional[float]
    agreement_std: Optional[float]
    predicted_returns_sha256: Optional[str]
    record_path: str


@dataclass
class GateResult:
    """Pre-registered gate result with explicit threshold + observed."""

    name: str
    threshold: Optional[float]
    observed: Any
    passed: bool
    notes: str


@dataclass
class R20MultiSeedVerdict:
    """Full R-20 multi-seed verdict bundle."""

    sweep_id: str
    timestamp_utc: str
    n_seeds: int
    seed_results: List[Dict[str, Any]] = field(default_factory=list)
    mean_h10_ic: float = 0.0
    std_h10_ic: float = 0.0
    ci_low_h10_ic: float = 0.0
    ci_high_h10_ic: float = 0.0
    ci_n_nonfinite_replaced: int = 0
    mean_h60_ic: float = 0.0
    mean_h300_ic: float = 0.0
    distinct_compat_fps: List[str] = field(default_factory=list)
    distinct_epHs: List[str] = field(default_factory=list)
    distinct_mch: List[str] = field(default_factory=list)
    distinct_pred_shas: int = 0
    gates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verdict: str = ""
    verdict_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "timestamp_utc": self.timestamp_utc,
            "n_seeds": self.n_seeds,
            "seed_results": self.seed_results,
            "mean_h10_ic": self.mean_h10_ic,
            "std_h10_ic": self.std_h10_ic,
            "ci_low_h10_ic": self.ci_low_h10_ic,
            "ci_high_h10_ic": self.ci_high_h10_ic,
            "ci_n_nonfinite_replaced": self.ci_n_nonfinite_replaced,
            "mean_h60_ic": self.mean_h60_ic,
            "mean_h300_ic": self.mean_h300_ic,
            "distinct_compat_fps": self.distinct_compat_fps,
            "distinct_epHs": self.distinct_epHs,
            "distinct_mch": self.distinct_mch,
            "distinct_pred_shas": self.distinct_pred_shas,
            "gates": self.gates,
            "verdict": self.verdict,
            "verdict_summary": self.verdict_summary,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _compute_sha256(arr: np.ndarray) -> str:
    """SHA-256 of NumPy array (canonical bytes; cross-process comparable)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _find_training_records(sweep_id: str, pipeline_root: Path) -> List[Path]:
    """Locate the 5 seed_{43..47} training records for given sweep.

    Records are written under one of:
      - hft-ops/ledger/runs/<sweep_id>/<exp_name>_seed_NN_*.json (sweep-run dir)
      - hft-ops/ledger/records/<exp_name>__seed_NN_*.json (canonical records dir)

    We check sweep-run dir first (preferred for fresh sweeps), then fallback
    to records dir.
    """
    runs_dir = pipeline_root / "hft-ops" / "ledger" / "runs" / sweep_id
    records_dir = pipeline_root / "hft-ops" / "ledger" / "records"

    matches: List[Path] = []
    if runs_dir.exists():
        matches.extend(sorted(runs_dir.glob("*seed_*.json")))
    if not matches:
        # Fallback to canonical records dir; glob by experiment name prefix
        # (sweep_id prefix matches experiment.name from manifest)
        prefix = sweep_id.split("_2026", 1)[0] if "_2026" in sweep_id else sweep_id
        matches.extend(sorted(records_dir.glob(f"{prefix}*seed_*.json")))

    if not matches:
        raise FileNotFoundError(
            f"No training records found for sweep_id={sweep_id} "
            f"under {runs_dir} or {records_dir}"
        )

    return matches


def _find_signal_dir(seed: int, pipeline_root: Path) -> Path:
    """Locate per-seed signal export directory.

    NOTE (output_dir bug systemic, Cycle 11 anti-drift #10): sweep manifests
    expand `${sweep.point_name}` to BARE label segment (e.g., "seed_43") not
    fully-qualified `cycle12_r20_hmhp_r_multi_seed__seed_43`. Architecturally
    deferred — known systemic bug across cycle5-cycle10.

    Per Phase V.1 L1.2 signal_export_output_dir preservation: per-seed signal
    arrays are preserved at `outputs/experiments/seed_{N}/signals/test/`.
    """
    sig_dir = (
        pipeline_root
        / "outputs"
        / "experiments"
        / f"seed_{seed}"
        / "signals"
        / "test"
    )
    return sig_dir


def _validate_signal_metadata_belongs_to_this_sweep(
    signal_dir: Path, expected_horizons: List[int]
) -> Tuple[bool, str]:
    """Verify signal_metadata.json matches THIS sweep's contract signature.

    STALE-SIGNAL HAZARD MITIGATION: `outputs/experiments/seed_{N}/signals/test/`
    directories may contain stale signals from prior cycles (e.g., cycle10_r19
    TB classification with horizons=[30] / labels.npy / predictions.npy). If
    our sweep FAILS for some seed, stale signals remain. This check detects.

    Validation strategy (per signal_metadata.json empirical structure 2026-05-19):
    1. Check `compatibility.horizons` matches expected (R-20: [10, 60, 300])
       — R-20-HMHP-R distinguishes from cycle10_r19 TB classification (horizons=[30])
    2. Check `predicted_returns.npy` exists (regression artifact NOT in TB)
    3. Check `compatibility.data_source == "mbo_lob"` (sanity)

    Returns (is_valid, diagnostic_str).
    """
    metadata_path = signal_dir / "signal_metadata.json"
    if not metadata_path.exists():
        return False, f"signal_metadata.json missing at {metadata_path}"
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        return False, f"signal_metadata.json parse failed: {exc}"

    # Check horizons match (load-bearing distinguisher; R-20=[10,60,300] vs TB=[30])
    compatibility = metadata.get("compatibility", {}) or {}
    observed_horizons = compatibility.get("horizons", [])
    if list(observed_horizons) != list(expected_horizons):
        return False, (
            f"signal_metadata.compatibility.horizons={observed_horizons} does not "
            f"match expected={expected_horizons} — likely STALE signals from prior "
            f"cycle (cycle10_r19 TB used horizons=[30]). Re-run signal_export."
        )

    # Verify regression format (predicted_returns.npy exists; TB uses labels.npy + predictions.npy)
    if not (signal_dir / "predicted_returns.npy").exists():
        return False, (
            f"predicted_returns.npy missing — likely STALE classification signals "
            f"from prior cycle (TB had predictions.npy + labels.npy). "
            f"Re-run signal_export for this seed."
        )

    # Sanity: data_source must be mbo_lob (LOB-based features)
    data_source = compatibility.get("data_source", metadata.get("data_source", ""))
    if data_source and data_source != "mbo_lob":
        return False, f"data_source='{data_source}' (expected 'mbo_lob')"

    return True, "OK"


def _load_agreement_array(signal_dir: Path) -> Optional[np.ndarray]:
    """Load agreement_ratio.npy if present (HMHP-R ConfirmationModule output)."""
    agree_path = signal_dir / "agreement_ratio.npy"
    if not agree_path.exists():
        return None
    return np.load(agree_path, allow_pickle=False)


def _load_predicted_returns_sha(signal_dir: Path) -> Optional[str]:
    """Load predicted_returns.npy + compute SHA-256 for H3.f RNG sanity."""
    pred_path = signal_dir / "predicted_returns.npy"
    if not pred_path.exists():
        return None
    arr = np.load(pred_path, allow_pickle=False)
    return _compute_sha256(arr)


def _extract_seed_from_record_path(record_path: Path) -> Optional[int]:
    """Extract seed integer from filename like '...seed_43_...json'."""
    name = record_path.name
    # Look for "seed_NN" substring
    for token in name.split("_"):
        if token.startswith("seed"):
            continue
    # More robust: search for "seed_<int>"
    import re

    m = re.search(r"seed_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def _load_seed_result(record_path: Path, pipeline_root: Path) -> SeedResult:
    """Load per-seed training record + signal artifacts."""
    seed = _extract_seed_from_record_path(record_path)
    if seed is None:
        raise ValueError(f"Cannot extract seed from path: {record_path}")

    record = json.loads(record_path.read_text())
    training_metrics = record.get("training_metrics") or {}

    # Test IC metrics (HMHP-R emits test_h10_ic / test_h60_ic / test_h300_ic
    # via per-horizon test evaluation in trainer)
    test_h10_ic = float(
        training_metrics.get("test_h10_ic", training_metrics.get("test_ic", 0.0))
    )
    test_h60_ic = float(training_metrics.get("test_h60_ic", 0.0))
    test_h300_ic = float(training_metrics.get("test_h300_ic", 0.0))

    # Trust columns (Phase Y composer)
    compat_fp = record.get("compatibility_fingerprint")
    epH = record.get("experiment_provenance_hash")
    mch_top = record.get("model_config_hash")
    mch_nested = (record.get("training_config") or {}).get("model_config_hash")
    mch = mch_top if mch_top else mch_nested

    # Signal artifacts — validate signal_metadata belongs to THIS sweep
    # (stale-signal hazard: seed_43-47 dirs contain cycle10_r19 TB classification
    # signals from prior cycle; only OVERWRITTEN by fresh R-20 signal_export).
    # Use compatibility.horizons + predicted_returns.npy as load-bearing
    # distinguishers (signal_metadata has no experiment.name field empirically).
    signal_dir = _find_signal_dir(seed, pipeline_root)
    is_valid, diagnostic = _validate_signal_metadata_belongs_to_this_sweep(
        signal_dir, expected_horizons=[10, 60, 300]
    )
    if not is_valid:
        # Fail-loud per hft-rules §5; stale signals would silently corrupt verdict
        raise ValueError(
            f"seed_{seed} signals INVALID for this sweep: {diagnostic}"
        )

    agreement_array = _load_agreement_array(signal_dir)
    if agreement_array is not None and agreement_array.size > 0:
        agreement_mean = float(np.mean(agreement_array))
        agreement_std = float(np.std(agreement_array))
    else:
        agreement_mean = None
        agreement_std = None

    pred_sha = _load_predicted_returns_sha(signal_dir)

    return SeedResult(
        seed=seed,
        test_h10_ic=test_h10_ic,
        test_h60_ic=test_h60_ic,
        test_h300_ic=test_h300_ic,
        compat_fp=compat_fp,
        experiment_provenance_hash=epH,
        model_config_hash=mch,
        agreement_mean=agreement_mean,
        agreement_std=agreement_std,
        predicted_returns_sha256=pred_sha,
        record_path=str(record_path),
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_gates(verdict: R20MultiSeedVerdict, seed_results: List[SeedResult]) -> Dict[str, Dict[str, Any]]:
    """Evaluate H1-H5 gates per pre-registered decision matrix."""
    gates: Dict[str, Dict[str, Any]] = {}

    # H1 PRIMARY — 4-band architectural lift
    gates["H1.a_floor"] = GateResult(
        name="H1.a_floor",
        threshold=H1A_FLOOR,
        observed=verdict.mean_h10_ic,
        passed=verdict.mean_h10_ic > H1A_FLOOR,
        notes=f"mean(test_h10_ic_seeds_43_47)={verdict.mean_h10_ic:.4f} > {H1A_FLOOR} (absolute floor)",
    ).__dict__
    gates["H1.b_ci_lower"] = GateResult(
        name="H1.b_ci_lower",
        threshold=H1B_CI_LOWER_FLOOR,
        observed=verdict.ci_low_h10_ic,
        passed=verdict.ci_low_h10_ic > H1B_CI_LOWER_FLOOR,
        notes=f"bootstrap_ci_lower={verdict.ci_low_h10_ic:.4f} > {H1B_CI_LOWER_FLOOR} (BINDING; {N_BOOTSTRAP} replicates, α={ALPHA})",
    ).__dict__
    gates["H1.c_partial_lift"] = GateResult(
        name="H1.c_partial_lift",
        threshold=H1C_PARTIAL_LIFT,
        observed=verdict.mean_h10_ic,
        passed=verdict.mean_h10_ic > H1C_PARTIAL_LIFT,
        notes=f"mean={verdict.mean_h10_ic:.4f} > {H1C_PARTIAL_LIFT} (TLOB Stage 2 - 5pp)",
    ).__dict__
    gates["H1.d_clear_lift"] = GateResult(
        name="H1.d_clear_lift",
        threshold=H1D_CLEAR_LIFT,
        observed=verdict.mean_h10_ic,
        passed=verdict.mean_h10_ic > H1D_CLEAR_LIFT,
        notes=f"mean={verdict.mean_h10_ic:.4f} > {H1D_CLEAR_LIFT} (above TLOB Stage 2)",
    ).__dict__

    # H2 MULTI-HORIZON
    gates["H2.a_h60"] = GateResult(
        name="H2.a_h60",
        threshold=H2A_H60_FLOOR,
        observed=verdict.mean_h60_ic,
        passed=verdict.mean_h60_ic > H2A_H60_FLOOR,
        notes=f"mean(test_h60_ic)={verdict.mean_h60_ic:.4f} > {H2A_H60_FLOOR} (Stage 6: {STAGE_6_H60_ANCHOR:.4f})",
    ).__dict__
    gates["H2.b_h300"] = GateResult(
        name="H2.b_h300",
        threshold=H2B_H300_FLOOR,
        observed=verdict.mean_h300_ic,
        passed=verdict.mean_h300_ic > H2B_H300_FLOOR,
        notes=f"mean(test_h300_ic)={verdict.mean_h300_ic:.4f} > {H2B_H300_FLOOR} (Stage 6: {STAGE_6_H300_ANCHOR:.4f})",
    ).__dict__
    gates["H2.c_sigma_ceiling"] = GateResult(
        name="H2.c_sigma_ceiling",
        threshold=H2C_SIGMA_CEILING,
        observed=verdict.std_h10_ic,
        passed=verdict.std_h10_ic < H2C_SIGMA_CEILING,
        notes=f"std(test_h10_ic)={verdict.std_h10_ic:.4f} < {H2C_SIGMA_CEILING} (seed-stability)",
    ).__dict__

    # H3 ARCHITECTURAL INVARIANT (Phase Y composer SEED-INVARIANT)
    all_compat_populated = all(r.compat_fp is not None for r in seed_results)
    gates["H3.a_compat_populated"] = GateResult(
        name="H3.a_compat_populated",
        threshold=None,
        observed=sum(1 for r in seed_results if r.compat_fp is not None),
        passed=all_compat_populated,
        notes=f"all 5 seeds populated compat_fp: {all_compat_populated}",
    ).__dict__

    compat_identical = len(verdict.distinct_compat_fps) == 1 if all_compat_populated else False
    gates["H3.b_compat_identical"] = GateResult(
        name="H3.b_compat_identical",
        threshold=None,
        observed=verdict.distinct_compat_fps,
        passed=compat_identical,
        notes=f"all 5 seeds share IDENTICAL compat_fp: {compat_identical}; distinct={verdict.distinct_compat_fps}",
    ).__dict__

    all_eph_populated = all(r.experiment_provenance_hash is not None for r in seed_results)
    gates["H3.c_eph_populated"] = GateResult(
        name="H3.c_eph_populated",
        threshold=None,
        observed=sum(1 for r in seed_results if r.experiment_provenance_hash is not None),
        passed=all_eph_populated,
        notes=f"all 5 seeds populated epH: {all_eph_populated}",
    ).__dict__

    eph_identical = len(verdict.distinct_epHs) == 1 if all_eph_populated else False
    gates["H3.d_eph_identical"] = GateResult(
        name="H3.d_eph_identical",
        threshold=None,
        observed=verdict.distinct_epHs,
        passed=eph_identical,
        notes=f"all 5 seeds share IDENTICAL epH (Phase Y composer SEED-INVARIANT): {eph_identical}; distinct={verdict.distinct_epHs}",
    ).__dict__

    all_mch_populated = all(r.model_config_hash is not None for r in seed_results)
    mch_identical = len(verdict.distinct_mch) == 1 if all_mch_populated else False
    gates["H3.e_mch_populated_identical"] = GateResult(
        name="H3.e_mch_populated_identical",
        threshold=None,
        observed=verdict.distinct_mch,
        passed=all_mch_populated and mch_identical,
        notes=f"all 5 seeds populated + IDENTICAL mch: populated={all_mch_populated}, identical={mch_identical}; distinct={verdict.distinct_mch}",
    ).__dict__

    gates["H3.f_pred_divergence"] = GateResult(
        name="H3.f_pred_divergence",
        threshold=H3F_MIN_DISTINCT_PRED_SHAS,
        observed=verdict.distinct_pred_shas,
        passed=verdict.distinct_pred_shas >= H3F_MIN_DISTINCT_PRED_SHAS,
        notes=f"distinct predicted_returns SHA-256: {verdict.distinct_pred_shas} (≥{H3F_MIN_DISTINCT_PRED_SHAS} RNG sanity)",
    ).__dict__

    # H4 CONFIRMATION MODULE
    all_agreement_emitted = all(r.agreement_mean is not None for r in seed_results)
    gates["H4.a_emitted"] = GateResult(
        name="H4.a_emitted",
        threshold=None,
        observed=sum(1 for r in seed_results if r.agreement_mean is not None),
        passed=all_agreement_emitted,
        notes=f"agreement_ratio.npy emitted for all 5 seeds: {all_agreement_emitted}",
    ).__dict__

    if all_agreement_emitted:
        means_in_band = all(
            H4_MEAN_LOW <= r.agreement_mean <= H4_MEAN_HIGH for r in seed_results
        )
        stds_above_floor = all(r.agreement_std > H4_STD_MIN for r in seed_results)
        gates["H4.b_mean_band"] = GateResult(
            name="H4.b_mean_band",
            threshold=None,
            observed=[round(r.agreement_mean, 4) for r in seed_results],
            passed=means_in_band,
            notes=f"all 5 mean(agreement) ∈ [{H4_MEAN_LOW}, {H4_MEAN_HIGH}]: {means_in_band}",
        ).__dict__
        gates["H4.c_std_min"] = GateResult(
            name="H4.c_std_min",
            threshold=H4_STD_MIN,
            observed=[round(r.agreement_std, 4) for r in seed_results],
            passed=stds_above_floor,
            notes=f"all 5 std(agreement) > {H4_STD_MIN}: {stds_above_floor}",
        ).__dict__

    return gates


def _classify_verdict(gates: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    """Apply pre-registered decision matrix."""

    def passed(gate_key: str) -> bool:
        return gates.get(gate_key, {}).get("passed", False)

    # ABORT — infrastructure regression
    h3_gates = [
        "H3.a_compat_populated",
        "H3.b_compat_identical",
        "H3.c_eph_populated",
        "H3.d_eph_identical",
        "H3.e_mch_populated_identical",
    ]
    h3_failures = [g for g in h3_gates if not passed(g)]
    if h3_failures:
        return (
            "ABORT",
            f"H3 architectural-invariant FAIL: {h3_failures}. Phase Y composer regression OR corpus identity broken. "
            f"Investigate BEFORE issuing scientific verdict.",
        )

    if not passed("H4.a_emitted"):
        return (
            "ABORT",
            "H4.a FAIL — agreement_ratio.npy NOT emitted by signal export for ≥1 seed. "
            "ConfirmationModule wire-up regression. Investigate BEFORE verdict.",
        )

    # REFUTE — architecture-axis closed
    if not passed("H1.a_floor"):
        return (
            "REFUTE",
            f"H1.a FAIL — mean(test_h10_ic) below floor {H1A_FLOOR}. HMHP-R cannot establish "
            f"predictive floor under seed perturbation. Close architecture-axis cleanly. "
            f"Pivot recommendations: R-21 reframed feature-axis OR TB-first-touch OR TIER 1 hygiene.",
        )

    if not passed("H1.b_ci_lower"):
        return (
            "REFUTE",
            f"H1.b FAIL — paired-bootstrap CI lower below {H1B_CI_LOWER_FLOOR}. Seed variance "
            f"too high to establish architectural-line claim. Close architecture-axis.",
        )

    # INDETERMINATE-DOWNGRADED-NEED-N10 — variance assumption falsified
    if not passed("H2.c_sigma_ceiling"):
        return (
            "INDETERMINATE-DOWNGRADED-NEED-N10",
            f"H2.c FAIL — std(test_h10_ic) > {H2C_SIGMA_CEILING}. N=5 power insufficient. "
            f"Manual follow-up cycle adds seeds 48-52 (N=10) for variance characterization.",
        )

    # GO-CLEAR-LIFT
    if (
        passed("H1.d_clear_lift")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and passed("H4.b_mean_band")
        and passed("H4.c_std_min")
    ):
        return (
            "GO-CLEAR-LIFT",
            "HMHP-R cascade ARCHITECTURALLY SUPERIOR to TLOB Stage 2 across 5 seeds. "
            "Close TLOB-direction. Next cycle: R-21 feature-axis OR R-22 cost-aware sweep.",
        )

    # GO-COMPETITIVE
    if (
        passed("H1.c_partial_lift")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and passed("H4.b_mean_band")
        and passed("H4.c_std_min")
    ):
        return (
            "GO-COMPETITIVE",
            "HMHP-R competitive within 5pp of TLOB Stage 2 across 5 seeds. "
            "Multi-seed variance characterized. Pivot decision: bundle with R-21 OR close architecture-axis.",
        )

    # PARTIAL-LIFT
    if (
        passed("H1.a_floor")
        and passed("H1.b_ci_lower")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and not passed("H1.c_partial_lift")
    ):
        return (
            "PARTIAL-LIFT",
            "HMHP-R signal at H10 above absolute floor; underperforms TLOB beyond 5pp band. "
            "Close TLOB direction; HMHP-R documented as cleaner-but-weaker.",
        )

    return (
        "INDETERMINATE",
        "Mixed gate outcomes — review per-gate detail and decide next direction manually.",
    )


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


def analyze(sweep_id: str, pipeline_root: Path, allow_partial: bool = False, min_grid_points: int = 5) -> R20MultiSeedVerdict:
    """Main analyzer entry point.

    Args:
        sweep_id: Sweep timestamp ID (e.g. "cycle12_r20_hmhp_r_multi_seed_20260519T2030")
        pipeline_root: Repository root path
        allow_partial: If True, accept fewer than 5 seeds completed (default False)
        min_grid_points: Minimum seeds required to proceed (default 5)

    Returns:
        Full verdict bundle
    """
    record_paths = _find_training_records(sweep_id, pipeline_root)
    n_found = len(record_paths)

    if n_found < min_grid_points:
        if not allow_partial:
            raise RuntimeError(
                f"Only {n_found} seed records found (need ≥{min_grid_points}). "
                f"Pass --allow-partial to proceed with partial results."
            )

    # Load per-seed results
    seed_results: List[SeedResult] = []
    for path in record_paths:
        try:
            seed_results.append(_load_seed_result(path, pipeline_root))
        except Exception as exc:
            print(f"WARN: failed to load {path}: {exc}", file=sys.stderr)

    if not seed_results:
        raise RuntimeError("No seed results loaded — cannot analyze.")

    seed_results.sort(key=lambda r: r.seed)
    n_seeds = len(seed_results)

    # Multi-seed aggregation
    h10_ics = np.array([r.test_h10_ic for r in seed_results], dtype=np.float64)
    h60_ics = np.array([r.test_h60_ic for r in seed_results], dtype=np.float64)
    h300_ics = np.array([r.test_h300_ic for r in seed_results], dtype=np.float64)

    mean_h10 = float(np.mean(h10_ics))
    std_h10 = float(np.std(h10_ics, ddof=1)) if n_seeds > 1 else 0.0
    mean_h60 = float(np.mean(h60_ics))
    mean_h300 = float(np.mean(h300_ics))

    # Paired-bootstrap CI on test_h10_ic — paired=False single-array mode
    # (#PY-255 closure 2026-05-15; hft_metrics v0.1.11 SSoT)
    # API: block_bootstrap_ci(statistic_fn, x, y=None, *, paired, n_bootstraps,
    #                          block_length, ci=0.95, seed)
    # Returns 4-tuple: (estimate, ci_lower, ci_upper, n_nonfinite_replaced)
    # block_length=1 because seeds are INDEPENDENT (no temporal correlation)
    try:
        _estimate, ci_low, ci_high, n_nonfinite = block_bootstrap_ci(
            lambda x: float(np.mean(x)),  # statistic_fn (positional FIRST)
            h10_ics,                       # x (positional SECOND)
            paired=False,
            n_bootstraps=N_BOOTSTRAP,
            block_length=BLOCK_LENGTH,
            ci=1.0 - ALPHA,                # ci=0.95 for α=0.05
            seed=42,
        )
    except Exception as exc:
        print(f"WARN: bootstrap CI failed: {exc}", file=sys.stderr)
        ci_low, ci_high, n_nonfinite = 0.0, 0.0, 0

    # H3 invariant: distinct counts
    distinct_compat = sorted({r.compat_fp for r in seed_results if r.compat_fp})
    distinct_eph = sorted({r.experiment_provenance_hash for r in seed_results if r.experiment_provenance_hash})
    distinct_mch = sorted({r.model_config_hash for r in seed_results if r.model_config_hash})
    distinct_pred_shas = len({r.predicted_returns_sha256 for r in seed_results if r.predicted_returns_sha256})

    verdict = R20MultiSeedVerdict(
        sweep_id=sweep_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_seeds=n_seeds,
        seed_results=[
            {
                "seed": r.seed,
                "test_h10_ic": r.test_h10_ic,
                "test_h60_ic": r.test_h60_ic,
                "test_h300_ic": r.test_h300_ic,
                "compat_fp": r.compat_fp,
                "experiment_provenance_hash": r.experiment_provenance_hash,
                "model_config_hash": r.model_config_hash,
                "agreement_mean": r.agreement_mean,
                "agreement_std": r.agreement_std,
                "predicted_returns_sha256": r.predicted_returns_sha256,
                "record_path": r.record_path,
            }
            for r in seed_results
        ],
        mean_h10_ic=mean_h10,
        std_h10_ic=std_h10,
        ci_low_h10_ic=ci_low,
        ci_high_h10_ic=ci_high,
        ci_n_nonfinite_replaced=n_nonfinite,
        mean_h60_ic=mean_h60,
        mean_h300_ic=mean_h300,
        distinct_compat_fps=distinct_compat,
        distinct_epHs=distinct_eph,
        distinct_mch=distinct_mch,
        distinct_pred_shas=distinct_pred_shas,
    )

    verdict.gates = _evaluate_gates(verdict, seed_results)
    verdict.verdict, verdict.verdict_summary = _classify_verdict(verdict.gates)

    return verdict


def _print_verdict(verdict: R20MultiSeedVerdict) -> None:
    """Human-readable verdict output."""
    print(f"\n{'=' * 78}")
    print(f"R-20 HMHP-R MULTI-SEED VERDICT — Sweep: {verdict.sweep_id}")
    print(f"Timestamp: {verdict.timestamp_utc}")
    print(f"N seeds: {verdict.n_seeds}")
    print(f"{'=' * 78}\n")

    print("--- Per-Seed Results ---")
    print(f"  {'Seed':<6} {'H10 IC':<10} {'H60 IC':<10} {'H300 IC':<10} {'agree_μ':<10} {'agree_σ':<10}")
    for r in verdict.seed_results:
        am = r['agreement_mean']
        asd = r['agreement_std']
        print(
            f"  {r['seed']:<6} {r['test_h10_ic']:<10.4f} {r['test_h60_ic']:<10.4f} "
            f"{r['test_h300_ic']:<10.4f} {am if am is None else f'{am:<10.4f}':<10} "
            f"{asd if asd is None else f'{asd:<10.4f}':<10}"
        )

    print("\n--- Multi-Seed Aggregation ---")
    print(f"  mean(test_h10_ic)  = {verdict.mean_h10_ic:.4f}  (TLOB Stage 2: {TLOB_STAGE_2_BASELINE:.4f}; Stage 6: {STAGE_6_H10_ANCHOR:.4f})")
    print(f"  std(test_h10_ic)   = {verdict.std_h10_ic:.4f}  (ceiling: {H2C_SIGMA_CEILING})")
    print(f"  bootstrap CI 95%   = [{verdict.ci_low_h10_ic:.4f}, {verdict.ci_high_h10_ic:.4f}]")
    print(f"  CI n_nonfinite     = {verdict.ci_n_nonfinite_replaced}")
    print(f"  mean(test_h60_ic)  = {verdict.mean_h60_ic:.4f}  (Stage 6: {STAGE_6_H60_ANCHOR:.4f})")
    print(f"  mean(test_h300_ic) = {verdict.mean_h300_ic:.4f}  (Stage 6: {STAGE_6_H300_ANCHOR:.4f})")

    print("\n--- Phase Y Trust Columns (SEED-INVARIANT expected) ---")
    print(f"  distinct compat_fp:   {len(verdict.distinct_compat_fps)}  {verdict.distinct_compat_fps[:1] if verdict.distinct_compat_fps else []}")
    print(f"  distinct epH:         {len(verdict.distinct_epHs)}  {verdict.distinct_epHs[:1] if verdict.distinct_epHs else []}")
    print(f"  distinct mch:         {len(verdict.distinct_mch)}  {verdict.distinct_mch[:1] if verdict.distinct_mch else []}")
    print(f"  distinct pred SHAs:   {verdict.distinct_pred_shas}  (≥{H3F_MIN_DISTINCT_PRED_SHAS} expected RNG sanity)")

    print("\n--- Gate Evaluation ---")
    for gate_name, gate_info in verdict.gates.items():
        status = "PASS" if gate_info["passed"] else "FAIL"
        print(f"  [{status}] {gate_name}: {gate_info['notes']}")

    print(f"\n{'=' * 78}")
    print(f"VERDICT: {verdict.verdict}")
    print(f"{'=' * 78}")
    print(f"{verdict.verdict_summary}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_id", help="Sweep ID, e.g. cycle12_r20_hmhp_r_multi_seed_20260519T...")
    parser.add_argument(
        "--pipeline-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Pipeline root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Verdict JSON output dir (default: hft-ops/ledger/r20_multi_seed_verdicts/)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow analysis with fewer than --min-grid-points seeds (default: require all 5)",
    )
    parser.add_argument(
        "--min-grid-points",
        type=int,
        default=5,
        help="Minimum seeds required for valid verdict (default: 5)",
    )
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root)

    try:
        verdict = analyze(
            args.sweep_id,
            pipeline_root,
            allow_partial=args.allow_partial,
            min_grid_points=args.min_grid_points,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR in analyze: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    _print_verdict(verdict)

    # Save verdict JSON via SSoT atomic_write_json
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = pipeline_root / "hft-ops" / "ledger" / "r20_multi_seed_verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{args.sweep_id}_verdict_{ts}.json"
    atomic_write_json(out_path, verdict.to_dict(), sort_keys=True)
    print(f"Verdict saved: {out_path}\n")

    # Exit code semantics
    if verdict.verdict.startswith("GO"):
        return 0
    if verdict.verdict in ("REFUTE", "PARTIAL-LIFT"):
        return 1
    if verdict.verdict.startswith("INDETERMINATE"):
        return 3
    if verdict.verdict == "ABORT":
        return 2
    return 4


if __name__ == "__main__":
    sys.exit(main())
