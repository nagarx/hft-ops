"""Analyze R-20 HMHP-R cascading-decoder architecture-axis test (Cycle 12).

Single-record analyzer evaluating H1-H5 gates per cycle12_r20_hmhp_r.yaml
pre-registered decision matrix. R-20 is a SINGLE-SEED architecture-axis test
(seed=42) on the v3p0 e5_timebased_60s corpus comparing HMHP-R cascading-
decoder against TLOB Stage 2 baseline (test_h10_ic=0.3747).

Pre-registered gates (per cycle12_r20_hmhp_r.yaml LOCKED PRE-RUN):

  H1 PRIMARY — Architectural lift at H10 vs TLOB Stage 2 baseline 0.3747:
    H1.a Floor: test_h10_ic > 0.30 (Stage 6 was 0.3561)
    H1.b PARTIAL-LIFT: test_h10_ic > 0.3247 (TLOB Stage 2 - 5pp)
    H1.c CLEAR-LIFT: test_h10_ic > 0.3747

  H2 BASELINE — Multi-horizon signal:
    H2.a test_h60_ic > 0.10 (Stage 6 was 0.1408)
    H2.b test_h300_ic > 0.05 (Stage 6 was 0.0820)

  H3 ARCHITECTURAL INVARIANT (Phase Y composer correctness):
    H3.a compatibility_fingerprint matches Stage 6 anchor cdd723ae5024b877...
    H3.b model_config_hash OBSERVATIONAL (NOT a gate)
    H3.c experiment_provenance_hash populated (non-null)

  H4 CONFIRMATION MODULE (HMHP-R unique feature):
    H4.a agreement_ratio.npy emitted at signal export
    H4.b mean(agreement_ratio) ∈ [0.4, 0.9] (non-degenerate)
    H4.c std(agreement_ratio) > 0.05 (non-constant)

  H5 COST GATE — Informational (NOT binding per Lesson #106):
    POST-HF-1 cost model OptRet @ deep_itm_1.4bps + WinRate

Decision matrix (LOCKED PRE-RUN):

  GO-CLEAR-LIFT: H1.c PASS + H2.a/b PASS + H3.a/c PASS + H4.a/b/c PASS
  GO-COMPETITIVE: H1.b PASS + H2.a/b PASS + H3.a/c PASS + H4.a/b/c PASS
  PARTIAL-LIFT: H1.a PASS + H2.a/b PASS + H3.a/c PASS, but H1.b FAIL
  REFUTE: H1.a FAIL (test_h10_ic ≤ 0.30)
  ABORT: H3.a FAIL OR H4.a FAIL

Inputs:
- Sweep ID (positional arg) — e.g., cycle12_r20_hmhp_r_20260519T...
- Training record: hft-ops/ledger/records/cycle12_r20_hmhp_r__seed_42_*.json
- Signal metadata: outputs/experiments/cycle12_r20_hmhp_r__seed_42/signals/test/signal_metadata.json
- Signal arrays: predicted_returns.npy + regression_labels.npy + agreement_ratio.npy
- Backtest record (optional): from sweep aggregate

Output:
- Human-readable verdict to stdout
- JSON verdict at hft-ops/ledger/r20_verdicts/<sweep_id>_verdict_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Reuse hft_contracts atomic-write SSoT per hft-rules §0
from hft_contracts.atomic_io import atomic_write_json


# ---------------------------------------------------------------------------
# Pre-registered constants (LOCKED before script runs per cycle12 manifest)
# ---------------------------------------------------------------------------

# H1 gate thresholds
H1_FLOOR = 0.30
H1_PARTIAL_LIFT_FLOOR = 0.3247
H1_CLEAR_LIFT_FLOOR = 0.3747
TLOB_STAGE_2_BASELINE = 0.3747

# H2 gate thresholds
H2_H60_FLOOR = 0.10
H2_H300_FLOOR = 0.05
STAGE_6_H60_ANCHOR = 0.1408
STAGE_6_H300_ANCHOR = 0.0820

# H3 corpus invariant — set-based anchor accepts ALL known schema-eras per Lesson L49
# Both anchors are VALID identifications of the same `e5_timebased_60s_v3p0 +
# smoothed_return + H=[10,60,300] + 60s bins + HYBRID norm` corpus; they differ
# because CompatibilityContract schema evolved between schema-eras (Stage 6
# 2026-05-05 pre-Phase-8D vs γ-1 LITE 2026-05-10+ post-Phase-8D experiment_recorder
# SSoT migration). H3.a passes if compat_fp matches ANY known anchor.
# Per L49: future schema-eras add ONE entry to this frozenset per cycle; when
# N≥5 promote to `hft_contracts.corpus_anchors` Class A SSoT.
STAGE_6_COMPAT_FP_ANCHOR = (
    "cdd723ae5024b877683ed55e55a30c49e882e77260156ddb69ea192e6c05998b"
)
CURRENT_CORPUS_COMPAT_FP_ANCHOR = (
    "0ccd9f90bca06c868607b6520653e195d909a7fe6083a7aa29e7b8e02c2be160"
)
CORPUS_COMPAT_FP_ANCHORS = frozenset({
    STAGE_6_COMPAT_FP_ANCHOR,           # Stage 6 schema-era (2026-05-05)
    CURRENT_CORPUS_COMPAT_FP_ANCHOR,    # γ-1 LITE schema-era (2026-05-10+)
})

# H4 ConfirmationModule sanity gates
H4_MEAN_LOW = 0.4
H4_MEAN_HIGH = 0.9
H4_STD_MIN = 0.05


@dataclass(frozen=True)
class GateResult:
    """Single-gate evaluation outcome."""

    name: str
    threshold: Optional[float]
    actual: Any
    passed: bool
    notes: str = ""


@dataclass
class R20Verdict:
    """Full R-20 verdict bundle."""

    sweep_id: str
    timestamp_utc: str
    training_record_path: str
    signal_metadata_path: str
    test_h10_ic: float
    test_h60_ic: float
    test_h300_ic: float
    compat_fp: Optional[str]
    model_config_hash: Optional[str]
    experiment_provenance_hash: Optional[str]
    agreement_mean: Optional[float]
    agreement_std: Optional[float]
    backtest_opt_ret: Optional[float]
    backtest_win_rate: Optional[float]
    gates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verdict: str = ""
    verdict_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "timestamp_utc": self.timestamp_utc,
            "training_record_path": self.training_record_path,
            "signal_metadata_path": self.signal_metadata_path,
            "test_h10_ic": self.test_h10_ic,
            "test_h60_ic": self.test_h60_ic,
            "test_h300_ic": self.test_h300_ic,
            "compat_fp": self.compat_fp,
            "model_config_hash": self.model_config_hash,
            "experiment_provenance_hash": self.experiment_provenance_hash,
            "agreement_mean": self.agreement_mean,
            "agreement_std": self.agreement_std,
            "backtest_opt_ret": self.backtest_opt_ret,
            "backtest_win_rate": self.backtest_win_rate,
            "gates": self.gates,
            "verdict": self.verdict,
            "verdict_summary": self.verdict_summary,
        }


def _find_training_record(sweep_id: str, pipeline_root: Path) -> Path:
    """Locate the cycle12 seed_42 training record JSON for given sweep."""
    records_dir = pipeline_root / "hft-ops" / "ledger" / "records"
    if not records_dir.exists():
        raise FileNotFoundError(f"records dir not found: {records_dir}")
    # cycle12 records look like cycle12_r20_hmhp_r__seed_42_<timestamp>_<fingerprint>.json
    matches = sorted(records_dir.glob("cycle12_r20_hmhp_r__seed_42_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No cycle12_r20_hmhp_r__seed_42 record under {records_dir}"
        )
    # Most recent if multiple (re-runs)
    return matches[-1]


def _find_signal_metadata(pipeline_root: Path, record: Dict[str, Any]) -> Path:
    """Locate cycle12 signal metadata JSON.

    Tries (in order):
    1. `signal_export_output_dir` field from training record (canonical; Phase V.1 L1.2)
    2. Conventional path `outputs/experiments/cycle12_r20_hmhp_r__seed_42/signals/test/`
    3. Bare seed-only path `outputs/experiments/seed_42/signals/test/` (output_dir
       template bug — systemic across cycle5-cycle10 sweep manifests per CLAUDE.md
       banner anti-drift item 10)
    """
    # Path 1: training-record-provided signal_export_output_dir (most reliable)
    sig_dir_str = record.get("signal_export_output_dir")
    if sig_dir_str:
        sig_dir = Path(sig_dir_str)
        if sig_dir.is_absolute():
            metadata_path = sig_dir / "signal_metadata.json"
        else:
            metadata_path = pipeline_root / sig_dir / "signal_metadata.json"
        if metadata_path.exists():
            return metadata_path

    # Path 2: conventional sweep-point name
    metadata_path = (
        pipeline_root
        / "outputs"
        / "experiments"
        / "cycle12_r20_hmhp_r__seed_42"
        / "signals"
        / "test"
        / "signal_metadata.json"
    )
    if metadata_path.exists():
        return metadata_path

    # Path 3: bare seed-only (output_dir bug fallback)
    metadata_path = (
        pipeline_root
        / "outputs"
        / "experiments"
        / "seed_42"
        / "signals"
        / "test"
        / "signal_metadata.json"
    )
    if metadata_path.exists():
        return metadata_path

    raise FileNotFoundError(
        f"signal_metadata.json not found at any expected path. Tried: "
        f"{sig_dir_str}, "
        f"outputs/experiments/cycle12_r20_hmhp_r__seed_42/signals/test/, "
        f"outputs/experiments/seed_42/signals/test/"
    )


def _load_agreement_array(signal_dir: Path) -> Optional[np.ndarray]:
    """Load agreement_ratio.npy if present (HMHP-R ConfirmationModule output)."""
    agree_path = signal_dir / "agreement_ratio.npy"
    if not agree_path.exists():
        return None
    return np.load(agree_path, allow_pickle=False)


def _evaluate_gates(verdict: R20Verdict) -> Dict[str, Dict[str, Any]]:
    """Evaluate H1-H5 gates per pre-registered decision matrix."""
    gates: Dict[str, Dict[str, Any]] = {}

    # H1 PRIMARY (3-band)
    gates["H1.a_floor"] = GateResult(
        "H1.a_floor",
        H1_FLOOR,
        verdict.test_h10_ic,
        verdict.test_h10_ic > H1_FLOOR,
        f"test_h10_ic={verdict.test_h10_ic:.4f} > {H1_FLOOR} (Stage 6 floor)",
    ).__dict__
    gates["H1.b_partial_lift"] = GateResult(
        "H1.b_partial_lift",
        H1_PARTIAL_LIFT_FLOOR,
        verdict.test_h10_ic,
        verdict.test_h10_ic > H1_PARTIAL_LIFT_FLOOR,
        f"test_h10_ic={verdict.test_h10_ic:.4f} > {H1_PARTIAL_LIFT_FLOOR} (TLOB - 5pp)",
    ).__dict__
    gates["H1.c_clear_lift"] = GateResult(
        "H1.c_clear_lift",
        H1_CLEAR_LIFT_FLOOR,
        verdict.test_h10_ic,
        verdict.test_h10_ic > H1_CLEAR_LIFT_FLOOR,
        f"test_h10_ic={verdict.test_h10_ic:.4f} > {H1_CLEAR_LIFT_FLOOR} (clear lift)",
    ).__dict__

    # H2 BASELINE (multi-horizon)
    gates["H2.a_h60"] = GateResult(
        "H2.a_h60",
        H2_H60_FLOOR,
        verdict.test_h60_ic,
        verdict.test_h60_ic > H2_H60_FLOOR,
        f"test_h60_ic={verdict.test_h60_ic:.4f} > {H2_H60_FLOOR}",
    ).__dict__
    gates["H2.b_h300"] = GateResult(
        "H2.b_h300",
        H2_H300_FLOOR,
        verdict.test_h300_ic,
        verdict.test_h300_ic > H2_H300_FLOOR,
        f"test_h300_ic={verdict.test_h300_ic:.4f} > {H2_H300_FLOOR}",
    ).__dict__

    # H3 ARCHITECTURAL INVARIANT — set-based anchor check (handles schema evolution)
    h3a_pass = verdict.compat_fp in CORPUS_COMPAT_FP_ANCHORS
    # Guard compat_fp[:16] indexing against None (Phase Y composer failure mode)
    compat_fp_display = verdict.compat_fp[:16] if verdict.compat_fp else "None"
    if h3a_pass:
        if verdict.compat_fp == STAGE_6_COMPAT_FP_ANCHOR:
            anchor_note = f"Stage 6 anchor (2026-05-05 era)"
        else:
            anchor_note = f"γ-1 LITE anchor (2026-05-10+ era; corpus IDENTITY preserved)"
        h3a_notes = f"compat_fp={compat_fp_display}... matches {anchor_note}"
    else:
        h3a_notes = (
            f"compat_fp={compat_fp_display}... matches NEITHER Stage 6 "
            f"({STAGE_6_COMPAT_FP_ANCHOR[:16]}...) NOR γ-1 LITE "
            f"({CURRENT_CORPUS_COMPAT_FP_ANCHOR[:16]}...) anchor — possible corpus regression"
        )
    gates["H3.a_compat_fp"] = GateResult(
        "H3.a_compat_fp",
        None,
        verdict.compat_fp,
        h3a_pass,
        h3a_notes,
    ).__dict__
    gates["H3.b_mch_observational"] = {
        "name": "H3.b_mch_observational",
        "threshold": None,
        "actual": verdict.model_config_hash,
        "passed": True,  # always passes — OBSERVATIONAL not gate
        "notes": f"mch={verdict.model_config_hash}; OBSERVATIONAL not gate per design",
    }
    gates["H3.c_eph_populated"] = GateResult(
        "H3.c_eph_populated",
        None,
        verdict.experiment_provenance_hash,
        verdict.experiment_provenance_hash is not None,
        f"epH={verdict.experiment_provenance_hash}",
    ).__dict__

    # H4 CONFIRMATION MODULE
    h4a_pass = verdict.agreement_mean is not None
    gates["H4.a_emitted"] = {
        "name": "H4.a_emitted",
        "threshold": None,
        "actual": h4a_pass,
        "passed": h4a_pass,
        "notes": "agreement_ratio.npy file exists" if h4a_pass else "agreement_ratio.npy MISSING",
    }
    if verdict.agreement_mean is not None:
        gates["H4.b_mean_band"] = GateResult(
            "H4.b_mean_band",
            None,
            verdict.agreement_mean,
            H4_MEAN_LOW <= verdict.agreement_mean <= H4_MEAN_HIGH,
            f"mean(agreement)={verdict.agreement_mean:.4f} ∈ [{H4_MEAN_LOW}, {H4_MEAN_HIGH}]",
        ).__dict__
    if verdict.agreement_std is not None:
        gates["H4.c_std_min"] = GateResult(
            "H4.c_std_min",
            H4_STD_MIN,
            verdict.agreement_std,
            verdict.agreement_std > H4_STD_MIN,
            f"std(agreement)={verdict.agreement_std:.4f} > {H4_STD_MIN}",
        ).__dict__

    return gates


def _classify_verdict(gates: Dict[str, Dict[str, Any]]) -> tuple[str, str]:
    """Apply pre-registered decision matrix."""

    def passed(gate_key: str) -> bool:
        return gates.get(gate_key, {}).get("passed", False)

    # ABORT — infrastructure regression
    if not passed("H3.a_compat_fp"):
        return (
            "ABORT",
            "H3.a FAIL — compat_fp does NOT match Stage 6 anchor. Investigate corpus / window / normalization regression BEFORE issuing scientific verdict.",
        )
    if not passed("H4.a_emitted"):
        return (
            "ABORT",
            "H4.a FAIL — agreement_ratio.npy NOT emitted by signal export. ConfirmationModule wire-up regression. Investigate BEFORE verdict.",
        )

    # REFUTE — architecture-axis closed
    if not passed("H1.a_floor"):
        return (
            "REFUTE",
            "H1.a FAIL — HMHP-R cannot recover Stage 6 floor at production scale. Close architecture-axis cleanly. Pivot recommendations: R-21 reframed OR TIER 1 hygiene cluster.",
        )

    # GO-CLEAR-LIFT
    if (
        passed("H1.c_clear_lift")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and passed("H3.c_eph_populated")
        and passed("H4.b_mean_band")
        and passed("H4.c_std_min")
    ):
        return (
            "GO-CLEAR-LIFT",
            "HMHP-R cascade ARCHITECTURALLY SUPERIOR to TLOB on v3p0. Close TLOB-direction. Next cycle: HMHP-R multi-seed N=3 follow-up + R-21 axis tests.",
        )

    # GO-COMPETITIVE
    if (
        passed("H1.b_partial_lift")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and passed("H3.c_eph_populated")
        and passed("H4.b_mean_band")
        and passed("H4.c_std_min")
    ):
        return (
            "GO-COMPETITIVE",
            "HMHP-R competitive within 5pp of TLOB Stage 2. Multi-seed N=3 follow-up triggered for variance characterization.",
        )

    # PARTIAL-LIFT
    if (
        passed("H1.a_floor")
        and passed("H2.a_h60")
        and passed("H2.b_h300")
        and passed("H3.c_eph_populated")
        and not passed("H1.b_partial_lift")
    ):
        return (
            "PARTIAL-LIFT",
            "HMHP-R produces signal at H10 but underperforms TLOB beyond 5pp band. Pivot decision deferred to multi-seed characterization.",
        )

    return (
        "INDETERMINATE",
        "Mixed gate outcomes — review per-gate detail and decide next direction manually.",
    )


def analyze(sweep_id: str, pipeline_root: Path) -> R20Verdict:
    """Main analyzer entry point."""
    # Locate training record first (contains signal_export_output_dir)
    training_record_path = _find_training_record(sweep_id, pipeline_root)
    record = json.loads(training_record_path.read_text())

    # Locate signal metadata using record-provided path + fallbacks
    signal_metadata_path = _find_signal_metadata(pipeline_root, record)
    signal_dir = signal_metadata_path.parent
    training_metrics = record.get("training_metrics", {}) or {}

    test_h10_ic = float(training_metrics.get("test_h10_ic", training_metrics.get("test_ic", 0.0)))
    test_h60_ic = float(training_metrics.get("test_h60_ic", 0.0))
    test_h300_ic = float(training_metrics.get("test_h300_ic", 0.0))

    # Trust columns
    compat_fp = record.get("compatibility_fingerprint")
    epH = record.get("experiment_provenance_hash")
    mch_top = record.get("model_config_hash")
    mch = mch_top if mch_top else (record.get("training_config") or {}).get("model_config_hash")

    # Load agreement_ratio.npy if HMHP-R ConfirmationModule emitted
    agreement_array = _load_agreement_array(signal_dir)
    if agreement_array is not None and agreement_array.size > 0:
        agreement_mean = float(np.mean(agreement_array))
        agreement_std = float(np.std(agreement_array))
    else:
        agreement_mean = None
        agreement_std = None

    # Backtest results (read from sweep aggregate if present; informational only)
    backtest_opt_ret = None
    backtest_win_rate = None
    backtest_metrics = record.get("backtest_metrics") or {}
    if backtest_metrics:
        # Look for deep_itm_1.4bps best OptRet
        for key, val in backtest_metrics.items():
            if isinstance(val, dict) and "deep_itm" in key.lower():
                backtest_opt_ret = val.get("opt_ret")
                backtest_win_rate = val.get("win_rate")
                break

    verdict = R20Verdict(
        sweep_id=sweep_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        training_record_path=str(training_record_path),
        signal_metadata_path=str(signal_metadata_path),
        test_h10_ic=test_h10_ic,
        test_h60_ic=test_h60_ic,
        test_h300_ic=test_h300_ic,
        compat_fp=compat_fp,
        model_config_hash=mch,
        experiment_provenance_hash=epH,
        agreement_mean=agreement_mean,
        agreement_std=agreement_std,
        backtest_opt_ret=backtest_opt_ret,
        backtest_win_rate=backtest_win_rate,
    )

    # Evaluate gates + classify
    verdict.gates = _evaluate_gates(verdict)
    verdict.verdict, verdict.verdict_summary = _classify_verdict(verdict.gates)

    return verdict


def _print_verdict(verdict: R20Verdict) -> None:
    """Human-readable verdict output."""
    print(f"\n{'=' * 72}")
    print(f"R-20 HMHP-R VERDICT — Sweep: {verdict.sweep_id}")
    print(f"Timestamp: {verdict.timestamp_utc}")
    print(f"{'=' * 72}\n")

    print(f"Training record: {verdict.training_record_path}")
    print(f"Signal metadata: {verdict.signal_metadata_path}\n")

    print("--- Test Metrics ---")
    print(f"  test_h10_ic  = {verdict.test_h10_ic:.4f}  (TLOB Stage 2: {TLOB_STAGE_2_BASELINE:.4f}; Stage 6: 0.3561)")
    print(f"  test_h60_ic  = {verdict.test_h60_ic:.4f}  (Stage 6: {STAGE_6_H60_ANCHOR:.4f})")
    print(f"  test_h300_ic = {verdict.test_h300_ic:.4f}  (Stage 6: {STAGE_6_H300_ANCHOR:.4f})\n")

    print("--- Phase Y Trust Columns ---")
    print(f"  compat_fp                  = {verdict.compat_fp}")
    print(f"  model_config_hash          = {verdict.model_config_hash}")
    print(f"  experiment_provenance_hash = {verdict.experiment_provenance_hash}\n")

    print("--- ConfirmationModule (HMHP-R) ---")
    if verdict.agreement_mean is not None:
        print(f"  mean(agreement_ratio) = {verdict.agreement_mean:.4f}  [{H4_MEAN_LOW}, {H4_MEAN_HIGH}]")
        print(f"  std(agreement_ratio)  = {verdict.agreement_std:.4f}  (> {H4_STD_MIN})\n")
    else:
        print(f"  agreement_ratio.npy: NOT EMITTED\n")

    if verdict.backtest_opt_ret is not None:
        print("--- Backtest (POST-HF-1 Deep ITM 1.4bps; informational) ---")
        print(f"  OptRet   = {verdict.backtest_opt_ret:+.2%}")
        print(f"  WinRate  = {verdict.backtest_win_rate:.2%}\n")

    print("--- Gate Evaluation ---")
    for gate_name, gate_info in verdict.gates.items():
        status = "PASS" if gate_info["passed"] else "FAIL"
        print(f"  [{status}] {gate_name}: {gate_info['notes']}")
    print()

    print(f"{'=' * 72}")
    print(f"VERDICT: {verdict.verdict}")
    print(f"{'=' * 72}")
    print(f"{verdict.verdict_summary}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_id", help="Sweep ID, e.g. cycle12_r20_hmhp_r_20260519T...")
    parser.add_argument(
        "--pipeline-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Pipeline root directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Verdict JSON output dir (default: hft-ops/ledger/r20_verdicts/)",
    )
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root)

    try:
        verdict = analyze(args.sweep_id, pipeline_root)
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
        out_dir = pipeline_root / "hft-ops" / "ledger" / "r20_verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{args.sweep_id}_verdict_{ts}.json"
    atomic_write_json(out_path, verdict.to_dict())
    print(f"Verdict JSON: {out_path}\n")

    # Exit code maps to verdict:
    #   GO-CLEAR-LIFT / GO-COMPETITIVE → 0
    #   PARTIAL-LIFT / INDETERMINATE → 1
    #   REFUTE → 2
    #   ABORT → 3
    exit_map = {
        "GO-CLEAR-LIFT": 0,
        "GO-COMPETITIVE": 0,
        "PARTIAL-LIFT": 1,
        "INDETERMINATE": 1,
        "REFUTE": 2,
        "ABORT": 3,
    }
    return exit_map.get(verdict.verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
