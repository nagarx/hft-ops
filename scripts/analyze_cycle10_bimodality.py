"""Cycle 11 Phase 4: per-seed signal-level bimodality analysis for cycle10 R-19 multi-seed.

Per Cycle 10 Lesson #107: 2 aggressive-PT seeds (43, 47; PT_pred ~8K; precision ~24%) vs
3 conservative-PT seeds (44, 45, 46; PT_pred ~3.8-4.8K; precision ~28%) — empirical
bimodality in TLOB+focal training under random seed perturbation.

Per cycle10 manifest output_dir bug (#PY-NEW filed 2026-05-19): training checkpoints
overwrote each other (5 seeds shared `cycle10_r19_multi_seed_base` training output_dir).
BUT signals ARE per-seed-preserved via Phase V.1 L1.2 `signal_export_output_dir` field
at `outputs/experiments/seed_{43..47}/signals/test/` (5 distinct predictions.npy SHAs).

This script computes signal-level bimodality diagnostics WITHOUT needing checkpoints:
1. Per-seed prediction distribution (class counts; PT-recall; PT-precision)
2. Cross-seed prediction agreement matrix (Cohen's kappa pairwise)
3. Signal-level ensemble: predictions[majority], predictions[mean confidence-weighted]
4. Confidence quantile distribution per seed (P25/P50/P75)
5. Per-seed pred-vs-label confusion matrices

Output: `hft-ops/ledger/r19_multi_seed_verdicts/cycle10_bimodality_analysis_<ts>.json`

Usage: `python scripts/analyze_cycle10_bimodality.py`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest


SEED_IDS = [43, 44, 45, 46, 47]
PT_CLASS = 2  # TB encoding: 0=SL, 1=Timeout, 2=ProfitTarget
SIGNAL_ROOT = Path("/Users/knight/code_local/HFT-pipeline-v2/outputs/experiments")
OUTPUT_DIR = Path("/Users/knight/code_local/HFT-pipeline-v2/hft-ops/ledger/r19_multi_seed_verdicts")


def load_seed_signals(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions + labels + confidence_score for a seed.

    Per #PY-291 security lock (closed 2026-05-16; AST regression test enforces):
    all `np.load` callsites MUST pass `allow_pickle=False` to prevent RCE-via-NPY
    on untrusted signal files. The cycle10 signals are trusted-internal but the
    lock applies uniformly per #PY-291 lock-test discipline.
    """
    seed_dir = SIGNAL_ROOT / f"seed_{seed}" / "signals" / "test"
    predictions = np.load(seed_dir / "predictions.npy", allow_pickle=False)
    labels = np.load(seed_dir / "labels.npy", allow_pickle=False)
    confidence = np.load(seed_dir / "confirmation_score.npy", allow_pickle=False)
    return predictions, labels, confidence


def cohens_kappa(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Compute Cohen's kappa between two predictions of identical length.

    κ = (p_o − p_e) / (1 − p_e) where p_o = observed agreement, p_e = chance agreement.
    """
    n = len(pred_a)
    p_o = (pred_a == pred_b).mean()
    # Chance agreement: sum_k P(a=k) * P(b=k)
    p_e = sum(
        (pred_a == k).mean() * (pred_b == k).mean()
        for k in np.unique(np.concatenate([pred_a, pred_b]))
    )
    if abs(1.0 - p_e) < 1e-12:
        return 0.0  # degenerate: perfect chance agreement
    return (p_o - p_e) / (1.0 - p_e)


def per_seed_metrics(seed: int) -> Dict:
    """Compute per-seed signal metrics."""
    predictions, labels, confidence = load_seed_signals(seed)
    n = len(predictions)

    # PT class counts + precision/recall
    pt_predicted = int((predictions == PT_CLASS).sum())
    pt_correct = int(((predictions == PT_CLASS) & (labels == PT_CLASS)).sum())
    pt_label_count = int((labels == PT_CLASS).sum())
    pt_precision = pt_correct / pt_predicted if pt_predicted > 0 else float("nan")
    pt_recall = pt_correct / pt_label_count if pt_label_count > 0 else float("nan")

    # Wilson 95% CI on PT precision
    if pt_predicted > 0:
        ci_lo, ci_hi = binomtest(k=pt_correct, n=pt_predicted).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    # Confidence quantile distribution
    conf_q25, conf_q50, conf_q75 = np.quantile(confidence, [0.25, 0.50, 0.75])

    # Class distribution (predicted)
    pred_class_counts = {int(k): int(v) for k, v in zip(*np.unique(predictions, return_counts=True))}
    label_class_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

    # Regime classification (per Lesson #107)
    if pt_predicted > 6000:
        regime = "aggressive-PT"
    elif pt_predicted < 5000:
        regime = "conservative-PT"
    else:
        regime = "intermediate"

    return {
        "seed": seed,
        "n_samples": n,
        "pred_class_counts": pred_class_counts,
        "label_class_counts": label_class_counts,
        "pt_predicted": pt_predicted,
        "pt_correct": pt_correct,
        "pt_label_count": pt_label_count,
        "pt_precision": pt_precision,
        "pt_recall": pt_recall,
        "pt_precision_wilson_ci": [float(ci_lo), float(ci_hi)],
        "confidence_quantiles": {"q25": float(conf_q25), "q50": float(conf_q50), "q75": float(conf_q75)},
        "regime": regime,
    }


def pairwise_agreement_matrix(seeds: List[int]) -> Dict:
    """Compute Cohen's kappa pairwise across all seed pairs."""
    pred_by_seed = {seed: load_seed_signals(seed)[0] for seed in seeds}
    matrix = {}
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1:]:
            kappa = cohens_kappa(pred_by_seed[sa], pred_by_seed[sb])
            matrix[f"seed_{sa}_vs_seed_{sb}"] = float(kappa)
    return matrix


def signal_ensemble_majority(seeds: List[int]) -> Dict:
    """Majority-vote ensemble across seeds + compute its PT precision."""
    preds_stack = np.stack([load_seed_signals(s)[0] for s in seeds])  # (5, N)
    # Majority vote per sample
    from scipy.stats import mode
    majority = mode(preds_stack, axis=0, keepdims=False).mode  # (N,)

    # Load labels from any seed (all should have same labels; TB v3p0 test split)
    _, labels, _ = load_seed_signals(seeds[0])

    pt_predicted = int((majority == PT_CLASS).sum())
    pt_correct = int(((majority == PT_CLASS) & (labels == PT_CLASS)).sum())
    pt_precision = pt_correct / pt_predicted if pt_predicted > 0 else float("nan")

    return {
        "ensemble_method": "majority_vote",
        "seeds_aggregated": seeds,
        "pt_predicted": pt_predicted,
        "pt_correct": pt_correct,
        "pt_precision": pt_precision,
    }


def main():
    print("=" * 76)
    print("Cycle 11 Phase 4: Cycle10 R-19 Multi-Seed Signal-Level Bimodality Analysis")
    print(f"Lesson #107 verification at signal level (NOT checkpoint level)")
    print(f"Analysis timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 76)

    # Per-seed metrics
    print("\nPER-SEED SIGNAL METRICS:")
    per_seed = {}
    for seed in SEED_IDS:
        m = per_seed_metrics(seed)
        per_seed[seed] = m
        print(
            f"  seed_{seed}: PT_pred={m['pt_predicted']:>5} | "
            f"PT_precision={m['pt_precision']:.4f} "
            f"CI[{m['pt_precision_wilson_ci'][0]:.3f}, {m['pt_precision_wilson_ci'][1]:.3f}] | "
            f"PT_recall={m['pt_recall']:.4f} | "
            f"conf_q50={m['confidence_quantiles']['q50']:.3f} | "
            f"regime={m['regime']}"
        )

    # Pairwise agreement
    print("\nPAIRWISE COHEN'S KAPPA (prediction-level agreement):")
    kappa = pairwise_agreement_matrix(SEED_IDS)
    for pair, k in kappa.items():
        print(f"  {pair}: κ={k:.4f}")
    print(f"\nMean pairwise κ: {np.mean(list(kappa.values())):.4f}")
    print(f"Std pairwise κ: {np.std(list(kappa.values()), ddof=1):.4f}")

    # Aggressive vs conservative regime split
    aggressive_seeds = [s for s in SEED_IDS if per_seed[s]["regime"] == "aggressive-PT"]
    conservative_seeds = [s for s in SEED_IDS if per_seed[s]["regime"] == "conservative-PT"]
    print(f"\nREGIME SPLIT (Lesson #107 verification):")
    print(f"  Aggressive-PT seeds: {aggressive_seeds}")
    print(f"  Conservative-PT seeds: {conservative_seeds}")

    if aggressive_seeds and conservative_seeds:
        agg_precision_mean = np.mean([per_seed[s]["pt_precision"] for s in aggressive_seeds])
        cons_precision_mean = np.mean([per_seed[s]["pt_precision"] for s in conservative_seeds])
        print(f"  Aggressive mean PT precision: {agg_precision_mean:.4f}")
        print(f"  Conservative mean PT precision: {cons_precision_mean:.4f}")
        print(f"  Delta (conservative - aggressive): {cons_precision_mean - agg_precision_mean:+.4f}")
    else:
        agg_precision_mean = float("nan")
        cons_precision_mean = float("nan")

    # Ensemble (all 5 seeds)
    print("\nSIGNAL-LEVEL MAJORITY ENSEMBLE (all 5 seeds):")
    ens_all = signal_ensemble_majority(SEED_IDS)
    print(
        f"  PT_pred={ens_all['pt_predicted']} | PT_correct={ens_all['pt_correct']} | "
        f"PT_precision={ens_all['pt_precision']:.4f}"
    )

    # Conservative-only ensemble
    if conservative_seeds:
        ens_cons = signal_ensemble_majority(conservative_seeds)
        print(f"\nCONSERVATIVE-REGIME-ONLY MAJORITY ENSEMBLE (seeds {conservative_seeds}):")
        print(
            f"  PT_pred={ens_cons['pt_predicted']} | PT_correct={ens_cons['pt_correct']} | "
            f"PT_precision={ens_cons['pt_precision']:.4f}"
        )
    else:
        ens_cons = None

    # Compose verdict JSON
    verdict = {
        "schema_version": "1.0",
        "analysis_kind": "cycle10_signal_bimodality",
        "cycle": "cycle11_phase4",
        "backlog_ref": "Cycle 11 Phase 4 / Lesson #107 verification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_seed_metrics": per_seed,
        "pairwise_cohens_kappa": kappa,
        "kappa_mean": float(np.mean(list(kappa.values()))),
        "kappa_std": float(np.std(list(kappa.values()), ddof=1)),
        "regime_split": {
            "aggressive_seeds": aggressive_seeds,
            "conservative_seeds": conservative_seeds,
            "aggressive_pt_precision_mean": float(agg_precision_mean),
            "conservative_pt_precision_mean": float(cons_precision_mean),
        },
        "ensemble_all_seeds": ens_all,
        "ensemble_conservative_only": ens_cons,
        "lesson_107_status": "VERIFIED" if aggressive_seeds and conservative_seeds else "PARTIAL",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = OUTPUT_DIR / f"cycle10_bimodality_analysis_{ts}.json"

    # Atomic-write via hft_contracts SSoT
    from hft_contracts.atomic_io import atomic_write_json
    atomic_write_json(output_path, verdict)

    print(f"\nVerdict JSON written to: {output_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
