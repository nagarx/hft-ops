"""
Pairwise statistical comparison of sweep child records (Phase V.B.4b,
2026-04-21).

Adapter between the hft-ops ledger + the hft-metrics
``pairwise_paired_bootstrap_compare`` primitive (v0.1.5). Given a
``sweep_id``, loads each child record's signal export files
(``regression_labels.npy`` + ``predicted_returns.npy``), verifies paired
structure (all records produced against the same test-split labels), and
invokes the pairwise primitive to produce a table of pairwise deltas +
bootstrap CI + BH-corrected q-values.

Scope: MVP supports the single metric ``val_ic`` (Spearman rank
correlation at the primary horizon). Additional metrics + classification
support are deferred to a follow-up (keyed on researcher demand —
extending the ``_STATISTIC_FN_REGISTRY`` dispatch at the bottom of this
module).

Paired vs unpaired:
    All child records must share byte-identical ``regression_labels.npy``
    content at the chosen horizon — same test split, same FeatureSet,
    same extraction. This is the case for sweeps varying training-time
    treatments (``seed_stability``, ``loss_ablation``,
    ``model_family_classification``). Sweeps that vary the EXTRACTION
    (``horizon_sensitivity`` — bins/horizons differ) or the DATA
    (``feature_set_ablation`` — samples differ per FeatureSet) produce
    non-paired results; the adapter raises ``ValueError`` and points the
    caller to Phase VI unpaired tooling.

Module placement:
    ``statistical_compare.py`` is separated from ``comparator.py``
    (descriptive ranking / diffing) per hft-rules §4 one-module-one-
    responsibility. This module owns numpy + hft-metrics consumption +
    signal-file I/O; ``comparator.py`` stays pure-Python dict manipulation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hft_metrics.ic import spearman_ic
from hft_metrics.pairwise import (
    PairwiseResult,
    pairwise_paired_bootstrap_compare,
)

from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


# =============================================================================
# Metric dispatch
# =============================================================================

def _spearman_ic_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation — returns the statistic scalar only
    (discards the p-value from ``spearman_ic``'s 2-tuple return)."""
    rho, _ = spearman_ic(x, y)
    return float(rho)


_STATISTIC_FN_REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "val_ic": _spearman_ic_statistic,
}
"""Dispatch from ``--metric`` CLI flag to a ``(x, y) -> float`` callable.

MVP supports ``val_ic`` (Spearman). Extension path — add new keys here plus
a CLI flag help-text update. Keeps adapter pure — the caller's metric
choice is explicit, never inferred from the record."""


# =============================================================================
# Paired-signal loading
# =============================================================================


@dataclass(frozen=True)
class _RecordSignals:
    """Internal aggregate of per-record signal data, loaded once per record."""
    experiment_id: str
    label: str                    # human-readable (from axis_values or name)
    primary_horizon_idx: int      # from signal_metadata.json
    regression_labels_full: np.ndarray  # [N, H] float64 bps
    predicted_returns_full: np.ndarray  # [N, H] float64 bps
    regression_labels_sha256: str       # for paired-check


def _resolve_signal_dir(
    record_entry: Dict[str, Any],
    ledger: Any,                  # typing as Any to avoid circular import
    paths: PipelinePaths,
) -> Path:
    """Resolve a record's signal-export output directory.

    The index entry (lightweight) does NOT carry ``manifest_path``, so we
    load the full ``ExperimentRecord`` via ``ledger.get(experiment_id)``
    and then re-parse the manifest to get the signal-export output_dir
    (which may reference pipeline variables).

    Args:
        record_entry: ledger index entry (from ``ledger.filter(...)``).
        ledger: ``hft_ops.ledger.ledger.Ledger`` instance for record loading.
        paths: ``PipelinePaths`` for resolving relative paths to absolute.

    Returns:
        Absolute path to the directory containing
        ``predicted_returns.npy`` / ``regression_labels.npy`` /
        ``signal_metadata.json``.

    Raises:
        ValueError: if the record or manifest cannot be located.
    """
    exp_id = record_entry.get("experiment_id")
    full_record = ledger.get(exp_id)
    if full_record is None:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' not found in ledger "
            f"(index entry present but records/{exp_id}.json missing — stale "
            f"index? run `hft-ops ledger rebuild-index`)."
        )

    manifest_path = full_record.manifest_path
    if not manifest_path or not Path(manifest_path).exists():
        raise ValueError(
            f"compare_sweep_statistical: manifest for '{exp_id}' not at "
            f"{manifest_path!r}. Signal-dir cannot be resolved without the "
            f"manifest (post-Phase-8A sweep runs materialize manifests under "
            f"hft-ops/ledger/runs/<exp>_<ts>/; check that path still exists)."
        )

    manifest = load_manifest(manifest_path)
    signal_stage = manifest.stages.signal_export
    if not signal_stage.enabled:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' has "
            f"signal_export disabled — no signal files to compare. "
            f"Re-run with signal_export enabled or exclude from sweep."
        )
    if not signal_stage.output_dir:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' signal_export "
            f"stage has no output_dir set."
        )
    return paths.resolve(signal_stage.output_dir)


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 (8KB chunks) — used for the paired-labels check."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_record_signals(
    record_entry: Dict[str, Any],
    ledger: Any,
    paths: PipelinePaths,
) -> _RecordSignals:
    """Load one record's paired signal data + metadata.

    Reads:
      - ``<signal_dir>/regression_labels.npy`` → ground truth (x).
      - ``<signal_dir>/predicted_returns.npy`` → treatment predictions.
      - ``<signal_dir>/signal_metadata.json`` → primary_horizon_idx lookup.

    Primary-horizon resolution precedence (matches ``SignalManifest.validate``
    contract):
      1. Top-level ``primary_horizon_idx``.
      2. ``compatibility.primary_horizon_idx`` (nested under Phase II block).
      3. Default 0 (first horizon) with WARNING-level log.

    Raises:
        ValueError: if any of the required files are missing or the primary
            horizon cannot be determined.
    """
    sig_dir = _resolve_signal_dir(record_entry, ledger, paths)
    exp_id = record_entry.get("experiment_id", "<unknown>")

    labels_path = sig_dir / "regression_labels.npy"
    preds_path = sig_dir / "predicted_returns.npy"
    meta_path = sig_dir / "signal_metadata.json"

    for required in (labels_path, preds_path, meta_path):
        if not required.exists():
            raise ValueError(
                f"compare_sweep_statistical: record '{exp_id}' missing "
                f"required signal file: {required}. Expected a complete "
                f"regression signal export (regression_labels.npy + "
                f"predicted_returns.npy + signal_metadata.json). "
                f"Classification signals (predictions.npy + labels.npy) are "
                f"not yet supported by this MVP adapter."
            )

    labels_arr = np.load(labels_path)
    preds_arr = np.load(preds_path)

    if labels_arr.shape != preds_arr.shape:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' shape mismatch: "
            f"regression_labels={labels_arr.shape} vs "
            f"predicted_returns={preds_arr.shape}. Signal export is broken."
        )
    if labels_arr.ndim not in (1, 2):
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' regression_labels "
            f"has unexpected ndim={labels_arr.ndim}; expected 1 (single-horizon) "
            f"or 2 (multi-horizon)."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    # Primary horizon resolution (precedence per SignalManifest contract)
    primary_h = meta.get("primary_horizon_idx")
    if primary_h is None and isinstance(meta.get("compatibility"), dict):
        primary_h = meta["compatibility"].get("primary_horizon_idx")
    if primary_h is None:
        # Default to 0 — single-horizon signals collapse to 0 anyway;
        # multi-horizon without primary_horizon_idx is a legacy manifest.
        primary_h = 0
    if not isinstance(primary_h, int) or primary_h < 0:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' invalid "
            f"primary_horizon_idx={primary_h!r} in signal_metadata.json."
        )
    if labels_arr.ndim == 2 and primary_h >= labels_arr.shape[1]:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' "
            f"primary_horizon_idx={primary_h} out of bounds for labels shape "
            f"{labels_arr.shape}."
        )

    # Label for the treatment: prefer axis_values (sweep context), fall back to name
    axis_values = record_entry.get("axis_values", {}) or {}
    if axis_values:
        label = "__".join(f"{k}={v}" for k, v in sorted(axis_values.items()))
    else:
        label = record_entry.get("name", exp_id)

    return _RecordSignals(
        experiment_id=exp_id,
        label=label,
        primary_horizon_idx=primary_h,
        regression_labels_full=labels_arr,
        predicted_returns_full=preds_arr,
        regression_labels_sha256=_sha256_file(labels_path),
    )


def _assert_paired_labels(loaded: List[_RecordSignals]) -> None:
    """Verify every record shares byte-identical regression_labels.npy.

    This is the defining invariant of paired bootstrap. If violated, the
    pairwise primitive would silently produce nonsense (comparing
    treatment IC on DIFFERENT test samples — not a meaningful pairwise
    comparison). Fail-loud per hft-rules §8.
    """
    if len(loaded) < 2:
        return  # nothing to pair-check with 0 or 1 record
    first_sha = loaded[0].regression_labels_sha256
    first_h = loaded[0].primary_horizon_idx
    for r in loaded[1:]:
        if r.regression_labels_sha256 != first_sha:
            raise ValueError(
                f"compare_sweep_statistical: unpaired labels detected — "
                f"records '{loaded[0].experiment_id}' and '{r.experiment_id}' "
                f"have different regression_labels.npy (SHA-256 differ). "
                f"Paired bootstrap requires all treatments share the same "
                f"test-split ground truth. This sweep appears to vary the "
                f"extraction or FeatureSet (e.g., horizon_sensitivity, "
                f"feature_set_ablation). Unpaired comparison requires "
                f"Welch-style tooling — deferred to Phase VI."
            )
        if r.primary_horizon_idx != first_h:
            raise ValueError(
                f"compare_sweep_statistical: primary_horizon_idx differs "
                f"across records "
                f"('{loaded[0].experiment_id}' has {first_h}, "
                f"'{r.experiment_id}' has {r.primary_horizon_idx}). "
                f"Paired comparison requires a single shared horizon."
            )


# =============================================================================
# Public API
# =============================================================================


def compare_sweep_statistical(
    sweep_child_entries: List[Dict[str, Any]],
    ledger: Any,
    paths: PipelinePaths,
    *,
    metric: str = "val_ic",
    alpha: float = 0.05,
    n_bootstraps: int = 10_000,
    block_length: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[PairwiseResult], List[str]]:
    """Paired pairwise-bootstrap comparison of a sweep's child records.

    Pipeline:
      1. Load each child record's signal files.
      2. Assert paired structure (identical regression_labels.npy +
         primary_horizon_idx across records).
      3. Stack ``x = regression_labels[:, h]`` (shared) and
         ``Y[:, k] = predicted_returns_k[:, h]``.
      4. Call ``hft_metrics.pairwise.pairwise_paired_bootstrap_compare``.

    Args:
        sweep_child_entries: Ledger index entries for the sweep's child
            experiments (i.e., ``record_type != "sweep_aggregate"``). The
            CLI is responsible for filtering these via
            ``ledger.filter(sweep_id=...)`` + excluding ``sweep_aggregate``.
        ledger: ``hft_ops.ledger.ledger.Ledger`` instance (used to load
            full records for manifest resolution).
        paths: ``PipelinePaths`` for resolving manifest + signal paths.
        metric: Metric name — dispatched via ``_STATISTIC_FN_REGISTRY``.
            MVP supports ``val_ic`` only.
        alpha: Significance level for CI + BH FDR. 0.05 → 95% CI.
        n_bootstraps: Bootstrap iterations. Default 10,000.
        block_length: Optional block length for moving-block bootstrap.
            Default (None) → ``ceil(n^(1/3))`` per Politis-Romano 1994.
        seed: Random seed for reproducibility. Default 42.

    Returns:
        Tuple ``(pairwise_results, treatment_labels)``:
          - ``pairwise_results``: ``List[PairwiseResult]`` of length
            ``K*(K-1)/2``, ordered lexicographically (i, j) with i < j.
          - ``treatment_labels``: ``List[str]`` of length K — human-readable
            label for each column (from ``axis_values`` or record name).

    Raises:
        ValueError: on any of: < 2 records; unknown metric; missing signal
            files; unpaired labels; primary_horizon mismatch.

    Example:
        >>> from hft_ops.ledger.ledger import Ledger
        >>> from hft_ops.paths import PipelinePaths
        >>> paths = PipelinePaths(pipeline_root="/path/to/monorepo")
        >>> ledger = Ledger(paths.ledger_dir)
        >>> entries = ledger.filter(sweep_id="seed_stability_20260421")
        >>> entries = [e for e in entries if e.get("record_type") != "sweep_aggregate"]
        >>> results, labels = compare_sweep_statistical(entries, ledger, paths)
        >>> for r in results:
        ...     print(f"{labels[r.i]} vs {labels[r.j]}: "
        ...           f"delta={r.delta:+.4f} "
        ...           f"CI=[{r.ci_lower:+.4f},{r.ci_upper:+.4f}] "
        ...           f"p_bh={r.p_value_bh:.3f}")
    """
    # === Input validation ===
    if metric not in _STATISTIC_FN_REGISTRY:
        raise ValueError(
            f"compare_sweep_statistical: unsupported metric {metric!r}. "
            f"Supported: {sorted(_STATISTIC_FN_REGISTRY)}. Extend "
            f"_STATISTIC_FN_REGISTRY in statistical_compare.py to add more."
        )
    if len(sweep_child_entries) < 2:
        raise ValueError(
            f"compare_sweep_statistical: need >= 2 child records; got "
            f"{len(sweep_child_entries)}. Nothing to compare."
        )

    statistic_fn = _STATISTIC_FN_REGISTRY[metric]

    # === Load per-record signals ===
    loaded = [
        _load_record_signals(entry, ledger, paths)
        for entry in sweep_child_entries
    ]

    # === Paired-structure check ===
    _assert_paired_labels(loaded)

    # === Stack x + Y at the shared primary horizon ===
    h = loaded[0].primary_horizon_idx
    if loaded[0].regression_labels_full.ndim == 1:
        x = loaded[0].regression_labels_full.astype(np.float64)
        Y = np.column_stack([
            r.predicted_returns_full.astype(np.float64)
            for r in loaded
        ])
    else:
        x = loaded[0].regression_labels_full[:, h].astype(np.float64)
        Y = np.column_stack([
            r.predicted_returns_full[:, h].astype(np.float64)
            for r in loaded
        ])

    # Defense-in-depth: paired bootstrap requires clean paired arrays.
    # The primitive ALSO checks this but we want a more specific error.
    # Filter rows with any NaN/Inf in either x or Y, logging the count.
    mask = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    if not mask.all():
        dropped = int((~mask).sum())
        total = int(len(x))
        # Drop non-finite rows preserving pairing (same mask across x + every Y col).
        x = x[mask]
        Y = Y[mask]
        # Log via standard library — caller can pick this up or re-display via rich.
        import logging
        logging.getLogger(__name__).warning(
            "compare_sweep_statistical: dropped %d/%d rows with non-finite "
            "values (NaN/Inf) — preserves pairing across all treatments.",
            dropped, total,
        )

    # === Delegate to hft-metrics primitive ===
    results = pairwise_paired_bootstrap_compare(
        x=x,
        Y=Y,
        statistic_fn=statistic_fn,
        n_bootstraps=n_bootstraps,
        block_length=block_length,
        alpha=alpha,
        seed=seed,
    )

    treatment_labels = [r.label for r in loaded]
    return results, treatment_labels
