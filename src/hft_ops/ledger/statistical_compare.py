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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hft_contracts.provenance import hash_file
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

    Resolution precedence (Phase V.1 L1.2 2026-04-21 — closes Agent 2 H1
    manifest-move-resilience gap):

    1. PREFERRED: ``record.signal_export_output_dir`` — absolute path
       captured at RUN TIME by ``SignalExportRunner.run`` + attached by
       ``cli.py::_record_experiment``. Available for all Phase-V.1-and-later
       runs. If the path still exists, use it directly — no manifest
       re-parse, no variable substitution, no fragility against monorepo
       moves or post-run manifest edits.

    2. FALLBACK: re-parse manifest via ``record.manifest_path`` and
       re-resolve ``stages.signal_export.output_dir`` through
       ``paths.resolve(...)``. Used only for pre-V.1.L1.2 records
       (``signal_export_output_dir=None``) — graceful back-compat.

    Args:
        record_entry: ledger index entry (from ``ledger.filter(...)``).
        ledger: ``hft_ops.ledger.ledger.Ledger`` instance for record loading.
        paths: ``PipelinePaths`` for resolving relative paths to absolute.

    Returns:
        Absolute path to the directory containing
        ``predicted_returns.npy`` / ``regression_labels.npy`` /
        ``signal_metadata.json``.

    Raises:
        ValueError: if the record cannot be located, or if BOTH the
            preferred and fallback paths fail to resolve (manifest missing
            AND signal_export_output_dir absent/non-existent).
    """
    exp_id = record_entry.get("experiment_id")
    full_record = ledger.get(exp_id)
    if full_record is None:
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' not found in ledger "
            f"(index entry present but records/{exp_id}.json missing — stale "
            f"index? run `hft-ops ledger rebuild-index`)."
        )

    # PREFERRED: run-time-captured signal_export_output_dir (V.1.L1.2+)
    stored_sig_dir = getattr(full_record, "signal_export_output_dir", None)
    if isinstance(stored_sig_dir, str) and stored_sig_dir:
        candidate = Path(stored_sig_dir)
        # V.1 Phase-V.2 audit Agent 3 Q5 defensive fix (2026-04-21): use
        # `is_dir()` not `exists()` — `exists()` returns True for regular
        # files too, which would make a mis-written `signal_export_output_dir`
        # pointing to a FILE (not a directory) pass through. Downstream
        # `_load_record_signals` would then try `sig_dir / "regression_labels.npy"`
        # producing an opaque nested-path FileNotFoundError rather than the
        # actionable resolver error. `is_dir()` is stricter: only a real
        # directory (or a symlink that resolves to a directory) passes.
        # Cheap hardening against manual-edit / future-refactor regression,
        # per hft-rules §8 "validate inputs at system boundaries."
        if candidate.is_dir():
            return candidate
        # Captured path exists on record but directory was removed (or is
        # a file, not a directory) — fall through to manifest re-resolution
        # in case the manifest points somewhere else (e.g., operator moved
        # outputs to archival storage).
        #
        # V.1.5 follow-up WARN (2026-04-23) — SDR-6 closure per hft-rules §8
        # "Never silently drop ... without recording diagnostics." Emit a
        # single RuntimeWarning per unique stored_sig_dir (Python's stdlib
        # warning filters dedup by (category, module, lineno) by default so
        # sweep compare over 100 records sharing the same moved-dir scenario
        # emits ONE warning, not 100). Helps the operator notice when the
        # monorepo output tree was moved, so they can decide whether to
        # restore, re-run, or update the manifest.
        import warnings
        reason = "is a file, not a directory" if candidate.exists() else "no longer exists"
        warnings.warn(
            f"resolve_signal_dir: record '{exp_id}' stored "
            f"signal_export_output_dir={stored_sig_dir!r} {reason}. "
            f"Falling back to manifest re-resolution. "
            f"If output tree was moved intentionally, this warning is safe; "
            f"otherwise restore the directory or re-run the experiment.",
            RuntimeWarning,
            stacklevel=2,
        )

    # FALLBACK: re-parse manifest (pre-V.1.L1.2 records, or when the
    # captured path has been moved/deleted).
    manifest_path = full_record.manifest_path
    if not manifest_path or not Path(manifest_path).exists():
        raise ValueError(
            f"compare_sweep_statistical: record '{exp_id}' has no resolvable "
            f"signal directory. signal_export_output_dir={stored_sig_dir!r} "
            f"(unset or path missing); manifest_path={manifest_path!r} (also "
            f"missing). Re-run the experiment or restore the output tree."
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


# V.1.5 A1 (2026-04-23): SSoT consolidation — retired the local
# `_sha256_file` helper. Previously this module inlined a 3rd divergent
# copy of SHA-256 file hashing (alongside hft_contracts.provenance.hash_file
# and hft_evaluator.fast_gate._hash_file) with semantics that raised on
# missing file. Now delegates to the canonical
# `hft_contracts.provenance.hash_file(..., missing_ok=False)` — preserves
# the raise-on-missing contract this module needs (callers already assert
# file existence via `_load_record_signals`'s required-file gate at line
# 226-235) while collapsing the duplication identified by Agent 3's
# Phase V 3rd-round architecture audit.


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

    # V.1.5 follow-up (2026-04-23) — SDR-8 axis-semantic check closure.
    # Cross-check labels_arr.shape[1] against metadata horizons list to
    # catch transposed (H, N) arrays that would silently pass the
    # `ndim in (1, 2)` gate but read the WRONG axis as the primary horizon.
    # Fail-loud per hft-rules §8 + §5 with an actionable hint distinguishing
    # transpose (shape[0] matches horizons count) from genuine schema skew.
    if labels_arr.ndim == 2:
        meta_horizons = meta.get("horizons")
        # `horizons` is a Phase II manifest field. Legacy pre-Phase-II
        # manifests lack it — skip silently (graceful back-compat; not a
        # drift signal). Only cross-check when the field is a non-empty list.
        if isinstance(meta_horizons, list) and len(meta_horizons) > 0:
            expected_H = len(meta_horizons)
            if labels_arr.shape[1] != expected_H:
                if labels_arr.shape[0] == expected_H:
                    hint = (
                        f" Axis order appears TRANSPOSED: array is "
                        f"{labels_arr.shape} but horizons list has "
                        f"{expected_H} entries — expected shape "
                        f"(N, {expected_H}). Re-export signals with a "
                        f"trainer that respects the (N, H) layout contract."
                    )
                else:
                    hint = (
                        f" Signal-export schema mismatch: neither axis of "
                        f"{labels_arr.shape} matches len(horizons)="
                        f"{expected_H}. Signal export is broken."
                    )
                raise ValueError(
                    f"compare_sweep_statistical: record '{exp_id}' "
                    f"horizon-axis mismatch — regression_labels shape "
                    f"{labels_arr.shape} incompatible with metadata "
                    f"horizons={meta_horizons}.{hint}"
                )

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
        regression_labels_sha256=hash_file(labels_path, missing_ok=False),
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


# Phase V.1 L1.3 (2026-04-21). Default threshold on silent NaN-row drop —
# above this fraction, `compare_sweep_statistical` raises rather than
# quietly computing on the surviving rows. Rationale: per hft-rules §8
# "Never silently drop, clamp, or 'fix' data without recording
# diagnostics", a high NaN-row fraction likely indicates upstream data
# corruption (trainer bug, signal-export failure, etc.) and the resulting
# bootstrap CI would be based on a biased subset. 5% is chosen to tolerate
# real-world data-quality noise (~occasional price-spike rows flagged by
# the feature extractor) while catching broken exports. Configurable per
# call via the `max_drop_frac` kwarg.
DEFAULT_MAX_DROP_FRAC: float = 0.05


@dataclass(frozen=True)
class CompareSweepDiagnostics:
    """Observability payload returned alongside the pairwise results.

    Phase V.1 L3.3 (2026-04-21). Closes Agent 2 H2 observability gap:
    previously the only signal of data-quality issues was a stdlib
    logging.warning call that the CLI never surfaced to the operator.
    This dataclass makes the diagnostics part of the public return
    contract so the CLI (and any future programmatic consumer) can
    render them directly in the results table.

    Attributes:
        n_treatments: Number of treatments (K columns in Y).
        n_samples_paired: Number of rows after NaN-drop masking.
            (Equals the length of x + Y that the primitive sees.)
        n_samples_raw: Total rows before NaN-drop masking.
        n_dropped_nonfinite: Count of rows dropped because x OR any
            Y[k] was non-finite. Drop is paired (same row mask applied
            across all treatments) to preserve the paired-bootstrap
            invariant.
        drop_fraction: ``n_dropped_nonfinite / n_samples_raw``.
        n_bootstraps: Bootstrap iterations used.
        block_length: Auto-selected or caller-supplied block length.
        primary_horizon_idx: Shared horizon index used for slicing
            multi-horizon regression signals.
        metric: Metric name (e.g. "val_ic") — verbatim from the caller.
    """

    n_treatments: int
    n_samples_paired: int
    n_samples_raw: int
    n_dropped_nonfinite: int
    drop_fraction: float
    n_bootstraps: int
    block_length: int
    primary_horizon_idx: int
    metric: str


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
    max_drop_frac: float = DEFAULT_MAX_DROP_FRAC,
) -> Tuple[List[PairwiseResult], List[str], CompareSweepDiagnostics]:
    """Paired pairwise-bootstrap comparison of a sweep's child records.

    Pipeline:
      1. Load each child record's signal files.
      2. Assert paired structure (identical regression_labels.npy +
         primary_horizon_idx across records).
      3. Assert treatment-label uniqueness (L3.2 — prevents ambiguous
         pairwise output if two grid points share axis_values).
      4. Stack ``x = regression_labels[:, h]`` (shared) and
         ``Y[:, k] = predicted_returns_k[:, h]``.
      5. NaN-row drop (L1.3): drop paired rows where x or any Y column
         is non-finite; if dropped fraction exceeds ``max_drop_frac``,
         raise ValueError with per-record actionable breakdown.
      6. Call ``hft_metrics.pairwise.pairwise_paired_bootstrap_compare``.

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
        max_drop_frac: (L1.3 Phase V.1, 2026-04-21). Maximum fraction of
            rows allowed to be NaN-dropped before aborting. Default
            ``DEFAULT_MAX_DROP_FRAC`` = 0.05 (5%). Pass ``1.0`` to
            disable the guard (not recommended; bypasses hft-rules §8).

    Returns:
        Tuple ``(pairwise_results, treatment_labels, diagnostics)``:
          - ``pairwise_results``: ``List[PairwiseResult]`` of length
            ``K*(K-1)/2``, ordered lexicographically (i, j) with i < j.
          - ``treatment_labels``: ``List[str]`` of length K — human-readable
            label for each column (from ``axis_values`` or record name).
            Phase V.1 L3.2: guaranteed unique (ValueError on duplicate).
          - ``diagnostics``: ``CompareSweepDiagnostics`` — observability
            payload for CLI rendering. Phase V.1 L3.3.

    Raises:
        ValueError: on any of:
          * ``< 2 records`` (nothing to compare)
          * ``unknown metric`` (dispatch mismatch)
          * missing signal files (regression MVP only)
          * unpaired labels (SHA-256 mismatch on ``regression_labels.npy``)
          * primary_horizon mismatch
          * duplicate treatment labels (L3.2 — sweep produced
            non-unique ``axis_values``)
          * NaN-row fraction ``> max_drop_frac`` (L1.3 — hft-rules §8
            fail-loud on silent data drops)
          * non-finite ``observed_stats`` (via primitive L2.7 guard)

    Example:
        >>> from hft_ops.ledger.ledger import Ledger
        >>> from hft_ops.paths import PipelinePaths
        >>> paths = PipelinePaths(pipeline_root="/path/to/monorepo")
        >>> ledger = Ledger(paths.ledger_dir)
        >>> entries = ledger.filter(sweep_id="seed_stability_20260421")
        >>> entries = [e for e in entries if e.get("record_type") != "sweep_aggregate"]
        >>> results, labels, diag = compare_sweep_statistical(entries, ledger, paths)
        >>> print(f"K={diag.n_treatments}, paired_n={diag.n_samples_paired}, "
        ...       f"dropped={diag.n_dropped_nonfinite} ({diag.drop_fraction:.2%})")
        >>> for r in results:
        ...     print(f"{labels[r.i]} vs {labels[r.j]}: "
        ...           f"delta={r.delta:+.4f} "
        ...           f"CI=[{r.ci_lower:+.4f},{r.ci_upper:+.4f}] "
        ...           f"p_bh={r.p_value_bh:.3f} "
        ...           f"nonfinite={r.n_nonfinite_replaced}")
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
    if not (0.0 <= max_drop_frac <= 1.0):
        raise ValueError(
            f"compare_sweep_statistical: max_drop_frac must be in [0.0, 1.0]; "
            f"got {max_drop_frac}. Use 1.0 to disable the guard (not "
            f"recommended — bypasses hft-rules §8)."
        )

    statistic_fn = _STATISTIC_FN_REGISTRY[metric]

    # === Load per-record signals ===
    loaded = [
        _load_record_signals(entry, ledger, paths)
        for entry in sweep_child_entries
    ]

    # === Paired-structure check ===
    _assert_paired_labels(loaded)

    # === L3.2 Label uniqueness check ===
    # Defensive: sweep expansion SHOULD produce unique axis_values for
    # every grid point (Phase 5 FULL-A invariant), but a misconfigured
    # sweep YAML (e.g., empty values list collapsing to identical labels)
    # could produce duplicates — in which case the pairwise output table
    # would show ambiguous "A vs B" rows where A and B have the same
    # rendered label. Fail-loud per hft-rules §5 / §8.
    labels_for_check = [r.label for r in loaded]
    if len(set(labels_for_check)) != len(labels_for_check):
        from collections import Counter
        counts = Counter(labels_for_check)
        duplicates = {lbl: c for lbl, c in counts.items() if c > 1}
        raise ValueError(
            f"compare_sweep_statistical: duplicate treatment labels "
            f"detected: {duplicates}. Every grid point's axis_values "
            f"(or experiment name when axis_values is empty) must "
            f"produce a UNIQUE label — otherwise the pairwise output "
            f"table has ambiguous rows. Check the sweep manifest for "
            f"axis-values collisions (e.g., empty values list, or "
            f"duplicate labels)."
        )

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

    # === L1.3 NaN-row drop with threshold ===
    # Defense-in-depth: paired bootstrap requires clean paired arrays.
    # Drop rows where x or any Y column is non-finite (preserves pairing
    # by using the same mask across all treatments). Pre-V.1.L1.3 behavior
    # silently WARN'd via stdlib logging (invisible to CLI) + proceeded
    # regardless of drop fraction. Post-L1.3 behavior: raise when
    # `dropped / total > max_drop_frac` — the threshold distinguishes
    # real-world data-quality noise (< 5% typical) from broken upstream
    # (trainer bug, signal-export failure).
    n_samples_raw = int(len(x))
    mask = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    n_dropped = int((~mask).sum())
    drop_fraction = n_dropped / n_samples_raw if n_samples_raw > 0 else 0.0

    if drop_fraction > max_drop_frac:
        # Per-record breakdown to help operator diagnose.
        per_record_drops = []
        for k, r in enumerate(loaded):
            if r.regression_labels_full.ndim == 1:
                bad_x = (~np.isfinite(r.regression_labels_full)).sum()
                bad_y = (~np.isfinite(r.predicted_returns_full)).sum()
            else:
                bad_x = (~np.isfinite(r.regression_labels_full[:, h])).sum()
                bad_y = (~np.isfinite(r.predicted_returns_full[:, h])).sum()
            per_record_drops.append(
                f"{r.label} (exp_id={r.experiment_id}): "
                f"x_nonfinite={int(bad_x)}, y_nonfinite={int(bad_y)}"
            )
        raise ValueError(
            f"compare_sweep_statistical: NaN-row drop fraction "
            f"{drop_fraction:.2%} exceeds max_drop_frac={max_drop_frac:.2%} "
            f"(dropped {n_dropped}/{n_samples_raw} rows). Likely an "
            f"upstream issue (trainer bug, signal-export corruption, "
            f"feature-extractor flagging). Per-record counts:\n  "
            + "\n  ".join(per_record_drops)
            + f"\n\nTo override (not recommended), pass "
            f"max_drop_frac=1.0. This bypasses the hft-rules §8 "
            f"fail-loud guard."
        )

    if n_dropped > 0:
        # Under threshold — drop rows + log. Also surfaced via
        # CompareSweepDiagnostics for CLI rendering.
        x = x[mask]
        Y = Y[mask]
        import logging
        logging.getLogger(__name__).warning(
            "compare_sweep_statistical: dropped %d/%d rows (%.2f%%) with "
            "non-finite values (NaN/Inf) — within max_drop_frac=%.2f%% "
            "threshold; preserves pairing across all treatments.",
            n_dropped, n_samples_raw, 100 * drop_fraction,
            100 * max_drop_frac,
        )

    # === Delegate to hft-metrics primitive ===
    # Resolve block_length for the diagnostics payload (primitive uses
    # the same default if None; we echo for observability).
    effective_block_length = (
        block_length
        if block_length is not None
        else max(1, math.ceil(len(x) ** (1.0 / 3.0)))
    )

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

    diagnostics = CompareSweepDiagnostics(
        n_treatments=len(loaded),
        n_samples_paired=int(len(x)),
        n_samples_raw=n_samples_raw,
        n_dropped_nonfinite=n_dropped,
        drop_fraction=drop_fraction,
        n_bootstraps=n_bootstraps,
        block_length=effective_block_length,
        primary_horizon_idx=h,
        metric=metric,
    )

    return results, treatment_labels, diagnostics
