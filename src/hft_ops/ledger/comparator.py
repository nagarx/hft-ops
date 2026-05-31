"""
Experiment comparison and ranking.

Provides utilities to compare experiments by metrics, diff configs,
rank by performance, and group by model type or labeling strategy.
Outputs formatted tables via Rich for terminal display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from hft_contracts.experiment_record import ExperimentRecord


from hft_ops.utils import get_nested as _get_nested


def compare_experiments(
    entries: List[Dict[str, Any]],
    *,
    metric_keys: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    top_k: Optional[int] = None,
    group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compare experiments by selected metrics.

    Args:
        entries: List of ledger index entries.
        metric_keys: Which metrics to include. Defaults to common metrics.
            Supports dotted keys into training_metrics and backtest_metrics.
        sort_by: Metric key to sort by. Supports "training_metrics.macro_f1", etc.
        ascending: Sort direction.
        top_k: Return only top K entries after sorting.
        group_by: Group results by this field (e.g., "model_type").

    Returns:
        List of comparison rows, each a dict with experiment info + metrics.
    """
    if metric_keys is None:
        metric_keys = [
            "training_metrics.accuracy",
            "training_metrics.macro_f1",
            "backtest_metrics.total_return",
            "backtest_metrics.sharpe_ratio",
        ]

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        row: Dict[str, Any] = {
            "experiment_id": entry.get("experiment_id", ""),
            "name": entry.get("name", ""),
            "model_type": entry.get("model_type", ""),
            "status": entry.get("status", ""),
            "created_at": entry.get("created_at", "")[:10],
        }

        for key in metric_keys:
            value = _get_nested(entry, key)
            short_key = key.split(".")[-1]
            row[short_key] = value if value is not None else ""

        rows.append(row)

    if sort_by:
        sort_key = sort_by.split(".")[-1]

        def _sort_val(row: Dict[str, Any]) -> float:
            v = row.get(sort_key, "")
            if isinstance(v, (int, float)):
                return float(v)
            return float("-inf") if not ascending else float("inf")

        rows.sort(key=_sort_val, reverse=not ascending)

    if top_k is not None and top_k > 0:
        rows = rows[:top_k]

    if group_by:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            gval = str(row.get(group_by, "unknown"))
            groups.setdefault(gval, []).append(row)
        grouped_rows: List[Dict[str, Any]] = []
        for group_name, group_entries in sorted(groups.items()):
            for entry in group_entries:
                entry["_group"] = group_name
                grouped_rows.append(entry)
        rows = grouped_rows

    return rows


def find_unresolved_metric_keys(
    entries: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
) -> List[str]:
    """Return the metric keys that resolve to ``None`` in EVERY entry.

    Such a key contributes nothing to the comparison — almost always a typo or a
    metric not projected into the ledger index. The CLI surfaces these as a WARN
    so a blank ``compare`` column becomes distinguishable from a genuine all-blank
    result (the ``--stages`` footgun's twin; metric keys are an OPEN dotted-path
    set -> WARN, not the closed-set RAISE). Empty ``entries`` -> all keys unresolved
    (vacuous: nothing to resolve against — the CLI only WARNs when entries is
    non-empty, so this never false-alarms). Uses the same ``_get_nested`` resolver
    the comparison itself uses, so "resolves to None here" == "renders blank there".
    """
    return [
        key
        for key in metric_keys
        if all(_get_nested(entry, key) is None for entry in entries)
    ]


def diff_experiments(
    record_a: ExperimentRecord,
    record_b: ExperimentRecord,
) -> Dict[str, Any]:
    """Produce a detailed diff between two experiment records.

    Returns a dict with:
    - config_diffs: list of (key, value_a, value_b) for differing config values
    - metric_diffs: list of (metric, value_a, value_b, delta) for metric differences
    - compatibility_fingerprint: Phase V.1 L2.2 (2026-04-21) — surfaces the
      CompatibilityContract fingerprint divergence as a first-class diff field.
      Values: ``None`` when the fingerprints AGREE (or both are None);
      a ``Tuple[Optional[str], Optional[str]]`` (fp_a, fp_b) when they
      DIFFER. A differing fingerprint means the two records were produced
      against different signal-boundary contract versions — config diffs
      alone may not surface this (e.g., same training_config but different
      extraction provenance). The fingerprint is the authoritative
      cross-experiment-comparability flag per the V.A.4 trust-column
      design. If one side has the fingerprint and the other doesn't,
      the diff still surfaces — the caller can decide whether that's a
      meaningful provenance asymmetry (legacy record without harvest vs
      post-V.A.4 record with harvest).
    - experiment_provenance_hash: Phase Y / γ-1 LITE / #PY-95 (2026-05-10) —
      surfaces the 4-source composer fingerprint divergence (data_export_fp +
      feature_set_content_hash + compatibility_fp + model_config_hash composed
      via ``compute_experiment_provenance_hash``). Same Tuple-on-divergence
      semantics as ``compatibility_fingerprint``. Closes #PY-95 inherited
      from Phase Y deployment 2026-05-05 — pre-#PY-95 the diff surfaced
      compat-fp divergence but stayed silent on the higher-level epH which
      catches model-axis architecture drift compat-fp alone cannot.
    - model_config_hash: Phase Y / γ-1 LITE / #PY-95 (2026-05-10) — surfaces
      the model-axis identity divergence (filtered ``model.params`` SHA-256;
      ``_LOSS_TUNING_KEYS`` denylist preserves stability under loss-tuning
      hyperparam changes). Read from
      ``(record.training_config or {}).get("model_config_hash")`` per the
      Bundle Commit 1+2 nested-storage convention; surfaced as Tuple on
      divergence. Use case: cross-experiment ablation queries via
      ``ledger list --model-config-hash <hex>`` find re-runs of identical
      arch on different data; ``diff`` between two such records will show
      mch=None (matching) AND data-axis divergence in compat-fp.

    Args:
        record_a: First experiment record.
        record_b: Second experiment record.

    Returns:
        Structured diff dict with keys:
          - experiment_a (str), experiment_b (str)
          - config_diffs (list)
          - metric_diffs (list)
          - compatibility_fingerprint (None | Tuple[Optional[str], Optional[str]])
          - experiment_provenance_hash (None | Tuple[Optional[str], Optional[str]])
          - model_config_hash (None | Tuple[Optional[str], Optional[str]])
          - producer_commits (None | Tuple[Dict[str, str], Dict[str, str]]) —
            Foundation-Integrity producer-code lineage divergence (Step 5,
            2026-05-31); None when equal/both-empty, else the two dicts.
    """
    config_diffs: List[Tuple[str, Any, Any]] = []
    metric_diffs: List[Tuple[str, Any, Any, Any]] = []

    _diff_dicts(
        record_a.extraction_config,
        record_b.extraction_config,
        prefix="extraction",
        diffs=config_diffs,
    )
    _diff_dicts(
        record_a.training_config,
        record_b.training_config,
        prefix="training",
        diffs=config_diffs,
    )
    _diff_dicts(
        record_a.backtest_params,
        record_b.backtest_params,
        prefix="backtest",
        diffs=config_diffs,
    )

    all_metric_keys = set(record_a.training_metrics) | set(record_b.training_metrics)
    for key in sorted(all_metric_keys):
        va = record_a.training_metrics.get(key)
        vb = record_b.training_metrics.get(key)
        if va != vb:
            delta = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
            metric_diffs.append((f"training.{key}", va, vb, delta))

    all_bt_keys = set(record_a.backtest_metrics) | set(record_b.backtest_metrics)
    for key in sorted(all_bt_keys):
        va = record_a.backtest_metrics.get(key)
        vb = record_b.backtest_metrics.get(key)
        if va != vb:
            delta = None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
            metric_diffs.append((f"backtest.{key}", va, vb, delta))

    # Phase V.1 L2.2 (2026-04-21): surface CompatibilityContract fingerprint
    # divergence as a first-class diff field. None when they match; tuple
    # when they differ. Pre-V.A.4 records carry None — (None, None) → agree;
    # (None, "abc...") → asymmetric provenance.
    fp_a = getattr(record_a, "compatibility_fingerprint", None)
    fp_b = getattr(record_b, "compatibility_fingerprint", None)
    compatibility_fp_diff: Optional[Tuple[Optional[str], Optional[str]]] = (
        (fp_a, fp_b) if fp_a != fp_b else None
    )

    # Phase Y / γ-1 LITE / #PY-95 (2026-05-10): surface
    # experiment_provenance_hash divergence (4-source composer fingerprint)
    # as a first-class diff field. Mirrors compatibility_fingerprint pattern.
    eph_a = getattr(record_a, "experiment_provenance_hash", None)
    eph_b = getattr(record_b, "experiment_provenance_hash", None)
    experiment_provenance_hash_diff: Optional[Tuple[Optional[str], Optional[str]]] = (
        (eph_a, eph_b) if eph_a != eph_b else None
    )

    # Phase Y / γ-1 LITE / #PY-95 (2026-05-10): surface model_config_hash
    # divergence (filtered model.params SHA-256). Read from nested
    # training_config per Bundle Commit 1+2 storage convention; the
    # top-level mirror in index_entry() is for fast-filter queries, but
    # ground-truth nested value is what the composer reads.
    mch_a = (record_a.training_config or {}).get("model_config_hash")
    mch_b = (record_b.training_config or {}).get("model_config_hash")
    model_config_hash_diff: Optional[Tuple[Optional[str], Optional[str]]] = (
        (mch_a, mch_b) if mch_a != mch_b else None
    )

    # Step 5 (2026-05-31): surface producer_commits (the Foundation-Integrity
    # producer-code lineage — extractor/reconstructor/hft_statistics git shas +
    # completeness) divergence as a first-class diff field. The phase CAPTURES it
    # (extraction.py:141,200 -> record.provenance.producer_commits) but diff never
    # read provenance, so two records built from DIFFERENT reconstructor commits
    # showed identical diff output — defeating the phase's purpose (catch
    # silently-wrong Rust-producer lineage). None when equal (or both empty); a
    # Tuple of the two dicts when they differ. Defensive getattr: a record
    # predating the provenance/producer_commits field reads as {}.
    pc_a = getattr(getattr(record_a, "provenance", None), "producer_commits", {}) or {}
    pc_b = getattr(getattr(record_b, "provenance", None), "producer_commits", {}) or {}
    producer_commits_diff: Optional[Tuple[Dict[str, str], Dict[str, str]]] = (
        (pc_a, pc_b) if pc_a != pc_b else None
    )

    return {
        "experiment_a": record_a.experiment_id,
        "experiment_b": record_b.experiment_id,
        "config_diffs": config_diffs,
        "metric_diffs": metric_diffs,
        "compatibility_fingerprint": compatibility_fp_diff,
        "experiment_provenance_hash": experiment_provenance_hash_diff,
        "model_config_hash": model_config_hash_diff,
        "producer_commits": producer_commits_diff,
    }


def _diff_dicts(
    a: Dict[str, Any],
    b: Dict[str, Any],
    prefix: str,
    diffs: List[Tuple[str, Any, Any]],
) -> None:
    """Recursively diff two dicts, appending (key, val_a, val_b) to diffs."""
    all_keys = set(a) | set(b)
    for key in sorted(all_keys):
        va = a.get(key)
        vb = b.get(key)
        full_key = f"{prefix}.{key}"

        if isinstance(va, dict) and isinstance(vb, dict):
            _diff_dicts(va, vb, full_key, diffs)
        elif va != vb:
            diffs.append((full_key, va, vb))


def rank_experiments(
    entries: List[Dict[str, Any]],
    *,
    metric: str = "training_metrics.macro_f1",
    top_k: int = 10,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Rank experiments by a single metric.

    Convenience wrapper around compare_experiments for simple ranking.
    """
    return compare_experiments(
        entries,
        metric_keys=[metric],
        sort_by=metric,
        ascending=ascending,
        top_k=top_k,
    )
