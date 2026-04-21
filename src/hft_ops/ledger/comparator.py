"""
Experiment comparison and ranking.

Provides utilities to compare experiments by metrics, diff configs,
rank by performance, and group by model type or labeling strategy.
Outputs formatted tables via Rich for terminal display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from hft_ops.ledger.experiment_record import ExperimentRecord


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

    Args:
        record_a: First experiment record.
        record_b: Second experiment record.

    Returns:
        Structured diff dict with keys:
          - experiment_a (str), experiment_b (str)
          - config_diffs (list)
          - metric_diffs (list)
          - compatibility_fingerprint (None | Tuple[Optional[str], Optional[str]])
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

    return {
        "experiment_a": record_a.experiment_id,
        "experiment_b": record_b.experiment_id,
        "config_diffs": config_diffs,
        "metric_diffs": metric_diffs,
        "compatibility_fingerprint": compatibility_fp_diff,
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
