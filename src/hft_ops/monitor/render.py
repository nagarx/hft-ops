"""Renderers for the monitor surface — text (rich), markdown (GFM), json (machine).

``--format json`` is the contract for downstream programmatic monitoring; it
preserves the load-bearing ``edge`` None-vs-False distinction. Torch-free.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

from rich.console import Console
from rich.table import Table as RichTable

from .drift import DriftReport
from .table import MonitorTable

_COLUMNS = (
    "kind", "id", "name", "source", "status_or_verdict", "edge",
    "authority", "metric", "dsr", "prov", "stats_v", "drift",
)


def _edge(edge):
    if edge is None:
        return "—"
    return "yes" if edge else "no"


def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else ""


def _metric(row):
    if row.primary_metric is None:
        return ""
    name = f" ({row.primary_metric_name})" if row.primary_metric_name else ""
    return f"{row.primary_metric:.4f}{name}"


def _row_cells(row):
    return (
        row.kind, row.id, row.name, row.source, row.status_or_verdict,
        _edge(row.edge), row.authority, _metric(row), _f(row.dsr),
        row.provenance_id, row.stats_version, ",".join(row.drift_flags),
    )


def render_json(table: MonitorTable) -> str:
    return json.dumps(asdict(table), sort_keys=True, indent=2, default=str)


def render_markdown(table: MonitorTable) -> str:
    header = "| " + " | ".join(_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines = [header, sep]
    for row in table.rows:
        lines.append("| " + " | ".join(str(c) for c in _row_cells(row)) + " |")
    return "\n".join(lines)


def render_text(table: MonitorTable) -> str:
    rt = RichTable(show_header=True, header_style="bold cyan")
    for col in _COLUMNS:
        rt.add_column(col, overflow="fold")
    for row in table.rows:
        rt.add_row(*(str(c) for c in _row_cells(row)))
    buf = io.StringIO()
    Console(file=buf, width=240).print(rt)
    out = buf.getvalue()
    if table.drift.findings:
        out += "\n" + render_drift_text(table.drift)
    return out


def render_drift_json(report: DriftReport) -> str:
    return json.dumps(
        {
            "n_errors": report.n_errors,
            "n_warn": report.n_warn,
            "findings": [asdict(f) for f in report.findings],
        },
        sort_keys=True,
        indent=2,
    )


def render_drift_text(report: DriftReport) -> str:
    if not report.findings:
        return "No drift detected."
    lines = [f"Drift: {report.n_errors} error(s), {report.n_warn} warning(s)"]
    for f in report.findings:
        lines.append(f"  [{f.severity}] {f.kind} :: {f.subject} — {f.detail}")
    return "\n".join(lines)
