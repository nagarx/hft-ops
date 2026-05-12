"""
Resolve a record's signal-export output directory.

Phase R-16c Sub-cycle 4b SSoT extraction (2026-05-12): extracted from
``hft_ops.ledger.statistical_compare._resolve_signal_dir`` to support
consumption by ``r16c_analysis.py`` (2nd consumer). The 2-consumer
threshold triggers §0 reuse-first SSoT extraction per CLAUDE.md
"Reuse-first: no duplication" + "extends Class B SSoT primitives".

The function preserves all V.1.L1.2 + V.A.5 + V.2 Q5 hardening accumulated
across the prior cycles:

- PREFERRED: ``record.signal_export_output_dir`` — absolute path captured
  at RUN TIME by ``SignalExportRunner.run`` + attached by
  ``cli.py::_record_experiment``. Available for all Phase-V.1-and-later
  runs. Direct use — no manifest re-parse, no variable substitution.

- FALLBACK: re-parse manifest via ``record.manifest_path`` and re-resolve
  ``stages.signal_export.output_dir`` through ``paths.resolve(...)``.
  Used for pre-V.1.L1.2 records (signal_export_output_dir=None) — graceful
  back-compat.

- Both paths fail-loud on missing/inconsistent state per hft-rules §8.
- ``is_dir()`` (not ``exists()``) gates the preferred path per V.2 Q5
  defensive check (mis-written str pointing to a FILE not a DIR would
  silently pass ``exists()`` then produce opaque nested-path errors
  downstream).
- Single ``RuntimeWarning`` per unique stored_sig_dir on fallback
  (V.1.5 SDR-6 closure).

Module placement (per Agent G design review 2026-05-12): public sibling
inside ``hft_ops.ledger`` subpackage; consumed by ``statistical_compare``
+ ``r16c_analysis``. Both are pure-Python consumers of the same producer
metadata (ledger records + manifest stages).

Back-compat alias preserved at ``statistical_compare._resolve_signal_dir``
for the 16 existing references (12 mock-patch sites in
``tests/test_statistical_compare.py`` + 10 direct call sites in
``tests/test_resolve_signal_dir.py``). The alias is a one-line module-
level assignment ``_resolve_signal_dir = resolve_signal_dir`` so the
canonical dotted import path
``hft_ops.ledger.statistical_compare._resolve_signal_dir`` still resolves
without test churn.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


def resolve_signal_dir(
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
        ``signal_metadata.json`` (and for R-16c, the
        ``{run_name}__option_trade_pnls__{label}.npy`` per-threshold dumps).

    Raises:
        ValueError: if the record cannot be located, or if BOTH the
            preferred and fallback paths fail to resolve (manifest missing
            AND signal_export_output_dir absent/non-existent).
    """
    exp_id = record_entry.get("experiment_id")
    full_record = ledger.get(exp_id)
    if full_record is None:
        raise ValueError(
            f"resolve_signal_dir: record '{exp_id}' not found in ledger "
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
            f"resolve_signal_dir: record '{exp_id}' has no resolvable "
            f"signal directory. signal_export_output_dir={stored_sig_dir!r} "
            f"(unset or path missing); manifest_path={manifest_path!r} (also "
            f"missing). Re-run the experiment or restore the output tree."
        )

    manifest = load_manifest(manifest_path)
    signal_stage = manifest.stages.signal_export
    if not signal_stage.enabled:
        raise ValueError(
            f"resolve_signal_dir: record '{exp_id}' has "
            f"signal_export disabled — no signal files to compare. "
            f"Re-run with signal_export enabled or exclude from sweep."
        )
    if not signal_stage.output_dir:
        raise ValueError(
            f"resolve_signal_dir: record '{exp_id}' signal_export "
            f"stage has no output_dir set."
        )
    return paths.resolve(signal_stage.output_dir)


__all__ = ["resolve_signal_dir"]
