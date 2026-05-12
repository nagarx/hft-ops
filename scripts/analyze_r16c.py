#!/usr/bin/env python3
"""
CLI wrapper for R-16c multi-seed power analysis (Sub-cycle 4b, 2026-05-12).

Pre-registered decision-gate analyzer for the cycle7_r16c_multi_seed_r16a
sweep. Loads 40 grid-point records from the ledger, computes pooled
per-trade bootstrap CI for each (model_type × return_type × threshold)
cell, applies H1 + H4 + H5 gates, and renders the 4-way verdict
(GO / REFUTE / INDETERMINATE / ABORT).

Standalone script per Agent I T10 verdict 2026-05-12: NOT extended into
``hft-ops cli.py`` to preserve #PY-121 (cli.py god-object) + #PY-167
(LOC ratchet) anti-ratchets. Library is at
``hft_ops.ledger.r16c_analysis``; this wrapper is ~80 LOC of argparse +
verdict-rendering glue.

Usage:
    # Human-readable verdict + per-cell summary table
    python scripts/analyze_r16c.py <sweep_id>

    # JSON output (for downstream tooling)
    python scripts/analyze_r16c.py <sweep_id> --json

    # Custom ledger root (default: auto-resolve via PipelinePaths)
    python scripts/analyze_r16c.py <sweep_id> --ledger-root /path/to/ledger

Exit codes:
    0  GO          — F7 replicated as alpha (H1 + H4 + H5 all PASS)
    1  REFUTE      — H5 OK but any H1 conjunct fails OR H4 fails
       INDETERMINATE — borderline; trigger R-16c-extended N=20
    2  ABORT       — H5 invariant violated (Phase A.3 REDESIGN ship-blocker)
    3  INCOMPLETE  — missing records or per-trade .npy files

Same-session ledger discipline per hft-rules §13: after invocation,
document verdict + key statistics in:
- lob-model-trainer/EXPERIMENT_INDEX.md (R-16c entry)
- lob-backtester/BACKTEST_INDEX.md (R-16c round)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze R-16c sweep + render pre-registered verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_id",
        help="Sweep aggregate identifier (e.g., "
             "cycle7_r16c_multi_seed_r16a_20260513T120000)",
    )
    parser.add_argument(
        "--ledger-root", type=Path, default=None,
        help="Ledger root directory. Default: auto-resolve via PipelinePaths.",
    )
    parser.add_argument(
        "--paths-root", type=Path, default=None,
        help="Pipeline root directory. Default: auto-resolve.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Bootstrap iterations per cell. Default: 10000 (tight CI).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap RNG seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Backtester initial capital (USD) for DOLLAR→FRACTION conversion "
             "at the per-trade-pnls load boundary (#PY-180 close, 2026-05-13). "
             "Default 100_000.0 matches producer-side hardcoded "
             "lob-backtester/scripts/run_regression_backtest.py:158 + manifest "
             "schema.py:205 BacktestParams.initial_capital default. Override "
             "if your sweep manifest used a non-default initial_capital "
             "(#PY-181 follow-up will persist this to backtest record summary "
             "for automatic discovery).",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow analysis with fewer than 40 grid records (e.g., when "
             "seed_42 was deduped against an earlier sweep). Caller must "
             "document the rationale in the EXPERIMENT_INDEX/BACKTEST_INDEX "
             "entry per hft-rules §13. Per-cell bootstrap CI may widen with "
             "reduced sample-per-cell.",
    )
    parser.add_argument(
        "--min-grid-points", type=int, default=36,
        help="Minimum grid record count when --allow-partial. Default 36 "
             "(36/40 = 10%% missing — typical cross-sweep dedup case).",
    )
    args = parser.parse_args()

    from hft_ops.ledger.ledger import ExperimentLedger
    from hft_ops.ledger.r16c_analysis import (
        analyze_r16c_sweep,
        render_verdict,
        outcome_to_json_dict,
        R16cIncompleteSweepError,
        R16cH5InvariantError,
    )
    from hft_ops.paths import PipelinePaths

    # Resolve paths (auto-detect from CWD if not supplied)
    if args.paths_root is not None:
        paths = PipelinePaths.from_pipeline_root(args.paths_root)
    else:
        paths = PipelinePaths.auto_detect()

    ledger_root = args.ledger_root or (paths.pipeline_root / "hft-ops" / "ledger")
    ledger = ExperimentLedger(ledger_root)

    try:
        cell_results, outcome = analyze_r16c_sweep(
            sweep_id=args.sweep_id,
            ledger=ledger,
            paths=paths,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            initial_capital=args.initial_capital,
            allow_partial=args.allow_partial,
            min_grid_points=args.min_grid_points,
        )
    except R16cIncompleteSweepError as e:
        print(f"ERROR (exit 3): {e}", file=sys.stderr)
        return 3
    except R16cH5InvariantError as e:
        print(f"ERROR (exit 2): H5 architectural invariant FAILED: {e}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps(outcome_to_json_dict(outcome, cell_results), indent=2))
    else:
        print(render_verdict(outcome, cell_results))

    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
