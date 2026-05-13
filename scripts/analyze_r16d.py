#!/usr/bin/env python3
"""
CLI wrapper for R-16d horizon-axis sweep analysis (Cycle 8 / Phase B.4, 2026-05-13).

Pre-registered decision-gate analyzer for the cycle8_r16d_horizon_axis
sweep. Loads 12 grid-point records from the ledger, extracts per-cell
test_ic, computes pooled per-trade bootstrap CI for each
(model_type × return_type × horizon × threshold) cell, applies the
5-gate matrix (H1 PRIMARY horizon-decay + H2 BASELINE Ridge/TLOB ratio +
H3 COST + H4 NEGATIVE CONTROL + H5 ARCHITECTURAL horizon-axis distinct),
and renders the 4-way verdict (GO / REFUTE / INDETERMINATE / ABORT).

Standalone script per #PY-121 (cli.py god-object) + #PY-167 (LOC ratchet)
anti-ratchets — mirrors `scripts/analyze_r16c.py` precedent. Library at
``hft_ops.ledger.r16d_analysis``.

Usage:
    # Human-readable verdict + per-arm decay + per-cell H2 table
    python scripts/analyze_r16d.py <sweep_id>

    # JSON output
    python scripts/analyze_r16d.py <sweep_id> --json

    # Allow partial sweep (e.g., cross-sweep dedup case)
    python scripts/analyze_r16d.py <sweep_id> --allow-partial --min-grid-points 10

Exit codes:
    0  GO          — horizon-decay validated (H1 + H2 + H4 + H5 all PASS)
    1  REFUTE      — H5 OK but H1 <2/4 arms decay (or H2/H4 borderline)
       INDETERMINATE — H1 = 2/4 arms; trigger R-16d-extended N=4 seeds
    2  ABORT       — H5 FAIL (horizon axis cosmetic; shadow-precedence bug)
    3  INCOMPLETE  — missing records or per-trade .npy files

Same-session ledger discipline per hft-rules §13: after invocation,
document verdict + key statistics in:
- lob-model-trainer/EXPERIMENT_INDEX.md (R-16d entry)
- lob-backtester/BACKTEST_INDEX.md (Round 16d)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze R-16d horizon-axis sweep + render pre-registered verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_id",
        help="Sweep aggregate identifier (e.g., "
             "cycle8_r16d_horizon_axis_20260513T220000)",
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
             "(#PY-180 close, 2026-05-13). Default 100_000.0 matches "
             "lob-backtester producer-side hardcoded value. Override only if "
             "the sweep manifest used a non-default initial_capital.",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow analysis with fewer than 12 grid records. Caller must "
             "document the rationale in the EXPERIMENT_INDEX/BACKTEST_INDEX "
             "entry per hft-rules §13. Per-cell bootstrap CI may widen.",
    )
    parser.add_argument(
        "--min-grid-points", type=int, default=10,
        help="Minimum grid record count when --allow-partial. Default 10 "
             "(10/12 = 17%% missing). R-16d is single-seed (no per-seed dedup "
             "expected); use --allow-partial primarily for retry-after-failure "
             "scenarios where individual cells failed mid-sweep.",
    )
    args = parser.parse_args()

    from hft_ops.ledger.ledger import ExperimentLedger
    from hft_ops.ledger.r16d_analysis import (
        analyze_r16d_sweep,
        render_verdict,
        outcome_to_json_dict,
        R16dIncompleteSweepError,
        R16dH5HorizonAxisError,
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
        cell_results, outcome = analyze_r16d_sweep(
            sweep_id=args.sweep_id,
            ledger=ledger,
            paths=paths,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            initial_capital=args.initial_capital,
            allow_partial=args.allow_partial,
            min_grid_points=args.min_grid_points,
        )
    except R16dIncompleteSweepError as e:
        print(f"ERROR (exit 3): {e}", file=sys.stderr)
        return 3
    except R16dH5HorizonAxisError as e:
        print(
            f"ERROR (exit 2): H5 horizon-axis FAILED: {e}\n"
            f"\nHorizon axis is COSMETIC — same bug class as closed #PY-87/#PY-88. "
            f"Check sweep.axes[horizon] override paths in the manifest "
            f"(must override data.horizon_idx + data.labels.primary_horizon_idx + "
            f"horizon_value all 3 fields per cell).",
            file=sys.stderr,
        )
        return 2

    if args.json_output:
        print(json.dumps(outcome_to_json_dict(outcome, cell_results), indent=2))
    else:
        print(render_verdict(outcome, cell_results))

    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
