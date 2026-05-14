#!/usr/bin/env python3
"""
CLI wrapper for R-16e EXTENDED sweep analysis (Cycle 9 / Phase 2, 2026-05-14).

Pre-registered decision-gate analyzer for the cycle9_r16e_multi_seed_h60_point
sweep. Loads 40 grid-point records from the ledger, extracts per-cell
test_ic, computes pooled per-trade bootstrap CI for each
(model_type × return_type × threshold) cell at H60-hold, applies the
4-gate matrix (H1 PRIMARY tradeability + H2 BASELINE Ridge/TLOB IC ratio +
H4 ARCHITECTURAL Ridge bit-exact + H6 E8 label-execution diagnostic),
and renders the 4-way verdict (GO / REFUTE / INDETERMINATE / ABORT).

Standalone script mirrors `scripts/analyze_r16c.py` + `scripts/analyze_r16d.py`
precedents per #PY-121 (cli.py god-object) + #PY-167 (LOC ratchet) anti-ratchets.
Library at ``hft_ops.ledger.r16e_analysis``.

Usage:
    # Human-readable verdict + per-cell summary + H2 ratio block + H6 diagnostic
    python scripts/analyze_r16e.py <sweep_id>

    # JSON output
    python scripts/analyze_r16e.py <sweep_id> --json

    # Allow partial sweep (e.g., cross-cycle dedup or retry-after-failure)
    python scripts/analyze_r16e.py <sweep_id> --allow-partial --min-grid-points 30

Exit codes:
    0  GO           — H1 (3-conjunctive) + H4 all PASS — Ridge×Point×H60 tradeable
    1  REFUTE       — H4 OK but any H1 conjunct FAILS — closes Ridge×Point×H60
                       direction; unblocks Option C (Triple-Barrier) or
                       Option E (feature ablation)
       INDETERMINATE — H4 OK + borderline H1 — trigger R-16e-extended N=20
    2  ABORT        — H4 FAIL (Ridge non-deterministic across seeds — Phase A.3
                       REDESIGN ship-blocker)
    3  INCOMPLETE   — missing records or per-trade .npy files

Same-session ledger discipline per hft-rules §13: after invocation,
document verdict + key statistics in:
- lob-model-trainer/EXPERIMENT_INDEX.md (R-16e entry, NEW)
- lob-backtester/BACKTEST_INDEX.md (Round 16e, NEW)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze R-16e EXTENDED sweep + render pre-registered verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sweep_id",
        help="Sweep aggregate identifier (e.g., "
             "cycle9_r16e_multi_seed_h60_point_20260514T060000)",
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
        help="Backtester initial capital (USD) for DOLLAR→FRACTION conversion. "
             "Default 100_000.0 matches lob-backtester producer-side hardcoded. "
             "Override only if the sweep manifest used a non-default initial_capital.",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Emit JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow analysis with fewer than 40 grid records. Per-cell bootstrap "
             "CI may widen. Caller must document rationale in EXPERIMENT_INDEX/"
             "BACKTEST_INDEX entry per hft-rules §13.",
    )
    parser.add_argument(
        "--min-grid-points", type=int, default=30,
        help="Minimum grid record count when --allow-partial. Default 30 "
             "(30/40 = 25%% missing tolerance). Use for retry-after-failure "
             "scenarios where individual cells failed mid-sweep.",
    )
    args = parser.parse_args()

    from hft_ops.ledger.ledger import ExperimentLedger
    from hft_ops.ledger.r16e_analysis import (
        analyze_r16e_sweep,
        render_verdict,
        outcome_to_json_dict,
        R16eIncompleteSweepError,
        R16eH4InvariantError,
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
        cell_results, ratio_results, outcome = analyze_r16e_sweep(
            sweep_id=args.sweep_id,
            ledger=ledger,
            paths=paths,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            initial_capital=args.initial_capital,
            allow_partial=args.allow_partial,
            min_grid_points=args.min_grid_points,
        )
    except R16eIncompleteSweepError as e:
        print(f"ERROR (exit 3): {e}", file=sys.stderr)
        return 3
    except R16eH4InvariantError as e:
        print(
            f"ERROR (exit 2): H4 ARCHITECTURAL FAILED: {e}\n"
            f"\nRidge × N-seed produced DIFFERENT predicted_returns.npy SHA-256 "
            f"hashes within a (model, return_type) cell. This refutes the Phase A.3 "
            f"REDESIGN architectural lock — Ridge is supposed to be RNG-FREE.\n"
            f"Check `lob-models/src/lobmodels/models/simple/temporal_ridge.py:102` "
            f"`Ridge(alpha=...)` should have no `random_state` argument. If sklearn "
            f"Ridge implementation has changed, update the parity test and re-derive.",
            file=sys.stderr,
        )
        return 2

    if args.json_output:
        print(json.dumps(outcome_to_json_dict(outcome), indent=2))
    else:
        print(render_verdict(outcome, cell_results, ratio_results))

    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
