"""hft_ops.monitor (F5) — a read-only, torch-free monitor over BOTH the hft-ops
experiment ledger AND the intraday-discovery harness verdicts.

Exposes one queryable experiment x verdict x provenance x drift surface so the
discover-many-experiments mission stays traceable + monitorable. It READS only —
never writes the ledger or any harness output, never rebuilds the index.

TORCH-FREE INVARIANT (root CLAUDE.md §Module Technical Map): no module here may
import torch / lobmodels / lobtrainer at module scope (which would pull 700+ torch
modules via lobmodels.__init__ and silently break hft-ops' torch-free design).
Locked by tests/test_monitor_torch_free.py (AST scan + runtime sys.modules sentinel).
"""
# Public surface is added in lock-step with the TDD GREEN steps
# (ledger_reader -> discovery_reader -> drift -> table -> render).
