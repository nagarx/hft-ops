"""DiscoveryVerdictReader — reads the intraday-discovery harness verdicts
(``<tree>/**/results/*.json``) and normalizes each via the shared
``discovery_verdict`` adapters into ``Verdict``s.

Torch-free: imports ONLY ``discovery_verdict`` (stdlib-only) + stdlib. Read-only:
it touches no harness code or output (the harnesses don't expose importable
verdict builders and don't import each other; the monitor is a pure reader/adapter).
"""

from __future__ import annotations

import json
from pathlib import Path

from discovery_verdict import KNOWN_ADAPTERS, Verdict, normalize_verdict


class DiscoveryVerdictReader:
    """Reads ``<repo_root>/<tree>/**/results/*.json`` -> ``list[Verdict]``.

    Skips internals (``_``-prefixed) + a filename denylist (feature dumps, re-run
    snapshots). ``include_gate_outs=False`` drops ``GATED_OUT`` verdicts. A per-file
    parse/normalize failure is recorded into ``read_errors`` and skipped, never raised.
    """

    DISCOVERY_TREES = (
        "glbx_discovery",
        "xsec_equity_discovery",
        "nvda_discovery",
        "opra_discovery",
        "pead_discovery",
    )
    SKIP_PREFIXES = ("_",)
    DEFAULT_DENYLIST = frozenset({"gex_features.json", "gate_rerun_2026_06_19.json"})

    def __init__(
        self,
        repo_root: Path | str,
        *,
        adapters=KNOWN_ADAPTERS,
        include_gate_outs: bool = True,
        extra_denylist=(),
    ):
        self.repo_root = Path(repo_root)
        self.adapters = adapters
        self.include_gate_outs = include_gate_outs
        self.denylist = set(self.DEFAULT_DENYLIST) | set(extra_denylist)
        self.read_errors: list[tuple[str, str]] = []

    def read_all(self) -> list[Verdict]:
        self.read_errors = []
        verdicts: list[Verdict] = []
        for tree in self.DISCOVERY_TREES:
            base = self.repo_root / tree
            if not base.is_dir():
                continue
            for path in sorted(base.glob("**/results/*.json")):
                name = path.name
                if any(name.startswith(p) for p in self.SKIP_PREFIXES):
                    continue
                if name in self.denylist:
                    continue
                try:
                    raw = json.loads(path.read_text())
                    verdict = normalize_verdict(
                        raw, source_path=str(path), source_tree=tree, adapters=self.adapters
                    )
                except Exception as exc:  # malformed JSON / unmatched shape -> skip + record
                    self.read_errors.append((str(path), f"{type(exc).__name__}: {exc}"))
                    continue
                if not self.include_gate_outs and verdict.verdict == "GATED_OUT":
                    continue
                verdicts.append(verdict)
        verdicts.sort(key=lambda v: (v.source_tree, v.probe_id, v.source_path))
        return verdicts
