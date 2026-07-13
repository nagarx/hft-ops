"""DiscoveryVerdictReader — reads the intraday-discovery harness verdicts
(``<tree>/**/results/*.json``) and normalizes each via the shared
``discovery_verdict`` adapters into ``Verdict``s.

Scope: the reader scans EXACTLY the trees in ``DISCOVERY_TREES`` (the 7
root-allowlisted harnesses; ``crypto_discovery``/``multiday_discovery`` added
in curation Phase 3, 2026-07-07) — it does NOT "fuse every harness on disk".
A ``SKIP_PATH_PARTS`` guard drops any results-glob hit under a ``.venv``/
``site-packages`` path component BEFORE parsing: ``crypto_discovery/.venv``
embeds third-party test fixtures matching ``**/results/*.json`` (statsmodels)
that would otherwise inject phantom UNRESOLVED rows. ``pead_discovery`` is
design-only today (has never emitted a ``results/`` dir) and is kept in scope
as a zero-cost no-op so a future run is picked up without a code change.

Torch-free: imports ONLY ``discovery_verdict`` (stdlib-only) + stdlib. Read-only:
it touches no harness code or output (the harnesses don't expose importable
verdict builders and don't import each other; the monitor is a pure reader/adapter).
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from discovery_verdict import KNOWN_ADAPTERS, Verdict, normalize_verdict


class DiscoveryVerdictReader:
    """Reads ``<repo_root>/<tree>/**/results/*.json`` -> ``list[Verdict]``.

    Skips any hit under a ``.venv``/``site-packages`` path component
    (``SKIP_PATH_PARTS`` — embedded-virtualenv third-party fixtures, dropped
    BEFORE parsing) + internals (``_``-prefixed) + a filename denylist (feature
    dumps, re-run snapshots, SUPERSEDED/VOID verdicts, non-verdict result
    artifacts) + a glob denylist (sharded data caches like
    ``nvda_atm_iv.shard0of4.json`` that share the ``results/`` dir but are NOT
    verdicts). ``include_gate_outs=False`` drops
    ``GATED_OUT`` verdicts. A per-file parse/normalize failure is recorded into
    ``read_errors`` and skipped, never raised.
    """

    # The FULL scan scope — the 7 root-allowlisted harnesses. crypto_discovery/
    # + multiday_discovery/ added in curation Phase 3 (2026-07-07) together with
    # the SKIP_PATH_PARTS guard (crypto's embedded .venv would otherwise leak
    # statsmodels test fixtures into the scan). pead_discovery is design-only
    # (no results/ ever emitted) — scanned as a zero-cost no-op, not a blindspot.
    DISCOVERY_TREES = (
        "glbx_discovery",
        "xsec_equity_discovery",
        "nvda_discovery",
        "opra_discovery",
        "pead_discovery",
        "crypto_discovery",
        "multiday_discovery",
    )
    SKIP_PREFIXES = ("_",)
    # Path-component guard (checked FIRST, before any basename rule and before
    # parsing): a **/results/*.json hit whose path contains a `.venv` or
    # `site-packages` component is an embedded-virtualenv third-party file,
    # never a harness verdict. crypto_discovery/.venv ships statsmodels test
    # fixtures (influence_lsdiag_R.json, fit_ets_results*.json) matching the
    # scan glob; without this guard they render as phantom UNRESOLVED rows via
    # the CommonCoreAdapter catch-all. Component matching (exact dir-name
    # equality via Path.parts, not substring) — `my.venv-notes/` never false-hits.
    SKIP_PATH_PARTS = (".venv", "site-packages")
    # Filename denylist (basename match, checked before parsing). Three classes:
    #   1. Feature dumps / re-run snapshots — not verdicts.
    #   2. SUPERSEDED / VOID verdicts — ``variance_dl_verdict.json``
    #      (STOP-DL-SPANNED-BY-CARRIERS) is VOID per its sibling
    #      ``results/SUPERSEDED.md``: a crippled-model artifact (scale-stripping
    #      input z-score), NOT a real null (FINDING-110). The corrected verdict is
    #      ``variance_dl_v2_verdict.json`` (INDETERMINATE-DL-UNDERPOWERED-V2),
    #      which the reader serves normally. Serving the VOID row would surface a
    #      corrupt result on the fused monitor table.
    #   3. Non-verdict result artifacts co-located in scanned ``results/`` dirs
    #      (model freezes, IV/strike-grid data panels, descriptive curves). They
    #      have no top-level ``verdict`` key and would fall through to the
    #      CommonCoreAdapter catch-all as phantom UNRESOLVED rows.
    DEFAULT_DENYLIST = frozenset(
        {
            # class 1 — dumps / snapshots
            "gex_features.json",
            "gate_rerun_2026_06_19.json",
            # class 2 — VOID verdict (FINDING-110; see SUPERSEDED.md beside it)
            "variance_dl_verdict.json",
            # class 2 — obsolete-STOP pilot of the same VOID run (AUD-29,
            # 2026-07-13 ground-truth audit): carries STOP-DL-SPANNED-BY-CARRIERS
            # from the crippled v1 model; the live verdict is
            # variance_dl_v2_verdict.json (INDETERMINATE-DL-UNDERPOWERED-V2).
            # Rendering it would surface the superseded STOP on the fused table.
            "variance_dl_PILOT_verdict.json",
            # class 3 — non-verdict artifacts (phantom-UNRESOLVED guards)
            "composite_vrp_confront_env_gates.json",  # nvda composite_vrp_confront env-gate report
            "frozen_scale_model.json",  # nvda conditional_scale_variance model freeze
            "strike_grids.json",  # nvda expiry_friday_memo strike-grid data
            "nvda_0dte_iv.json",  # nvda iv_noninertness IV panel cache
            "nvda_0dte_atm_iv.json",  # multiday vrp_0dte ATM-IV panel cache (shape {"days": ...})
            "execution_timing_curve.json",  # xsec hks_periodicity descriptive curve
        }
    )
    # Glob denylist for sharded data caches co-located in a harness ``results/``
    # dir but which are NOT verdicts (shape ``{"days": ...}``, not underscore-
    # prefixed). e.g. iv_shadow's ``nvda_atm_iv.shard0of4.json`` cache — without
    # this they fall through to the CommonCoreAdapter catch-all and inject phantom
    # UNRESOLVED rows. The pattern is shard-count-agnostic (``*.shard*of*.json``).
    SKIP_GLOBS = ("*.shard*of*.json",)

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
            # DIRECT children of results/ ONLY — intentional (#PY-392, 2026-07-13):
            # nested results/ subdirs (e.g. variance_dl/results/oof_v2/) hold
            # ancillary evidence by construction (discovery_verdict/
            # ARTIFACT_POLICY.md Class 1), never verdict rows, and must not
            # render on the fused table. The root orphan guard
            # (scripts/ci/check_results_registered.py) diverges deliberately:
            # it scans results/ RECURSIVELY so nested evidence JSONs still get
            # the tracked+registered check.
            for path in sorted(base.glob("**/results/*.json")):
                if any(part in self.SKIP_PATH_PARTS for part in path.parts):
                    continue
                name = path.name
                if any(name.startswith(p) for p in self.SKIP_PREFIXES):
                    continue
                if name in self.denylist:
                    continue
                if any(fnmatch.fnmatch(name, g) for g in self.SKIP_GLOBS):
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
