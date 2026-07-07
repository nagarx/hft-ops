# hft-ops Experiment Ledger — DORMANT (frozen-in-practice since 2026-05-20)

**What this is.** The subprocess-orchestrator's per-experiment record store:
`records/<run_id>.json` (one per training/sweep/calibration/analysis run),
the `index.json` envelope (fast query projection, `INDEX_SCHEMA_VERSION`-gated
auto-rebuild), and `runs/<name>_<ts>/` (config snapshot + logs + diagnostics).
All three are gitignored runtime data (auto-rebuilt by code) — see `.gitignore`.

**Status: FROZEN IN PRACTICE.** Last record written 2026-05-20 (newest
experiment `created_at` 2026-05-19T22:56:50Z, `record_type: sweep_aggregate`).
No new records since the 2026-06/07 discovery arc moved research OFF the
orchestrator lane. `record_type` across all 186 records: 168 training /
15 sweep_aggregate / 2 calibration / 1 analysis — **zero `DISCOVERY`**
(`RecordType.DISCOVERY` was added to hft-contracts v2.10.0 for a discovery
ledger lane but is UNUSED).

**Where experiments are registered NOW (the live surfaces).**
1. `hft-wiki/FINDINGS_MASTER_REGISTER.md` — the authoritative research register
   (query via the `hft-wiki` CLI; findings cite their verdict-JSON SSoT).
2. Per-harness `results/*.json` verdict JSONs (glbx_/xsec_/nvda_/opra_/pead_/
   crypto_/multiday_ discovery harnesses).
3. The read-only `hft-ops monitor` fuses both surfaces.

**Tracked snapshot.** A one-time copy of the 186 records + `index.json`
(frozen 2026-07-07, this file's date) lives at
`ledger/archive/2026-07-07-snapshot/` so the orchestrator-era provenance
survives a fresh clone even though the live store is gitignored. Do NOT
hand-edit records; do NOT delete the snapshot.

**Revival condition.** If a future arc PROMOTES a validated discovery approach
back into the contract + orchestrator + sweep machinery (root `CLAUDE.md`
"Promote a validated approach…"), this ledger becomes live again with no
teardown — the store is append-only. Re-enable simply by running
`hft-ops run <manifest>` (writes new `records/*.json`) and, if needed,
`hft-ops ledger rebuild-index`.
