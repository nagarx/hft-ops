# Changelog

All notable changes to the `hft-ops` experiment orchestrator are documented
in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) for the Python package; the
cross-module **contract schema version** is tracked independently by the
producer `hft-contracts.SCHEMA_VERSION`.

---

## [0.3.0-dev] — in progress

### Phase A.5.2 (2026-04-24) — BUG #1 timezone-naive cutoff comparison fix

**Fixed**

- `_harvest_compatibility_fingerprint` no longer silently returns pre-cutoff
  for non-UTC-offset `exported_at` timestamps that cross midnight. The
  pre-A.5.2 implementation compared ISO-8601 strings lexicographically
  (`exported_at >= FINGERPRINT_REQUIRED_AFTER_ISO`), which is correct ONLY
  when both sides carry matching zero-offset (`+00:00` / `Z`) or naive
  UTC semantics. A producer shipping from a non-UTC host (e.g., CDT
  `-05:00`) would emit manifests like `"2026-04-22T23:59:00-05:00"` which
  is strictly post-cutoff in UTC (`2026-04-23T04:59:00+00:00`) but
  lex-compares as pre-cutoff — silently suppressing the operator-facing
  post-Phase-A WARN for every one of those records. No ledger records
  were corrupted (the WARN is observability-only, not data-bearing); the
  fix restores the diagnostic signal that operators depend on to detect
  producer-path regressions.
- Harvester now routes through `hft_contracts.timestamp_utils.is_after_cutoff`
  — the new SSoT for ISO-8601 UTC-aware comparison introduced in
  hft-contracts 2.3.0. Both sides of the comparison are normalized to
  timezone-aware UTC datetimes before the `>=` check.

**Changed**

- On MALFORMED `exported_at` (non-parseable ISO-8601), the harvester now
  emits a DIAGNOSTIC WARN per hft-rules §8 ("never silently drop / clamp /
  'fix' data without recording diagnostics") citing the manifest path +
  offending value, then treats the record as pre-cutoff (conservative:
  cannot determine which side of the cutoff a malformed timestamp
  belongs to). Pre-A.5.2 behavior was silent-drop on the lex-compare
  False branch — new code surfaces producer drift to operators.
- Empty / non-string / absent `exported_at` remains silent pre-cutoff
  (legacy behavior preserved — these are pre-V.A.4 manifests, expected
  on historical records).

**Added**

- 4 new regression tests in `tests/test_signal_export_harvest.py::TestHarvestCompatFpTimezoneAware`:
    - `test_non_utc_offset_post_cutoff_triggers_warn` — THE BUG FIX.
      Pre-A.5.2 would have silently returned pre-cutoff for a CDT `-05:00`
      timestamp crossing midnight. Locks the post-cutoff WARN.
    - `test_naive_exact_cutoff_triggers_warn` — naive timestamps
      interpreted as UTC (hft-rules §3 canonical convention); inclusive
      `>=` semantics at the exact boundary.
    - `test_malformed_exported_at_does_not_crash` — malformed ISO-8601
      emits diagnostic WARN + treats as pre-cutoff (no crash, no
      post-Phase-A WARN since we can't determine cutoff side).
    - `test_z_suffix_post_cutoff_triggers_warn` — `Z` suffix
      post-cutoff exercises the `Z`→`+00:00` normalization path in
      `parse_iso8601_utc` (complements the existing `+00:00`-notation
      test in `TestHarvestCompatFpPhaseACutoff`).

**Dependencies**

- Pin bumped `"hft-contracts"` → `"hft-contracts>=2.3.0"`. Required
  because `hft_contracts.timestamp_utils.is_after_cutoff` first shipped
  in hft-contracts 2.3.0 (A.5.1). A pre-2.3.0 install would fail at
  module import time with ImportError — fail-fast per hft-rules §5.

**Migration notes for operators**

- No action required for existing ledger records. The fingerprint
  harvest logic remains purely observability (structural changes
  land in A.5.4 / A.5.7). Pre-A.5.2 behavior was a silent omission of
  diagnostic WARNs, not a data-corrupting bug.
- Operators running from non-UTC hosts (e.g., a CDT-local CI runner)
  will now see additional post-Phase-A WARNs for post-cutoff manifests
  that were previously misclassified as pre-cutoff. This is the INTENDED
  restored signal — cross-reference with `signal_metadata.json::exported_at`
  on each WARN'd record to identify the producer-path regression.

---

*Pre-0.3.0-dev changelog history is tracked via git log on main. See
`git log --oneline --all -- hft-ops/` for the full history.*
