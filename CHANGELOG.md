# Changelog

All notable changes to the `hft-ops` experiment orchestrator are documented
in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) for the Python package; the
cross-module **contract schema version** is tracked independently by the
producer `hft-contracts.SCHEMA_VERSION`.

---

## [0.3.0-dev] — in progress

### Backlog Hygiene Bundle (2026-05-13) — #PY-191 + #PY-93-PARTIAL

**Changed (#PY-93-PARTIAL — xgboost_lob preflight constraint table sync)**

- `src/hft_ops/stages/contract_preflight.py::_INPUT_CONTRACTS_RAW` adds
  `xgboost_lob` entry mirroring live
  `lobmodels.ModelRegistry.get('xgboost_lob').input_contract`.
  lob-models commit `ef17bb4` registered the model 2026-05-09 via Phase VI
  snapshot architecture but this constraint table was not synced per the
  CLAUDE.md Change-Coordination Checklist row "Add a new model architecture".
  Closes the forward-drift `test_table_covers_live_registry` failure that
  4 prior cycles cited as "pre-existing #PY-93 xgboost_lob registry preflight
  failure UNRELATED".
- **STATUS:CLOSED-PARTIAL** — symptom-only. Architectural root cause
  (wire `ParamOwnership.validate` at `lobmodels/registry/core.py:281`)
  remains OPEN as Sub-cycle 5 candidate per PHASE_P_BACKLOG #PY-93.
- **Test delta**: `TestConstraintTableSanity::test_table_covers_live_registry`
  4/5 → 5/5 PASS. Full hft-ops suite: 987 pass + 1 fail → 988 pass + 0 fail.

**Removed (#PY-191 dedup site — orphan hashlib import)**

- `src/hft_ops/ledger/dedup.py:16` — orphan `import hashlib` removed.
  Zero callers post-#PY-41 closure 2026-05-07 (`hft_contracts.canonical_hash`
  SSoT migration). Companion site at
  `hft-feature-evaluator/src/hft_evaluator/pipeline.py:22` closed in
  parallel commit. **Test delta**: zero new tests (cosmetic).

---

### Phase Y / γ-1 LITE / #PY-95 + #PY-96 + #PY-97 (2026-05-10 evening) — `diff_experiments` + `ledger_show` trust-column surface + EXPERIMENT_GUIDE docs

**Added (#PY-95 — `comparator.diff_experiments` 4-source provenance divergence)**

- `src/hft_ops/ledger/comparator.py::diff_experiments` extended return dict
  with 2 new keys mirroring the existing `compatibility_fingerprint`
  pattern at L172-180:
  - `experiment_provenance_hash`: 4-source composer fingerprint divergence.
    `None` when records agree; `Tuple[Optional[str], Optional[str]]` when
    they differ.
  - `model_config_hash`: model-axis identity divergence (filtered
    `model.params` SHA-256). Read from
    `(record.training_config or {}).get("model_config_hash")` per the
    Bundle Commit 1+2 nested-storage convention.

**Added (#PY-96 — `cli.py::ledger_show` trust-column display)**

- New "Trust-column Fingerprints" section displays compat-fp / epH / mch
  when any populated (truncated to first 16 hex chars). Section omitted
  for legacy pre-V.A.4 records carrying None.

**Added (#PY-95 consumer — `cli.py::diff` trust-column divergence display)**

- New "Trust-column Fingerprint Divergence" section in `hft-ops diff
  <id_a> <id_b>` output displays only the fingerprints that differ
  between the two records (sourced from `comparator.diff_experiments`
  return). Section omitted when all 3 fingerprints agree.

**Documentation (#PY-97 — `EXPERIMENT_GUIDE.md` trust-column queries)**

- Rewrote "CompatibilityContract Fingerprint Filter" section as
  "Trust-column Fingerprint Filters" covering all 3 fingerprints
  (compat-fp + epH + mch) with example queries:
  ```bash
  hft-ops ledger list --compatibility-fp <64-hex>     # signal-boundary contract
  hft-ops ledger list --provenance-hash <64-hex>      # 4-source composer identity
  hft-ops ledger list --model-config-hash <64-hex>    # model arch + hyperparams
  ```

**Tests added (9 new tests in `tests/test_comparator_fingerprint_diff.py`)**

- `TestExperimentProvenanceHashDiff` (4 tests): matching/differing/
  both-unset/asymmetric divergence cases.
- `TestModelConfigHashDiff` (5 tests): matching/differing/both-missing/
  asymmetric/unrelated-keys-ignored cases.

(File now has **14 tests total**: 5 pre-existing `TestCompatibilityFingerprintDiff`
+ 9 new — verified by `pytest --collect-only`.)

**Test counts**: **811 passed + 1 deselected** (was 802 + 1 — net +9 tests).

**Cross-cycle bundle context**

Consumer-side bundle (sites in hft-ops) for the Phase Y / γ-1 LITE
close-out triggered by lob-model-trainer's #PY-88 Phase 2 sklearn
return_type axis closure. The 3-fingerprint trust-column display is
symmetric with the post-Bundle-Commits-1+2 filter capability
(`hft-ops ledger list --model-config-hash <hex>` shipped in commit
`097e83c` 2026-05-10 morning).

---

### HYBRID Phase α-1.3 (2026-05-10) — caller-side cycle-detection invariant + α-3 walk-up class (#PY-83-cluster)

**Fixed (#PY-83-cluster — α-1.3 follow-up to α-1.1 + α-1.2)**

α-1.2 in lob-model-trainer flipped 4 sites in `merge.py` + `schema.py` to
`.absolute()` for cycle-detection consistency. Adversarial cycle-level audit
2026-05-10 (per Option D mandate) found **8 caller-side sites** in hft-ops
that pass pre-`.resolve()`'d paths to `merge_mod.resolve_inheritance` —
violating the α-1.2 "all sites must flip together" invariant. This commit
flips the 8 sites:

- `src/hft_ops/ledger/dedup.py:502` — `path.resolve()` → `.absolute()`
  (caller-side of resolve_inheritance from `_resolve_trainer_inheritance`)
- `src/hft_ops/ledger/dedup.py:556` — `(paths.trainer_dir / "configs").resolve()`
  → `.absolute()` (trainer_configs root for inline trainer_config dispatch)
- `src/hft_ops/stages/training.py:178` — exact mirror of dedup.py:502
- `src/hft_ops/stages/training.py:249` — `(trainer_configs_root / p).resolve()`
  → `.absolute()` (rewritten _base values re-enter resolve_inheritance)
- `src/hft_ops/stages/training.py:401` — same pattern as dedup.py:556
- `src/hft_ops/stages/contract_preflight.py:326` — exact mirror
- `src/hft_ops/manifest/loader.py:531` — `Path(manifest_path).resolve()` →
  `.absolute()` (α-3 walk-up class — scans ancestors for
  `contracts/pipeline_contract.toml`; if manifest under symlinked checkout,
  walk would jump off-tree)
- `scripts/migrate_feature_presets_to_registry.py:129` — same α-3 walk-up
  class in migration script

**Why bundled together**: per Agent C 2026-05-10 audit recommendation, sites
1-6 form a single tightly-coupled cluster (caller-side of α-1.2
cycle-detection contract). Sites 7-8 share the α-3 walk-up class. Single
atomic commit avoids partial-flip regression class.

**Impact**: post-α-1.2, recursion inside `resolve_inheritance` correctly
preserves symlink-source. But callers passing pre-`.resolve()`'d paths
caused the cycle-detection key (line 135) to land in `_seen` as a deref'd
path while subsequent recursion levels used `.absolute()` (preserved).
For symlinked-configs deployments, this would silently break either
cycle-detection or relative-base resolution. β-1 Cycle 4 RE-RUN succeeded
because the user's deployment has symlinked `data/` (not `configs/`), but
this latent defect would surface in Cycle 5 multi-arm sweeps via different
manifest layouts.

**Tests**: 800 passed, 31 warnings, 0 failures (no count change; existing
tests cover the corrected behavior because no symlink in test fixtures).

**Discovered by**: Agent C of Option D adversarial cycle audit 2026-05-10
("Missed Path.resolve sites sweep"). Pre-impl design verified by Agent C
itself (recommendation: single α-1.3 commit with all 8 sites). Mid-impl
+ pre-commit gates per saved feedback memory.

**Remaining (a)-class sites — KEPT `.resolve()` by design**:
`paths.canonical()` (escape hatch by design), `extraction_cache.py` cache
keys, `feature_sets/producer.py` lineage manifests, `_testing.py`
require_monorepo_root helper.

### HYBRID Phase α-1.1 (2026-05-10) — `paths.resolve()` symlink-deref defect (#PY-83) + α-1/α-2 catch-up

**Fixed (#PY-83 — α-1.1 hotfix)**

- `PipelinePaths.resolve(rel)` no longer dereferences symlinks. Pre-α-1.1
  it called `Path.resolve()` which derefs the `data/` symlink at start;
  in deployments where `data/` is symlinked to an external mount (e.g.
  `data/` → `/Volumes/WD_Black/HFT-data/`), the rebased path produced by
  α-1's `_maybe_rebase_path` (loader.py:103) was a 5-level cross-mount
  relpath like `'../../../../../Volumes/WD_Black/HFT-data/exports/...'`
  that the trainer subprocess could not interpret. **DEFEATED α-1's
  #PY-78 fix for substitution-based manifests in the user's actual
  deployment.** Sister defect to #PY-79 (closed by α-3 in
  `lob-model-trainer/feature_set_resolver.py:442`).
- Sites flipped from `Path.resolve()` to `Path.absolute()` (preserves
  symlink-source lineage; no FS access; never derefs):
  - `paths.py:30` (`PipelinePaths.__post_init__` self-resolve)
  - `paths.py:117-119` (`PipelinePaths.resolve()` body — affects 47+ callers)
  - `paths.py:185` (`PipelinePaths.auto_detect()` walk-up via `__file__`)
  - `cli.py:194` (user-supplied `--pipeline-root` value)
  - `config.py:60` (`OpsConfig.from_pipeline_root` user input)

**Added**

- `PipelinePaths.canonical(rel) -> Path` — explicit escape hatch for the
  rare cases that genuinely need the deref'd canonical filesystem path
  (cache-key inputs, content-addressed hashes, lineage manifests where
  symlink-equivalence must collapse). Default consumers should use
  `resolve()`. `extraction_cache.py` (3 sites) + `feature_sets/producer.py`
  (3 sites) call `(extractor_dir / x).resolve()` directly (NOT via
  `paths.resolve()`) — they continue to work as-is for cache-key purposes.
- 8 NEW tests:
  - `tests/test_py83_paths_resolve_canonical.py` (7 tests across 4 classes)
    — locks resolve preserves / canonical derefs / divergence-only-for-
    symlinks / auto_detect symlink-source preservation
  - `tests/test_py78_path_relativity_rebasing.py::TestRebasePreservesSymlinkSourceForSubstitution::test_substitution_through_symlinked_data_dir_stays_intra_monorepo`
    — production-critical regression locking the α-1 fix in symlinked-data
    deployments

**BREAKING (semantic; pre-release internal API)**

- `paths.resolve()` semantics changed from "deref + absolute" to "preserve
  symlink-source + absolute". External callers (none in tree) that
  depended on deref behavior must migrate to `paths.canonical()`.

**Discovered by**: 8-agent prep round 2026-05-10 (Agent I FINDING 1,
"Hidden Findings Hunt"). Pre-impl design verified by feature-dev:code-
architect agent; mid-impl + pre-commit adversarial gates per saved
feedback memory.

### HYBRID Phase α-2 (2026-05-10) — signal_export argparse + universal stderr (#PY-80)

**Fixed (#PY-80)**

- Orchestrator `signal_export` stage was failing in 0.1s with argparse
  exit code 2 because `e5_60s_importance_audit.yaml` (and other
  manifests) lacked an explicit `checkpoint:` field; the empty-string
  default propagated to `--checkpoint` and `argparse` rejected it before
  the trainer subprocess could begin. Manual run with the same args
  succeeded — signaling a manifest gap, NOT a code bug.
- Validate-time gate at `stages/signal_export.py::validate_inputs` now
  fail-loud rejects empty `stage.checkpoint` when stage is enabled, citing
  the canonical `${stages.training.output_dir}/checkpoints/best.pt` path
  pattern. Per hft-rules §5 (fail-fast with precise error).
- `experiments/e5_60s_importance_audit.yaml` adds explicit
  `checkpoint: "${stages.training.output_dir}/checkpoints/final.pt"`
  under signal_export.

**Added**

- NEW `_format_subprocess_failure(proc, script_basename, *,
  stderr_tail_lines=20)` helper at `stages/base.py` — tails last 20
  stderr lines into the `error_message` field. Uniformly consumed by all
  6 stage runners (extraction, raw_analysis, dataset_analysis, training,
  signal_export, backtesting). Eliminates the prior pattern where each
  runner constructed its own error_message ad-hoc.
- 8 NEW tests in `tests/test_py80_subprocess_failure_diagnostics.py`.

### HYBRID Phase α-1 (2026-05-10) — orchestrator path-relativity contract (#PY-78)

**Fixed (#PY-78)**

- Orchestrator did NOT relativize `${stages.extraction.output_dir}`
  (monorepo-root-relative) when substituted into trainer-cwd-relative
  slots like `data.data_dir`. Result: trainer subprocess `cwd=lob-model-
  trainer/` failed to find data at `data/exports/...` (which resolves to
  `lob-model-trainer/data/exports/...` — non-existent). Symptom: training
  stage fails with `Cannot resolve horizons: split directory does not
  exist: data/exports/...`.

**Added**

- NEW `manifest/slot_taxonomy.py` (~125 LoC) — `PathBase` enum
  (`PIPELINE_ROOT` / `TRAINER_CWD` / `NONE`) + `detect_slot_path_base(slot_dotted_key)`
  classifier. 23 regex patterns ordered TRAINER_CWD before PIPELINE_ROOT
  (more-specific first); 21 PIPELINE_ROOT slots + 4 TRAINER_CWD slots.
- NEW `_maybe_rebase_path(resolved_str, source_key, target_key, paths)`
  helper at `manifest/loader.py:56-108`. Rebases when `src=PIPELINE_ROOT,
  tgt=TRAINER_CWD` via `paths.resolve() + os.path.relpath(..., trainer_dir)`.
  HIGH-1 absolute-path short-circuit + HIGH-2 fail-loud `ValueError` with
  actionable slot-key context per hft-rules §8.
- `_resolve_variables` extended with `paths: Optional[PipelinePaths]`
  kwarg + `current_dotted_path` recursion via `_substitute(value,
  current_dotted_path)`. Walk-up auto-detect via
  `manifest_path.parents`.
- 36 + 16 = 52 NEW tests in `tests/test_slot_taxonomy.py` +
  `tests/test_py78_path_relativity_rebasing.py`. Cycle 4 manifest
  fingerprint UNCHANGED at `fed14a3efffa1297...` (uses literal paths).

**Note (caveat to operators)**

- 7 substitution-based production manifests will fingerprint-rotate
  post-α-1 (architecturally CORRECT — pre-α-1 fingerprints represented
  broken state). Operators with stored compat_fps may see "duplicate"
  prevention misses on re-run. Cycle 4 manifest is unaffected (literal
  paths). Documented in PHASE_P_BACKLOG.md (Task #285).

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
