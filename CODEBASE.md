# hft-ops — Codebase Reference

> **Pipeline scope (2026-06-02).** This module is part of an **intraday trading research pipeline** — an experiment-first platform for discovering and validating *any* profitable **intraday** trading edge (no overnight positions), across approach classes (microstructure/HFT, scalping, intraday momentum, intraday statistical arbitrage, …) and instruments (equities, futures, same-day options). The pipeline *originated* as a high-frequency NVDA MBO/LOB microstructure system — that origin explains the "HFT" / "LOB" / "MBO" naming here — and that microstructure-direction program is now one (largely-closed) track among many. **Names are historical; the mission is general.** This module's role: the experiment orchestrator — a manifest-driven, subprocess-based multi-stage runner (extraction → analysis → IC-gate → training → signal-export → backtesting) with a JSON ledger, fingerprint dedup, the FeatureSet registry, sweep manifests, and content-addressed caching; the "control panel" for reproducible experiments (extending it to new approach classes is additive — register §9). For the full mission + approach taxonomy + capability-readiness boundary, see root `CLAUDE.md` §Research Scope & Charter (+ `CROSS_ASSET_OFI_FINDINGS_AND_ISSUES_2026_06_01.md` §9).

> **Version**: 0.3.0-dev | **Schema**: 3.0 (Phase G G.6.A bump 2.2 → 3.0 MAJOR per CLAUDE.md root rule: any modification to stable features 0-97 = BREAKING) | **Tests**: 1054 pass (Phase 8B envelope auto-rebuild + Phase 8A scheduler + Phase 8C-α post-stage artifact routing + Phase V.A.4 trust-column harvest + R-NN cycle hygiene cumulative; +421 since 633 banner). | **Last Updated**: 2026-05-21 (Cycle A-rev #PY-341 doc-hygiene bundle — banner refresh per validation cycle Wave 2K finding)
>
> **Phase 7.5 SHIPPED (2026-04-23)** — orchestrator integration gaps closed; first `hft-ops validate` + `hft-ops run --dry-run` successful in pipeline history:
>   - **Task 1a** (`680ba77`): 3 HMHP manifests re-pointed from archived `export_hmhp_signals.py` (Phase 6 6D fossil) to canonical `lob-model-trainer/scripts/export_signals.py`.
>   - **Task 1c** (`03c8b71`): Script-path convention unified across `SignalExportRunner` + `BacktestRunner` — the `<stage>_dir / stage.script` pattern was DOUBLING the prefix; now uses `config.paths.resolve(stage.script)` matching pipeline-root-relative convention of `extraction.config` / `data.data_dir` / `stage.checkpoint`.
>   - **Phase 7.5-A** (`0913253`): `SignalExportRunner._resolve_signal_export_config` 3-tier cascade (Priority 1: `stages.signal_export.config` escape hatch; Priority 2: `<training.output_dir>/config.yaml` auto-persisted by `train.py:318 save_config(...)`; Priority 3: `stages.training.config` legacy wrapper fallback) + `--experiment → --config` arg-protocol migration + `split="train"` rejection at validate-time + enabled-stage cross-check validator + 14 new tests in `tests/test_signal_export_config_resolver.py`.
>   - **Phase 7.5-B.1** (`75dcd6c`): `BacktestRunner` cmd reconstruction — dropped 7 dead flags never accepted by any current backtester script (`--experiment`, `--data-dir`, `--signals-dir`, `--horizon-idx`, `--params-file`, `--spread-bps`, `--slippage-bps`, `--threshold`, `--no-short`, `--device`); renamed `--signals-dir → --signals` + `--spread-bps → --max-spread-bps` to match current scripts' argparse; added `--signals` / `--name` / `--manifest` required flags; documented `stage.extra_args` as the operator-facing escape hatch for script-specific flags (`--min-agreement`, `--zero-dte`, `--exchange ARCX`, etc.).
>   - **Phase 7.5-B.2** (`4ce938b`): Schema deprecation docstrings for 8 dead fields — `BacktestParams.{slippage_bps, threshold, no_short, device}` + `BacktestingStage.{model_checkpoint, data_dir, horizon_idx, params_file}` — all marked DEPRECATED with 2026-10-31 removal calendar. Fields retained for 2 existing manifests' back-compat (`nvda_hmhp_tb_volscaled.yaml` + `e5_60s_importance_audit.yaml`); runner ignores.
>   - **Phase 7.5-B.3** (`3dd3ccb`): `_build_backtesting` loader emits `RuntimeWarning` on unknown top-level YAML keys (silent-drop class bug surfaced by 5-agent final validation round). Prior loader silently dropped `readability:` / `holding:` / `costs:` blocks declared by 3 production HMHP manifests — operator-facing impact: ARCX manifest `costs: {exchange: ARCX}` silently ran on XNAS cost model. Closes hft-rules §8 "never silently drop without recording diagnostics." 7 new tests in `tests/test_loader_backtesting_unknown_keys.py`.
>
> **Phase V.1 SHIPPED (2026-04-21)** — correctness + observability hardening from the 5-agent Phase V post-audit:
>   - **L1.1**: `hft-feature-evaluator/.github/workflows/test.yml` — closes the V.A.2 "5 repos × CI" off-by-one claim. 7 of 7 monorepo repos now have CI.
>   - **L1.2**: New `ExperimentRecord.signal_export_output_dir: Optional[str]` field (hft-contracts) + harvest wiring in `stages/signal_export.py` + attachment in `cli.py::_record_experiment`. `statistical_compare._resolve_signal_dir` now prefers the run-time-captured absolute path over manifest re-parsing — closes Agent 2 H1 manifest-move-resilience gap.
>   - **L1.3**: `compare_sweep_statistical` + `hft-ops sweep compare` now raise with per-record NaN-breakdown when drop-fraction exceeds `max_drop_frac` (default 5%). Closes Agent 2 H2 silent-data-drop gap per hft-rules §8.
>   - **L2.2**: `comparator.diff_experiments` return dict includes `compatibility_fingerprint: Optional[Tuple[Optional[str], Optional[str]]]` surfacing V.A.4 trust-column divergence. Closes Agent 4 H3.
>   - **L2.4**: `contract_preflight._INPUT_CONTRACTS` now exposed as `MappingProxyType` (read-only view); mutation attempts raise TypeError.
>   - **L2.5**: Reverse-drift detection — `test_table_does_not_contain_deleted_models` asserts `table ⊆ registry`.
>   - **L2.6**: AST-based `test_contract_preflight_module_imports_are_torch_free` regression — defensive lock on the hft-ops torch-free invariant for Phase VI snapshot migration.
>   - **L3.1/L3.2/L3.3**: `statistical_compare.compare_sweep_statistical` return type EXPANDED to 3-tuple `(List[PairwiseResult], List[str], CompareSweepDiagnostics)`. Diagnostics carry `n_treatments`, `n_samples_paired`, `n_samples_raw`, `n_dropped_nonfinite`, `drop_fraction`, `n_bootstraps`, `block_length`, `primary_horizon_idx`, `metric`. CLI renders summary panel + per-pair `nf_repl` column (yellow/red thresholds for CI-point-mass warnings) consuming `PairwiseResult.n_nonfinite_replaced` (hft-metrics v0.1.7 addition). Label-uniqueness check fails loud on axis_values collisions.
>   - Dep pin: `hft-metrics>=0.1.6` (via v0.1.7 additive release; the `n_nonfinite_replaced` field is backward-compatible at the frozen-dataclass level).
>
> **Phase V SHIPPED (2026-04-21)** — Experiment Trust + Velocity Closure cycle (V.A.1 + V.A.2 + V.A.4 + V.A.5 + V.A.8 MVP + V.B.1 + V.B.4a + V.B.4b). Details in root CLAUDE.md §"Last verified" + `pipeline_contract.toml` v2.24 cumulative entry. New hft-ops modules / CLI added in Phase V:
>   - `hft_ops.stages.contract_preflight` (V.A.8 MVP) — InputContract pre-flight; hardcoded `_INPUT_CONTRACTS` synced from live `lobmodels.ModelRegistry`, wrapped in `TrainingRunner.run` try/except → StageResult(FAILED) with gate_report. Phase VI will replace with snapshot-file architecture.
>   - `hft_ops.ledger.statistical_compare` (V.B.4b) — `compare_sweep_statistical` adapter consuming `hft_metrics.pairwise_paired_bootstrap_compare`; paired-bootstrap CI + BH q-values for sweep child records. Surfaces via `hft-ops sweep compare <sweep_id>` CLI.
>   - `stages/signal_export.py::_harvest_compatibility_fingerprint` (V.A.4) — mirror of `_harvest_feature_set_ref`; populates `ExperimentRecord.compatibility_fingerprint` for `ledger list --compatibility-fp <hex>` trust-column query.
>   - `experiments/sweeps/seed_stability.yaml` (V.B.1) — single-axis `train.seed ∈ {42..46}` template; prerequisite for V.B.4 statistical tooling.
>
> **Phase 8A.1 Part 1 SHIPPED (2026-04-20)**: Parallel-sweep scheduler INFRASTRUCTURE (CLI surface + primitives + SWEEP_FAILURE contract + failure taxonomy). The sweep_run loop-body wire-in lands in Part 2. `hft_ops.scheduler` package adds `resources.py` (`ResourceSpec` frozen dataclass + `GPUSemaphore` filelock-backed — exclusive-GPU coordination across workers + SIGKILL-survivable via fcntl.flock kernel semantics), `executor.py` (`WorkerPoolExecutor` ThreadPoolExecutor-backed MVP since workers are subprocess-bound → GIL released → real parallelism; `WorkerResult` + `WorkerStatus` + `OnFailureMode` enum + `parse_on_failure` CLI-grammar parser; `TRANSIENT_ERROR_KINDS` / `FATAL_ERROR_KINDS` taxonomy; `BrokenProcessPool` single-attempt recovery defensive branch), `signal_handler.py` (`SubprocessPidTracker` thread-safe set + `install_scheduler_signal_handler` context-manager with SIGTERM→5s→SIGKILL cascade + prior-handler restore for pytest safety). CLI flags on `hft-ops sweep run`: `--parallel N`, `--on-failure continue|abort|retry:N`, `--gpus "0,1"|none|auto`, `--cpu-budget N` (all parsed + validated; Part 1 falls back to sequential with WARN on `--parallel > 1`, Part 2 wires into dispatch). `--continue-on-failure` flag deprecated (removal 2026-10-31). `INDEX_SCHEMA_VERSION` 1.1.0 → 1.2.0 (MINOR additive): new `ExperimentRecord.sweep_failure_info` field + `index_entry()` projection. `RecordType.SWEEP_FAILURE` enum variant. `dedup.py::check_duplicate` filters SWEEP_FAILURE records so retries re-run (critical correctness: without filter, single failure blocks all retries). `filelock>=3.12` + `packaging>=23.0` declared as runtime deps. +17 new hft-ops tests (460→477). See plan file `~/.claude/plans/fuzzy-discovering-flask.md` §Phase 8A.1 Rev 2 + TOML changelog v2.17.
>
> **Phase 8A.0 SHIPPED (2026-04-20)**: Content-addressed extraction cache at `data/exports/_cache/<64hex>/`. Cache-key hashes 9 orthogonal inputs (resolved config + extractor git SHA + Cargo.lock SHA + reconstructor git SHA + hft-statistics git SHA + raw-input manifest hash + compiled binary SHA + platform target + contract_version) so behavior tracks both TREATMENT and BUILD ENVIRONMENT. Platform-dispatch linking: clonefile (macOS APFS) / reflink (Linux Btrfs+XFS) / hardlink-chmod-readonly (Linux ext4) / relative-symlink (cross-fs). Full-file SHA-256 validation per hit (per-process memoized). Atomic populate + atomic tree replication via `.tmp/` staging. `_CACHE_HIT.json` sidecar in output_dir for operator debugging. LRU-by-mtime GC with size-budget + pin-list. `ExtractionStage.skip_if_exists=false` emits `DeprecationWarning` (Phase 9 removal target Q3 2026). `hft-ops cache {ls,gc,pin,unpin}` P0 CLI group. `OpsConfig.cache_extraction: bool = True` default-on; `--cache-extraction`/`--no-cache-extraction` top-level CLI flag. `INDEX_SCHEMA_VERSION` 1.0.0 → 1.1.0 (MINOR additive): new `ExperimentRecord.cache_info` field + `index_entry()` projection extended with cache_info. Fingerprint-stable — cache outcome is observation, not treatment (Invariant 4; locked by `test_extraction_cache.py::TestFingerprintStabilityAcrossCache` asserting the 5 cache-only inputs are DISJOINT from `dedup.py::compute_fingerprint`'s surface). +13 hft-ops tests (447→460); golden key-set 21→22. See `[extraction_cache]` TOML section, plan file `~/.claude/plans/fuzzy-discovering-flask.md` Rev 3.
>
> **Phase 8B SHIPPED (2026-04-20)**: Ledger `index.json` envelope + auto-rebuild eliminates the silent-omission class when `index_entry()` whitelist grows. `_load_index` 4-branch detection (envelope-match / version-mismatch / legacy-bare-list / malformed-JSON); auto-rebuilds on the latter three with loud WARN log. `--strict-index` CLI flag + env auto-detect (`HFT_OPS_STRICT_INDEX=1` OR `CI=true` case-insensitive) elevates to fail-fast. `StaleLedgerIndexError(LedgerError(ValueError))` exception hierarchy. `dedup.py::check_duplicate` routes through ExperimentLedger (BUG-1 fix — was direct `json.load` bypass). +19 hft-ops tests (428→447); +3 hft-contracts tests (303→306).
>
> **Phase 7 Stage 7.4 Round 4 (SHIPPED 2026-04-20)**: Post-audit adversarial pass closed three latent Round-1-through-3 gaps and one new BLOCKER: (a) **C1-complete** — `TrainingRunner._capture_training_metrics` now iterates the full regression val_* taxonomy (5 max + 3 min imported from `post_training_gate` SSoT) + classification keys (`val_accuracy`, `val_macro_f1`, `val_signal_rate`) with NaN/Inf finite-guards, replacing the Round-1 3-key subset that silently dropped regression metrics; (b) **Gate-report surfacing (Option C)** — `cli.py::_record_experiment` now harvests via a generic loop reading `captured_metrics["gate_report"]` from every stage and stores under `ExperimentRecord.gate_reports[stage_name]` — Phase 8 `post_backtest_gate` plugs in with zero schema change; `validation.py` and `post_training_gate.py` both emit under the uniform `"gate_report"` key (was `"validation_report"` / `"post_training_gate"`); lazy `from_dict` migration shim lifts pre-Round-4 nested records with 2026-08-01 removal deadline; (c) **Atomic `_save_index`** — tmp+fsync+`os.replace` eliminates the transient bad-state window that would force full `_rebuild_index` on the next load. 29 new unit tests (17 `_capture_training_metrics` coverage + 12 `gate_reports` integration/migration/fingerprint-stability). Test counts 399→428.
>
> **Phase 7 Stage 7.4 Post-Training Regression Detection Gate (SHIPPED 2026-04-19)**: New `PostTrainingGateStage` dataclass + `PostTrainingGateRunner` (`stages/post_training_gate.py`) wired between `training` and `signal_export`. Three automated checks: (a) floor check (primary metric ≥ min_metric_floor, rule §13); (b) prior-best-ratio check (matching `(model_type, labeling_strategy, horizon_value)` signature via ledger scan); (c) cost-breakeven check (regression only, informational, IBKR 1.4 bps Deep ITM). Three dispositions: warn (default, annotate + proceed) / abort (raise StageFailure) / record_only (silent). Default `enabled=False` (opt-in). Prerequisites bundled: `TrainingRunner._capture_training_metrics` extended to merge `test_metrics.json` (regression scalars) + list-of-dicts `training_history.json`. Post-validation (2026-04-19): degenerate-signature guard in `_find_prior_best_experiment` prevents false-positives when match_signature has no meaningful fields. 43 new unit tests. hft-ops tests 356→399.
>
> **Phase 7 Stage 7.1 FeatureSet Registry Activation (SHIPPED 2026-04-18)**: `scripts/migrate_feature_presets_to_registry.py` (data-prep utility) created 3 production FeatureSet JSONs at `contracts/feature_sets/`. 3 hft-ops manifest migrations from `feature_preset:` to `feature_set:` (`nvda_hmhp_40feat_xnas_h10.yaml`, `nvda_hmhp_128feat_arcx_h10.yaml`, `nvda_hmhp_128feat_xnas_h10.yaml`) closing the 2026-08-15 ImportError deadline.
>
> **Phase 6 Post-Audit Hardening (2026-04-17)**: Stage 1 (11 correctness fixes 6A.1-11) + Stage 2 (4 per-phase backlog commits O1-O4 capturing Phase 0/1/2/4 work) + Stage 3 (5 contract-plane primitives moved to `hft_contracts.*` with re-export shims preserving legacy imports — `feature_sets/{schema,hashing}.py`, `provenance/lineage.py`, `ledger/experiment_record.py` are now shims; `feature_sets/{writer,registry,producer}` stay hft-ops-side). Post-validation (2026-04-18): H1 sweep `_rebuild_index` regression fixed via O(N)-scan in-place index update; M1 DRY `_CONTENT_HASH_RE` — harvester now imports from `hft_contracts.signal_manifest` SSoT.
>
> **Phase 5 FULL-A SHIPPED (2026-04-17)**: Cross-stage override routing in sweeps, strategy registry shell, fingerprint coverage extension, `SweepAggregateWriter` module, variable resolution post-expansion, 3 MVP templates.
>
> **Phase 5 Preview + Phase 2b + Phase 3**: `TrainingStage.trainer_config` inline alternative; `ValidationStage` + `ValidationRunner` (IC gate / `fast_gate` library); 5 critical bug fixes from Phase 2b; compute_fingerprint resolves `_base:` inheritance before hashing (Phase 3 §3.3b).
>
> **New in 0.2.0**: Sweep/grid expansion (`SweepConfig`, `expand_sweep()`), sweep CLI commands (`hft-ops sweep expand/run/results`), `sweep_id`+`axis_values` on `ExperimentRecord`, shared `utils.py`, `training_config` population in `_record_experiment`.

Central experiment orchestrator for the intraday trading research pipeline (origin: HFT microstructure). Defines, validates, runs, tracks, and compares experiments across all 7 pipeline modules. Supports parameter sweeps (grid search) from a single YAML manifest.

---

## 1. Architecture Overview

```
Experiment Manifest (YAML)
    │
    ├─ manifest/loader.py      Parse + resolve ${...} variables
    ├─ manifest/validator.py   Cross-module config validation
    │
    │  Stage runners (_build_stage_runners order; subprocess unless noted):
    ├─ stages/extraction.py       → cargo run --bin export_dataset
    ├─ stages/raw_analysis.py     → MBO-LOB-analyzer/scripts/run_analysis.py
    ├─ stages/dataset_analysis.py → lob-dataset-analyzer/scripts/run_analysis.py
    ├─ stages/validation.py       → IC gate (in-process: hft_evaluator.fast_gate)
    ├─ stages/training.py         → lob-model-trainer/scripts/train.py
    ├─ stages/post_training_gate  → regression-detection gate (in-process; opt-in)
    ├─ stages/signal_export.py    → lob-model-trainer/scripts/export_signals.py
    ├─ stages/backtesting.py      → lob-backtester backtest script
    │  (stages/contract_preflight.py — in-process preflight inside training; NOT a registered stage)
    │
    ├─ scheduler/              Parallel sweep exec + content-addressed extraction cache (Phase 8A)
    ├─ feature_sets/           FeatureSet registry (Phase 4 producer/writer/registry)
    ├─ provenance/lineage.py   Capture git hash, config hash, data hash
    ├─ ledger/dedup.py         Fingerprint-based duplicate detection
    ├─ ledger/ledger.py        Append-only JSON-backed storage
    ├─ ledger/comparator.py    Cross-experiment comparison + ranking
    └─ monitor/                Read-only ledger × discovery-verdict × drift surface (F5)
```

### Key Invariants

1. **No internal imports**: hft-ops never imports lobtrainer, lobanalyzer, or lobbacktest. It invokes modules as subprocesses via their CLI interfaces.
2. **No parameter duplication**: Parameters like feature_count, window_size, stride are validated against the extractor config but never re-specified in the manifest.
3. **Validation-first**: All cross-module consistency checks happen BEFORE any computation.
4. **Append-only ledger**: Records are immutable after creation. Only `notes` can be updated.
5. **Fingerprint dedup**: SHA-256 of (resolved config + data manifest + contract version). Prevents accidental repeated experiments.

---

## 2. Module Reference

### 2.1 paths.py

`PipelinePaths(pipeline_root: Path)` — resolves all module paths from a single root.

| Property | Path |
|----------|------|
| `extractor_dir` | `{root}/feature-extractor-MBO-LOB` |
| `trainer_dir` | `{root}/lob-model-trainer` |
| `backtester_dir` | `{root}/lob-backtester` |
| `raw_analyzer_dir` | `{root}/MBO-LOB-analyzer` |
| `dataset_analyzer_dir` | `{root}/lob-dataset-analyzer` |
| `contract_toml` | `{root}/contracts/pipeline_contract.toml` |
| `ledger_dir` | `{root}/hft-ops/ledger` |

`auto_detect()` walks up from the package location to find the pipeline root.

### 2.2 manifest/schema.py

Dataclasses defining the experiment manifest structure:

| Class | Purpose |
|-------|---------|
| `ExperimentManifest` | Top-level container (+ optional `sweep: SweepConfig`) |
| `ExperimentHeader` | name, description, hypothesis, contract_version, tags |
| `ExtractionStage` | config path, output_dir, skip_if_exists |
| `RawAnalysisStage` | profile, symbol, analyzers |
| `DatasetAnalysisStage` | profile, split, analyzers |
| `TrainingStage` | config path **or** inline `trainer_config` dict (exactly-one-of), overrides dict, horizon_value. Inline dict supports `_base: str \| list[str]` multi-base composition; paths resolved relative to `<trainer_dir>/configs/` via `_absolutize_inline_base_paths` (not the manifest directory — bases live under trainer configs), materialised to a temp YAML by `_materialize_inline_config` before being consumed by `ExperimentConfig.from_yaml`. |
| `BacktestingStage` | model_checkpoint, params (BacktestParams) |
| `SweepConfig` | name, strategy ("grid"), axes list |
| `SweepAxis` | axis name, list of SweepAxisValue |
| `SweepAxisValue` | label, overrides (dotted-key dict) |

### 2.2b manifest/sweep.py

Sweep/grid expansion logic:

| Function | Purpose |
|----------|---------|
| `validate_sweep(config)` | 8 validation checks: name, strategy, labels, conflicts |
| `expand_sweep(manifest)` | Cartesian product of axes → `List[ExperimentManifest]` |

Expansion merges axis overrides into `stages.training.overrides` (trainer-YAML keys) or applies directly to `TrainingStage` fields (manifest-level keys like `horizon_value`). Cross-axis key conflicts are detected and rejected.

### 2.3 manifest/loader.py

`load_manifest(path, now=None) -> ExperimentManifest`

Parses YAML, resolves `${...}` variables in multiple passes:
- `${experiment.name}` — manifest header fields
- `${stages.extraction.output_dir}` — cross-stage references
- `${timestamp}` / `${date}` — execution time
- `${resolved.*}` — deferred: resolved at runtime by stage runners

### 2.4 manifest/validator.py

`validate_manifest(manifest, paths) -> ValidationResult`

Validation checks:
1. `contract_version` matches `hft_contracts.SCHEMA_VERSION`
2. Referenced config files exist
3. Cross-module consistency: feature_count, window_size, stride, labeling_strategy
4. `horizon_value` resolves to valid index in extractor's horizons array
5. Existing exports have valid metadata
6. Logical stage dependencies (training needs data source, backtest needs checkpoint)

### 2.5 stages/

Each stage runner implements: `validate_inputs()`, `run()`, `validate_outputs()`. `_build_stage_runners()` (in `cli.py`) is the single dispatch-order SSoT shared by `run` and `sweep run`; a stage runs only when `manifest.stages.<name>.enabled`. The registered stages, in order (subprocess stages invoke a sibling-repo CLI with no Python imports; two gate stages run **in-process**):

| Stage | Runner | How it runs |
|-------|--------|-------------|
| extraction | `ExtractionRunner` | subprocess → `cargo run --release --bin export_dataset --features parallel -- --config ...` (consults the extraction cache first — §2.9) |
| raw_analysis | `RawAnalysisRunner` | subprocess → `MBO-LOB-analyzer/scripts/run_analysis.py` |
| dataset_analysis | `DatasetAnalysisRunner` | subprocess → `lob-dataset-analyzer/scripts/run_analysis.py` |
| validation | `ValidationRunner` | **in-process** — the mandatory IC gate; calls `hft_evaluator.fast_gate.run_fast_gate` as a **library** (Phase 2b: library-over-subprocess for lower latency + simpler tests) |
| training | `TrainingRunner` | subprocess → `lob-model-trainer/scripts/train.py`; runs `contract_preflight` in-process first (see below) |
| post_training_gate | `PostTrainingGateRunner` | **in-process** — regression-detection gate (floor / prior-best-ratio via `ExperimentLedger.filter()` / cost-breakeven); default `enabled=False`, opt-in |
| signal_export | `SignalExportRunner` | subprocess → `lob-model-trainer/scripts/export_signals.py` |
| backtesting | `BacktestRunner` | subprocess → `lob-backtester` backtest script |

**Not a registered stage**: `stages/contract_preflight.py` — an **in-process** InputContract pre-flight invoked inside `TrainingRunner.run`. It validates the resolved trainer config against `_INPUT_CONTRACTS` (a read-only `MappingProxyType` mirror of the live `lobmodels.ModelRegistry`) and fails the training stage with a `gate_report` on a contract violation. The module is AST-locked torch-free (Phase VI snapshot-migration guard). Support modules: `stages/base.py` (shared runner base) + `stages/_override_discipline.py` (sweep-override SSoT helper).

Training stage: accepts either `config: <path>` (legacy) or inline `trainer_config: <dict>` (Phase 2b — exactly-one-of). Applies manifest overrides to the effective config, resolves `horizon_value` to `horizon_idx` from export metadata, materialises the effective config to disk when inline (`_materialize_inline_config`) after rewriting any relative `_base:` paths against the manifest directory (`_absolutize_inline_base_paths`).

### 2.6 provenance/lineage.py

| Function | Returns |
|----------|---------|
| `capture_git_info(repo_dir)` | `GitInfo` (commit_hash, branch, dirty) |
| `hash_file(path)` | SHA-256 hex of file contents |
| `hash_config_dict(config)` | SHA-256 hex of sorted JSON serialization |
| `hash_directory_manifest(dir)` | SHA-256 hex of (filename, size) pairs |
| `build_provenance(root, ...)` | `Provenance` with all captured info |

### 2.7 ledger/

**ExperimentRecord** — immutable record (Phase 6 6B.1a: canonical home is now
`hft_contracts.experiment_record`; this `hft_ops.ledger.experiment_record`
module is a re-export shim with removal deadline 2026-10-31):
- Identity: `experiment_id`, `fingerprint`, `manifest_path`
- Provenance: `Provenance` (git hash, config hashes, data hash, timestamp)
- Config snapshot: `extraction_config`, `training_config`, `backtest_params`
- Results: `training_metrics`, `backtest_metrics`, `dataset_health`
- Metadata: `tags`, `hypothesis`, `notes`, `status`, `stages_completed`
- **Phase 4 4c.4 (2026-04-16)**: `feature_set_ref: Optional[Dict[str, str]]`
  — `{name, content_hash}` propagated from trainer `signal_metadata.json`
  when `DataConfig.feature_set` is set; `None` otherwise
- **Phase 7 Stage 7.4 Round 4 (2026-04-20)**: `gate_reports: Dict[str, Dict[str, Any]]`
  — generic cross-stage gate-report surface keyed by stage name
  (`"validation"`, `"post_training_gate"`, future `"post_backtest_gate"`);
  replaces the Round 1 pattern of nesting post-training gate output under
  `training_metrics["post_training_gate"]`; fingerprint-stable (gate outcomes
  are observations, never hashed by `compute_fingerprint`). Pre-Round-4
  records migrate lazily via `ExperimentRecord.from_dict` shim with removal
  deadline 2026-08-01
- **Sweep metadata** (Phase 5): `sweep_id`, `axis_values`, `record_type`
  (`"training"` | `"sweep_aggregate"` | `"analysis"` | ...),
  `sub_records` (for aggregates), `parent_experiment_id`

**ExperimentLedger** — append-only JSON-backed storage:
- `register(record)` — store new record, update index
- `get(id)` — load full record
- `filter(tags=, model_type=, min_f1=, ...)` — query index
- `find_by_fingerprint(fp)` — dedup lookup
- `update_notes(id, notes)` — only mutable operation

**compute_fingerprint(manifest, paths)** — SHA-256 of outcome-affecting config (strips metadata like name, description, tags, output paths). **Resolves `_base:` inheritance in trainer YAMLs before hashing** (Phase 3, see §2.7b below) so base mutations correctly invalidate dependent-experiment fingerprints.

**compare_experiments(entries, metric_keys, sort_by, ...)** — sorted comparison table. `diff_experiments(a, b)` — config + metric diff between two records.

### 2.7b Fingerprint Resolution (Phase 3)

`compute_fingerprint` routes trainer YAMLs (both file-based `stages.training.config:` and inline `stages.training.trainer_config:`) through `lobtrainer.config.merge.resolve_inheritance` before hashing, so the fingerprint is computed over the **resolved effective dict** rather than the pre-inheritance raw YAML.

**Implementation**:

| Helper | Purpose |
|--------|---------|
| `_load_trainer_merge_module()` | Loads `lobtrainer.config.merge` via `importlib.util.spec_from_file_location` without importing the torch-dependent `lobtrainer` package. Cached per-process. Keeps the hft-ops venv free of torch. Falls back to raw-YAML loading if the loader is unavailable (matches the pre-existing fail-safe pattern). |
| `_load_trainer_config_resolved(path)` | File-based path. Loads the YAML, then calls `resolve_inheritance(data, path)`. |
| `_resolve_inline_trainer_config(data, manifest_path, paths)` | Inline path. Deep-copies the inline dict, then calls `resolve_inheritance` using `<trainer_dir>/configs/` as the base directory for relative `_base:` references (matches the runtime `_absolutize_inline_base_paths` convention in `stages/training.py`; bases live under trainer configs, not next to the manifest). |

**Error policy** (hard vs soft):

| Error class | Behaviour | Rationale |
|-------------|-----------|-----------|
| `ValueError` (cycle, depth, malformed `_base`) | **Propagates** | These are configuration bugs the user must fix; silent-success would mask them. |
| `FileNotFoundError` (referenced `_base:` file missing) | **Propagates** | Same reasoning. |
| `OSError` (I/O glitch on base file) | **Warn + return empty dict** | Matches pre-existing `_load_config_as_dict` fail-safe; keeps fingerprinting robust to transient filesystem issues. |
| `YAMLError` (malformed YAML in a base) | **Warn + return empty dict** | Same as `OSError`. |

**Why this matters**:

Pre-fix, `_load_config_as_dict` called `yaml.safe_load` directly without resolving inheritance. Since every Phase 3 experiment child is thin (`_base: [...]` + a handful of overrides), mutating a shared base (e.g., `bases/train/regression_default.yaml: epochs 30 → 40`) left every dependent experiment's fingerprint **unchanged** — pre/post-change ledger records were silently conflated under one fingerprint. This was a pre-existing bug (the five E5 configs already used `_base:` before Phase 3) that Phase 3 materially worsened by making every migrated config thin.

Post-fix fingerprints are **content-addressed** over the resolved effective dict:
- Changing a base file produces a different fingerprint for every dependent experiment.
- `config: legacy.yaml` and `trainer_config: <inline equivalent>` produce **identical** fingerprints when their resolved dicts match.

Regression guard: `tests/test_fingerprint_base_mutation.py` (5 tests):
1. Mutating a base changes the dependent-experiment fingerprint.
2. Path-based vs inline equivalents produce identical fingerprints.
3. Cycles raise `ValueError` (not a silent empty dict).
4. Depth overruns raise `ValueError`.
5. Malformed `_base:` values (wrong type) raise `ValueError`.

### 2.8 feature_sets/

The Phase 4 FeatureSet registry — content-addressed selection artifacts at `contracts/feature_sets/<name>.json` bridging the evaluator (producer) and the trainer (consumer). Flat submodule layout so callers write `from hft_ops.feature_sets import FeatureSet`:

| Submodule | Responsibility |
|-----------|----------------|
| `hashing` | content-addressed SHA-256 over `(sorted-unique indices, source_feature_count, contract_version)` |
| `schema` | `FeatureSet` / `FeatureSetRef` / `FeatureSetAppliesTo` / `FeatureSetProducedBy` dataclasses + validators |
| `writer` | atomic JSON writes (refuse-overwrite + idempotent-on-match) |
| `registry` | read-side queries (`list` / `get` / `verify`) |
| `producer` | evaluator → FeatureSet orchestration |

`schema` + `hashing` co-moved to `hft_contracts.feature_sets.*` in Phase 6; the package re-exports from those canonical homes (`writer` / `registry` / `producer` stay hft-ops-side). Consumed by the trainer's `resolve_feature_set`; surfaced via the `hft-ops feature-sets` group + `hft-ops evaluate --save-feature-set` (§5).

### 2.9 scheduler/ — parallel sweep execution + extraction cache (Phase 8A)

A CROSS-stage concern (parallelism dispatches stages; cache reuse spans grid points), so it lives outside `stages/`. Five modules by responsibility:

| Module | Responsibility |
|--------|----------------|
| `executor` | `WorkerPoolExecutor` — thread-backed concurrent grid-point dispatch (threads, not processes: workers are subprocess-bound, so the GIL is released during subprocess I/O). Applies the `--on-failure` policy (`continue` / `abort` / `retry:N`) + a transient-vs-fatal error taxonomy. |
| `resources` | `ResourceSpec` + `GPUSemaphore` — a `filelock`-backed exclusive-GPU lock that survives worker crashes (`fcntl.flock` releases on FD close per POSIX). |
| `signal_handler` | `SubprocessPidTracker` + a scoped SIGINT handler: on Ctrl-C, cascade SIGTERM → grace → SIGKILL to every tracked subprocess PID; restores the prior handler on exit (pytest-safe). |
| `sweep_dispatch` | `run_grid_point_stages` — the pure per-grid-point stage runner shared by the sequential and parallel `sweep run` paths (no ledger writes). |
| `extraction_cache` | content-addressed extraction cache at `data/exports/_cache/<64hex>/` (populate / resolve_or_link / gc). |

**Load-bearing invariants** (do not break when touching sweep execution or caching):

1. **Single-writer**: workers NEVER touch the ledger — they return `WorkerResult` dicts; only the parent serializes `_record_experiment` writes (and the final `sweep_aggregate` record). Dedup is centralized in the parent; workers never query the ledger for dedup.
2. **GPU-semaphore scope**: `run_grid_point_stages` holds the `GPUSemaphore` only around the GPU-using stages (`training` + `signal_export`); non-GPU stages run without it, maximizing cross-worker parallelism.
3. **Thread-count injection**: the parent computes `per_worker_threads = cpu_budget / n_parallel` and injects `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` per worker so the Rust extractor + Python trainer do not oversubscribe physical cores.
4. **Cache key = treatment + build environment**: the key hashes the resolved (post-override, post-`_base`) extractor config **and** the build environment (extractor / reconstructor / hft-statistics git SHAs + `Cargo.lock` + compiled-binary SHA + platform target + `contract_version` + raw-input manifest) — so a code/dep change invalidates a cache that a filename check would miss. Platform-dispatch linking (APFS clonefile / Btrfs+XFS reflink / ext4 hardlink-readonly / cross-fs symlink) + per-hit full-file SHA validation; finalized entries are chmod-readonly.
5. **Cache outcome is an OBSERVATION, not a treatment** (Invariant 4): the hit/miss outcome is recorded on the record as `cache_info` but never enters `dedup.py::compute_fingerprint` — a cache hit vs miss must not change the fingerprint. The cache-only build-environment inputs (the git/binary/platform SHAs that distinguish a hit from a miss) are DISJOINT from the fingerprint surface. `ResourceSpec` (GPU/CPU execution policy) likewise never enters the fingerprint.

For phase/version/CLI detail see the Phase 8A ship-banners above, the `[extraction_cache]` section of `contracts/pipeline_contract.toml`, and `CHANGELOG.md` (do not hand-copy counts here). CLI: `hft-ops cache {ls,gc,pin,unpin}` + the `--parallel` / `--on-failure` / `--gpus` / `--cpu-budget` flags on `sweep run` (§5).

### 2.10 Discovery integration — monitor/ + the discovery ledger lane (F5)

The bridge between hft-ops and the current research arc: every discovery probe (the `glbx_discovery/` / `xsec_equity_discovery/` / `nvda_discovery/` / … harnesses) emits a normalized verdict JSON, and this subsystem makes those verdicts queryable, fingerprinted, and drift-monitored alongside training runs — one experiment × verdict × provenance × drift surface.

**Read side — `monitor/` (READ-ONLY, torch-free).** A pure reader/adapter over BOTH the experiment ledger AND the discovery-harness `<tree>/**/results/*.json` verdicts (each normalized via the shared root-level `discovery_verdict` adapters). Invariants: it NEVER writes the ledger, rebuilds the index, or touches harness code; and no module here imports torch / lobmodels / lobtrainer at module scope (locked by an AST scan + a `sys.modules` runtime sentinel). Five modules:

| Module | Responsibility |
|--------|----------------|
| `ledger_reader` | projects `ledger/records/*.json` (via `ExperimentRecord.index_entry()`) into flat `LedgerRow`s; resolves one headline metric per `record_type`; skip-malformed (records read errors, never raises) |
| `discovery_reader` | reads the discovery trees' `results/*.json` → normalized `Verdict`s; skips `_`-prefixed internals + a filename/glob denylist (sharded data caches co-located in `results/`); a per-file parse failure is recorded, never raised |
| `drift` | four **observational** drift checks (never auto-fix): stale index-envelope vs disk, fingerprint divergence (same name → ≠ config fingerprints, or ≠ provenance hashes under one fingerprint), stale-stats-version verdict, contract/schema-version mismatch |
| `table` | `build_monitor_table` — fuses ledger rows + verdicts into one filterable `MonitorRow` surface annotated with drift flags; **collapses** a probe that is BOTH a registered ledger record AND an on-disk verdict onto their shared `config_sha256`/fingerprint so it is not double-counted |
| `render` | `render_text` / `render_markdown` / `render_json` (table) + `render_drift_text` / `render_drift_json` (drift report) |

The set of discovery tree names is a code constant in `discovery_reader.py` (`DISCOVERY_TREES`), and the file/glob denylists live there too — read them in code for the live lists rather than trusting any enumeration here.

**Write side — `ledger/discovery_record.py::record_from_verdict`.** Adapts a normalized `Verdict` (or a raw harness dict, normalized here via the same `discovery_verdict` adapters) into a fingerprinted `ExperimentRecord` with `record_type="discovery"` (`RecordType.DISCOVERY`). Construction delegates to the `hft_contracts.experiment_recorder.record_from_artifacts` SSoT (never hand-rolled). **The fingerprint IS the probe config hash** (`provenance.config_sha256`, the probe's treatment identity) — it FAILS LOUD (`FingerprintNormalizationError`) if that is not a 64-hex lowercase SHA-256, so distinct probes never conflate (the dedup Phase-3 §3.3b class). The verdict string + rails (`any_tradeable_edge`, `power_class`, `mde`, `family_fwer_p`, DSR, …) are OBSERVATIONS: they land on `training_metrics` / `notes` and NEVER enter any fingerprint input. A discovery probe structurally lacks the four Phase Y trust components, so it composes with `require_complete_provenance=False`.

**CLI**: `hft-ops monitor table` (the unified surface, with `--kind` / `--source-tree` / `--edge-only` / `--status` / `--name-contains` / `--format {text,markdown,json}` filters) and `hft-ops monitor drift` (`--fail-on warn|error` for a CI hook) — both lazy-import the torch-free `monitor/` modules so they never pull the torch import path. Separately, `hft-ops ledger backfill --record-type discovery` adds a discovery-typed record from a metadata-only retroactive manifest — a **distinct** write path that does not call `record_from_verdict`.

### 2.11 cli.py — navigation (a #PY-121 god-object)

`cli.py` is the single largest file in the module (root CLAUDE.md flags it as a `#PY-121` god-object; run `wc -l`). Its organizing structure, to orient without scrolling the whole file:

- **Top-level `main` group commands**: `run`, `validate`, `compare`, `diff`, `check-dup`, `evaluate`.
- **Sub-groups**: `ledger` (`list` / `show` / `search` / `rebuild-index` / `fingerprint-explain` / `backfill`), `sweep` (`expand` / `run` / `results` / `compare`), `feature-sets` (`list` / `show`), `cache` (`ls` / `gc` / `pin` / `unpin`), `monitor` (`table` / `drift`).
- **Shared helpers**: `_build_stage_runners(manifest)` — the stage dispatch-order SSoT (§2.5), used by both `run` and `sweep run`; `_record_experiment(...)` — the post-run harvest loop that assembles + registers the `ExperimentRecord` (captures training metrics, harvests every stage's `gate_report` + the extraction `cache_info`, and passes the signal_export captured metrics to the `record_from_artifacts` SSoT for trust-column harvest — `feature_set_ref` / `compatibility_fingerprint` / `signal_export_output_dir`).
- **Parallel split**: the `sweep run --parallel > 1` branch is extracted into `cli_parallel_sweep.py::run_sweep_parallel`, which preserves the single-writer invariant (§2.9) — workers run `run_grid_point_stages`; the parent serializes all ledger writes + the final `sweep_aggregate`.

---

## 3. Data Flow

Each stage runs only when `manifest.stages.<name>.enabled` (so the chain is a superset — analysis/gate/export stages are commonly skipped):

```
hft-ops run manifest.yaml
    │
    ├─0─ VALIDATE:         parse → resolve vars → check contract → cross-validate configs → dedup check
    ├─1─ EXTRACT:          cargo run (extraction-cache lookup; skip-if-exists + metadata validates)
    ├─2─ ANALYZE RAW:      python run_analysis.py (optional)
    ├─3─ ANALYZE DATASET:  python run_analysis.py (optional)
    ├─4─ IC GATE:          validation runner (in-process fast_gate; warn/abort/record_only)
    ├─5─ TRAIN:            contract_preflight → resolve horizon_idx → apply overrides → python train.py
    ├─6─ POST-TRAIN GATE:  regression-detection gate (in-process; opt-in, default disabled)
    ├─7─ SIGNAL EXPORT:    python export_signals.py
    ├─8─ BACKTEST:         python backtest_deeplob.py
    └─9─ RECORD:  build ExperimentRecord → harvest gate_reports + cache_info + trust columns → fingerprint → write to ledger
```

Parallel sweeps dispatch the same per-grid-point stage chain via `scheduler/` (§2.9); the F5 `monitor/` surface (§2.10) reads the resulting ledger records + discovery verdicts read-only.

---

## 4. Contract Surfaces

| Surface | Source | Consumed by |
|---------|--------|-------------|
| `SCHEMA_VERSION` | `pipeline_contract.toml` | validator (contract_version check) |
| Feature counts (40+8+36+14=98) | `pipeline_contract.toml` | validator (cross-module check) |
| Horizons array | extractor TOML `[labels].max_horizons` | training runner (horizon_value resolution) |
| Export metadata | `{day}_metadata.json` | training runner (feature_count, window_size) |
| Normalization stats | trainer output | backtester (denormalization) |

---

## 5. CLI Commands

| Command | Purpose |
|---------|---------|
| `hft-ops run <manifest>` | Run complete experiment pipeline |
| `hft-ops run <manifest> --dry-run` | Validate without executing |
| `hft-ops run <manifest> --stages extraction,training` | Run specific stages |
| `hft-ops validate <manifest>` | Validate manifest only |
| `hft-ops compare --metric ... --sort ...` | Compare experiments |
| `hft-ops diff <id1> <id2>` | Detailed diff |
| `hft-ops ledger list` | List all experiments |
| `hft-ops ledger show <id>` | Show full record |
| `hft-ops ledger search --tags ... --min-f1 ...` | Search ledger |
| `hft-ops ledger rebuild-index [--dry-run]` | Re-project all records through the current `index_entry()` whitelist (Phase 7.4 Round 4, commit `6ba4e93`). Needed after hft-contracts whitelist expansions; see `ExperimentRecord.index_entry()` for current fields. |
| `hft-ops ledger fingerprint-explain <manifest.yaml>` | Show the inputs `compute_fingerprint` hashed (Phase 4 4c.3) — debugging dedup decisions |
| `hft-ops ledger backfill <manifest> --record-type <type>` | Retroactively register a historical experiment (E1-E16-style pre-orchestrator runs) with `provenance.retroactive=True`. `--record-type` ∈ {training, analysis, calibration, backtest, evaluation, sweep_aggregate, **discovery**}. Optional `--metrics-file` / `--parent-id` / `--status` / `--notes`. |
| `hft-ops check-dup <manifest>` | Check for duplicates |
| `hft-ops sweep expand <manifest>` | Dry expansion showing grid points |
| `hft-ops sweep run <manifest>` | Execute all grid points. Scheduler flags (§2.9): `--parallel N`, `--on-failure continue\|abort\|retry:N`, `--gpus "0,1"\|none\|auto`, `--cpu-budget N` (`--continue-on-failure` deprecated, removal 2026-10-31). |
| `hft-ops sweep results <sweep_id>` | Compare results from a sweep |
| `hft-ops sweep compare <sweep_id>` | Paired moving-block-bootstrap significance across a sweep's arms (BH q-values); `compare_sweep_statistical` adapter over `hft_metrics.pairwise_paired_bootstrap_compare` (Phase V.B.4b) |
| `hft-ops feature-sets list` | List registry entries at `contracts/feature_sets/` |
| `hft-ops feature-sets show <name>` | Show full FeatureSet JSON |
| `hft-ops evaluate --save-feature-set <name>` | Produce a FeatureSet from an evaluator run (Phase 4 Batch 4b) |
| `hft-ops cache {ls,gc,pin,unpin}` | Manage the content-addressed extraction cache (§2.9): list / LRU-by-mtime GC with size-budget / pin / unpin |
| `hft-ops monitor table` | Unified read-only experiment × verdict × provenance × drift table (§2.10); filters `--kind` / `--source-tree` / `--edge-only` / `--status` / `--name-contains` / `--format {text,markdown,json}` |
| `hft-ops monitor drift` | Report ledger/verdict drift (stale index envelope, fingerprint divergence, stale stats-version, schema mismatch); `--fail-on warn\|error` exit-code hook for CI |

---

## 6. Testing

**433 tests** at HEAD (post Phase 7 Stage 7.4 Round 6 post-push-audit fix —
added `_build_post_training_gate` loader + 4 regression tests + 1 parity
test iterating `fields(Stages)` to prevent the class of silent-drop bug).

Test file inventory (30 files, grouped by concern). For exact per-file counts
run `pytest --collect-only -q`; the figures below are anchor points from
incremental phase landings:

| Group | Tests (approx.) | Coverage |
|---|---|---|
| **Manifest schema + loader** — `test_manifest_schema.py`, `test_validator.py` | 45 + 16 | Schema defaults, YAML loading, variable resolution, Stages parity, post_training_gate loader (Round 6 regression) |
| **Ledger + dedup** — `test_ledger.py`, `test_dedup.py`, `test_fingerprint_base_mutation.py`, `test_fingerprint_feature_set_*.py`, `test_fingerprint_hard_fail.py`, `test_ledger_rebuild_index.py`, `test_cli_fingerprint_explain.py`, `test_experiment_record_feature_set_ref.py` | 80+ | CRUD, fingerprint determinism, §2.7b resolved-dict hashing, feature_set normalization, rebuild-index CLI, gate_reports persistence |
| **Provenance** — `test_provenance.py` | 20+ | Hashing, git info, Provenance building, shim back-compat |
| **Sweep** — `test_sweep.py`, `test_sweep_cross_stage_routing.py`, `test_sweep_axis_values_preserved.py`, `test_sweep_aggregate_writer.py`, `test_variable_resolution_post_expansion.py`, `test_sweep_templates.py` | 70+ | Grid expansion, cross-stage routing, aggregate writer, post-expansion resolver, MVP templates |
| **Stage runners** — `test_validation_stage.py`, `test_post_training_gate.py`, `test_signal_export_harvest.py`, `test_training_capture_metrics.py` | 120+ | IC gate, PostTrainingGateRunner (3 checks × 3 dispositions), feature_set_ref harvest, C1-complete regression-metric capture |
| **FeatureSet** — `test_feature_sets.py`, `test_feature_sets_producer.py`, `test_feature_sets_registry_walkup.py`, `test_feature_sets_writer.py` | 60+ | Schema, hashing, producer, registry walk-up, atomic writer |
| **Phase 2b regression guards** — `test_bugfixes_phase2b.py` | 16 | Compat banner, inline-base absolutization |
| **Post-ship CLIs** — `test_ledger_rebuild_index.py`, `test_cli_fingerprint_explain.py` | 14 | Round 4 post-validation CLI additions |

```bash
.venv/bin/python -m pytest tests/ -v     # expect 433 passed
```

---

## 7. Dependencies

| Package | Purpose |
|---------|---------|
| `hft-contracts` | Contract constants and validation |
| `click` | CLI framework |
| `pyyaml` | YAML manifest parsing |
| `rich` | Terminal tables and formatting |
| `tomli` (Python <3.11) | TOML config parsing |
