# hft-ops — Codebase Reference

> **Version**: 0.2.0 | **Schema**: 2.2 | **Tests**: 93 | **Last Updated**: April 2026
>
> **New in 0.2.0**: Sweep/grid expansion (`SweepConfig`, `expand_sweep()`), sweep CLI commands (`hft-ops sweep expand/run/results`), `sweep_id`+`axis_values` on `ExperimentRecord`, shared `utils.py`, `training_config` population in `_record_experiment`.

Central experiment orchestrator for the HFT pipeline. Defines, validates, runs, tracks, and compares experiments across all 7 pipeline modules. Supports parameter sweeps (grid search) from a single YAML manifest.

---

## 1. Architecture Overview

```
Experiment Manifest (YAML)
    │
    ├─ manifest/loader.py      Parse + resolve ${...} variables
    ├─ manifest/validator.py   Cross-module config validation
    │
    ├─ stages/extraction.py    → cargo run --bin export_dataset
    ├─ stages/raw_analysis.py  → MBO-LOB-analyzer/scripts/run_analysis.py
    ├─ stages/dataset_analysis → lob-dataset-analyzer/scripts/run_analysis.py
    ├─ stages/training.py      → lob-model-trainer/scripts/train.py
    ├─ stages/backtesting.py   → lob-backtester/scripts/backtest_deeplob.py
    │
    ├─ provenance/lineage.py   Capture git hash, config hash, data hash
    ├─ ledger/dedup.py         Fingerprint-based duplicate detection
    ├─ ledger/ledger.py        Append-only JSON-backed storage
    └─ ledger/comparator.py    Cross-experiment comparison + ranking
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
| `TrainingStage` | config path, overrides dict, horizon_value |
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

Each stage runner implements: `validate_inputs()`, `run()`, `validate_outputs()`.

All stages invoke module CLIs as subprocesses (no Python imports):

| Stage | Command |
|-------|---------|
| extraction | `cargo run --release --bin export_dataset --features parallel -- --config ...` |
| raw_analysis | `python MBO-LOB-analyzer/scripts/run_analysis.py --profile ... --data-dir ...` |
| dataset_analysis | `python lob-dataset-analyzer/scripts/run_analysis.py --profile ... --data-dir ...` |
| training | `python lob-model-trainer/scripts/train.py --config ... --output-dir ...` |
| backtesting | `python lob-backtester/scripts/backtest_deeplob.py --experiment ...` |

Training stage: applies manifest overrides to a copy of the YAML config, resolves `horizon_value` to `horizon_idx` from export metadata.

### 2.6 provenance/lineage.py

| Function | Returns |
|----------|---------|
| `capture_git_info(repo_dir)` | `GitInfo` (commit_hash, branch, dirty) |
| `hash_file(path)` | SHA-256 hex of file contents |
| `hash_config_dict(config)` | SHA-256 hex of sorted JSON serialization |
| `hash_directory_manifest(dir)` | SHA-256 hex of (filename, size) pairs |
| `build_provenance(root, ...)` | `Provenance` with all captured info |

### 2.7 ledger/

**ExperimentRecord** — immutable record with:
- Identity: experiment_id, fingerprint, manifest_path
- Provenance: git hash, config hashes, data hash, timestamp
- Config snapshot: full extractor TOML + trainer YAML as dicts
- Results: training_metrics, backtest_metrics, dataset_health
- Metadata: tags, hypothesis, notes, status, stages_completed

**ExperimentLedger** — append-only JSON-backed storage:
- `register(record)` — store new record, update index
- `get(id)` — load full record
- `filter(tags=, model_type=, min_f1=, ...)` — query index
- `find_by_fingerprint(fp)` — dedup lookup
- `update_notes(id, notes)` — only mutable operation

**compute_fingerprint(manifest, paths)** — SHA-256 of outcome-affecting config (strips metadata like name, description, tags, output paths).

**compare_experiments(entries, metric_keys, sort_by, ...)** — sorted comparison table. `diff_experiments(a, b)` — config + metric diff between two records.

---

## 3. Data Flow

```
hft-ops run manifest.yaml
    │
    ├─1─ VALIDATE: parse → resolve vars → check contract → cross-validate configs → dedup check
    ├─2─ EXTRACT:  cargo run (skip if exists + metadata validates)
    ├─3─ ANALYZE RAW:  python run_analysis.py (optional)
    ├─4─ ANALYZE DATASET:  python run_analysis.py
    ├─5─ TRAIN:  resolve horizon_idx → apply overrides → python train.py
    ├─6─ BACKTEST:  python backtest_deeplob.py
    └─7─ RECORD:  build ExperimentRecord → compute fingerprint → write to ledger
```

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
| `hft-ops check-dup <manifest>` | Check for duplicates |
| `hft-ops sweep expand <manifest>` | Dry expansion showing grid points |
| `hft-ops sweep run <manifest>` | Execute all grid points (--continue-on-failure) |
| `hft-ops sweep results <sweep_id>` | Compare results from a sweep |

---

## 6. Testing

93 tests across 6 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_manifest_schema.py` | 18 | Schema defaults, variable resolution, YAML loading |
| `test_validator.py` | 14 | Feature count, window_size, horizon resolution, cross-module |
| `test_dedup.py` | 8 | Fingerprint determinism, dedup lookup |
| `test_ledger.py` | 18 | CRUD, filtering, persistence, notes update |
| `test_provenance.py` | 16 | Hashing, git info, provenance building |
| `test_sweep.py` | 19 | Sweep validation (8), grid expansion (11) |

```bash
.venv/bin/python -m pytest tests/ -v
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
