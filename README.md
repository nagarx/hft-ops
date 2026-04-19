# hft-ops

Central experiment orchestrator for the HFT pipeline.

Defines, validates, runs, tracks, and compares experiments across all pipeline
modules from a single YAML manifest. Replaces the manual 3+-step workflow
(write extractor TOML, write trainer YAML, run backtest script) with a single
command that validates cross-module consistency, tracks provenance, enforces
pre-training signal-quality gates, detects post-training regressions, prevents
duplicate experiments, and aggregates sweep results.

## Installation

```bash
cd hft-ops
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -e ../hft-contracts/
```

## Quick Start

```bash
# Run a complete experiment end-to-end
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml

# Validate without executing (dry run)
hft-ops run experiments/e5_60s_huber_cvml_unified.yaml --dry-run

# Validate manifest only
hft-ops validate experiments/e5_60s_huber_cvml_unified.yaml

# Compare experiments
hft-ops compare --metric training_metrics.test_ic --sort desc --top 10

# Diff two experiments
hft-ops diff <experiment_id_1> <experiment_id_2>

# Browse the ledger
hft-ops ledger list
hft-ops ledger show <experiment_id>
hft-ops ledger search --tags "nvda,tlob" --min-ic 0.30
hft-ops ledger rebuild-index           # re-project records through the current index_entry whitelist
hft-ops ledger fingerprint-explain <manifest.yaml>  # show fingerprint inputs

# Check for duplicates before running
hft-ops check-dup experiments/e5_60s_huber_cvml_unified.yaml

# Sweep expansion + execution
hft-ops sweep expand experiments/sweeps/e5_phase2_sweep.yaml     # preview grid points
hft-ops sweep run experiments/sweeps/e5_phase2_sweep.yaml        # execute all points
hft-ops sweep results <sweep_id>                                 # per-sweep summary

# Feature-set registry (Phase 4 + Phase 7 Stage 7.1)
hft-ops feature-sets list
hft-ops feature-sets show nvda_short_term_40_src128_v1
hft-ops evaluate --config <evaluator.yaml> --criteria <criteria.yaml> \
    --save-feature-set <name>_v1 --applies-to-assets NVDA --applies-to-horizons 10
```

## Experiment Manifest

Every experiment is defined by a single YAML file in `experiments/`:

```yaml
experiment:
  name: "e5_60s_huber_cvml_unified"
  hypothesis: "Huber loss + 60s bars improve regression IC over 5s bars"
  contract_version: "2.2"
  tags: [nvda, tlob, regression, h10, 60s]

stages:
  extraction:
    enabled: true
    config: "feature-extractor-MBO-LOB/configs/e5_timebased_60s.toml"
    output_dir: "data/exports/e5_timebased_60s"
    skip_if_exists: true

  validation:                           # Phase 2b IC gate (pre-training)
    enabled: true
    on_fail: abort
    min_ic: 0.05
    min_ic_count: 2
    min_return_std_bps: 5.0
    min_stability: 2.0

  training:
    trainer_config:                     # Phase 1 wrapper-less inline config
      data:
        data_dir: "${stages.extraction.output_dir}"
        feature_set: "nvda_short_term_40_src128_v1"   # Phase 7.1 registry entry
      model:
        model_type: tlob
      # ... resolved at runtime
    horizon_value: 10                   # resolved to horizon_idx at runtime

  post_training_gate:                   # Phase 7 Stage 7.4 regression gate
    enabled: true
    on_regression: warn
    min_metric_floor: 0.05              # rule §13 minimum signal
    min_ratio_vs_prior_best: 0.9        # within 10% of best matching prior
    cost_breakeven_bps: 1.4             # IBKR Deep ITM 0DTE

  signal_export:
    enabled: true
    checkpoint: "${stages.training.output_dir}/checkpoints/best.pt"
    split: test

  backtesting:
    enabled: true
    signals_dir: "${stages.signal_export.output_dir}"
```

Key design: the manifest **never duplicates** parameters. `feature_count`,
`window_size`, and `stride` are validated against the extractor config, not
re-specified. Variable substitution (`${...}`) resolves cross-stage paths.

## Architecture

```
hft-ops/
├── src/hft_ops/
│   ├── cli.py                    # Click CLI entry point (all subcommands)
│   ├── config.py                 # Global OpsConfig
│   ├── paths.py                  # PipelinePaths (multi-repo path resolution)
│   ├── manifest/                 # Manifest parsing, loading, validation, resolution
│   ├── stages/                   # 8 stage runners (subprocess wrappers + gate runners):
│   │                             # extraction, raw_analysis, dataset_analysis,
│   │                             # validation (IC gate), training,
│   │                             # post_training_gate (regression detection),
│   │                             # signal_export, backtesting
│   ├── ledger/                   # Experiment records, storage, dedup fingerprint,
│   │                             # sweep aggregate writer
│   ├── feature_sets/             # Phase 4 FeatureSet producer + writer + registry
│   │                             # (schema + hashing co-moved to hft_contracts in Phase 6)
│   ├── provenance/               # Shim re-exporting hft_contracts.provenance
│   ├── sweep.py                  # Sweep expansion + cross-stage override routing
│   └── utils.py                  # Shared utilities
├── experiments/                  # Experiment manifest YAMLs
│   └── sweeps/                   # Sweep templates (loss_ablation, horizon_sensitivity, ...)
├── ledger/                       # Experiment records (append-only JSON)
└── tests/                        # 433 tests across 30 test files
```

## Phase History Highlights

- **Phase 0-3**: Structural fixes, wrapper-less manifests, `ValidationStage`,
  multi-base trainer config composition (`_base:`), Phase 3 §3.3b fingerprint
  normalization (resolved-dict hashing).
- **Phase 4 (2026-04-15)**: FeatureSet Registry — content-addressed artifacts
  at `contracts/feature_sets/<name>.json`; fingerprint normalizes
  `feature_set` / `feature_preset` → canonical `feature_indices` before hashing.
- **Phase 5 FULL-A (2026-04-17)**: Sweep-first infrastructure — cross-stage
  override routing, `SweepAggregateWriter`, 3 MVP templates.
- **Phase 6 (2026-04-17)**: Contract-plane consolidation. Five primitives
  co-moved to `hft_contracts`: `Provenance`, `SignalManifest`,
  `ExperimentRecord`, `FeatureSet` schema + hashing. 12 legacy scripts
  archived to `*/scripts/archive/` per `hft-rules §4`.
- **Phase 7 Stage 7.1 (2026-04-18)**: FeatureSet Registry activation — 3
  production JSONs (`nvda_short_term_40_src128_v1`,
  `nvda_short_term_40_src116_v1`, `nvda_analysis_ready_119_src128_v1`);
  8 manifest migrations from `feature_preset:` → `feature_set:` closing
  2026-08-15 `feature_preset` ImportError deadline.
- **Phase 7 Stage 7.4 Rounds 4-5 (2026-04-20)**: Post-Training Regression
  Detection Gate — `PostTrainingGateStage` + `PostTrainingGateRunner`
  between training and signal_export. Three checks (floor / prior-best-ratio
  / cost-breakeven). Generic `gate_reports: Dict[str, Dict]` on
  `ExperimentRecord` (Option C) plugs in future gates with zero schema
  change. Atomic writes unified at `hft_contracts.atomic_io` SSoT (REV 2
  public rename, 2026-04-20 — `hft_contracts._atomic_io` is a
  `DeprecationWarning` shim retained through 2026-10-31).

## Dependencies

- `hft-contracts` — contract validation + canonical dataclasses
- `click>=8.0` — CLI framework
- `pyyaml>=6.0` — manifest parsing
- `rich>=13.0` — terminal output
- `tomli>=2.0; python_version < '3.11'` — TOML parsing pre-3.11

## Running Tests

```bash
cd hft-ops
.venv/bin/python -m pytest tests/ -v
# Expected: 433 passed (post Phase 7 Stage 7.4 Round 6)
```

## Related Documentation

- `CODEBASE.md` — detailed module reference, CLI commands, testing.
- `EXPERIMENT_GUIDE.md` — 10-step end-to-end walkthrough.
- Pipeline-wide ground-truth docs (at monorepo root):
  `CLAUDE.md`, `PIPELINE_ARCHITECTURE.md`, `DOCUMENTATION_INDEX.md`,
  `PHASE7_ROADMAP.md`, `contracts/pipeline_contract.toml`.
