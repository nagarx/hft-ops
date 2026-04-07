# Experiment Guide

How to create, run, track, and compare experiments across the full HFT pipeline.

---

## Overview

Every experiment is defined by a single YAML manifest in `hft-ops/experiments/`. This manifest is the **single source of truth** for the entire pipeline configuration of that experiment: extraction, training, signal export, backtesting, and profiler references.

```
hft-ops/experiments/nvda_hmhp_40feat_xnas_h10.yaml
  ├── Extraction config → feature-extractor-MBO-LOB
  ├── Training config → lob-model-trainer
  ├── Signal export → lob-model-trainer/scripts/export_hmhp_signals.py
  ├── Backtesting → lob-backtester/scripts/run_readability_backtest.py
  └── Profiler references → mbo-statistical-profiler analysis
```

---

## Creating a New Experiment

1. Copy an existing manifest as a template:
   ```bash
   cp experiments/nvda_hmhp_40feat_xnas_h10.yaml experiments/my_new_experiment.yaml
   ```

2. Edit the manifest sections:
   - `experiment.name`: unique identifier
   - `experiment.description`: what you are testing
   - `experiment.hypothesis`: what you expect to find
   - `stages.extraction`: data source config
   - `stages.training`: model config, feature preset, horizons
   - `stages.backtesting`: readability gates, holding policy, costs

3. Validate the manifest:
   ```bash
   python scripts/validate_manifest.py experiments/my_new_experiment.yaml
   ```

---

## Running an Experiment

### Step 1: Extraction (if not already done)
```bash
cd feature-extractor-MBO-LOB
cargo run --release --features parallel --bin export_dataset -- \
    --config configs/nvda_xnas_128feat_full.toml
```

### Step 2: Training
```bash
cd lob-model-trainer
python3 -m lobtrainer.cli train configs/experiments/my_config.yaml \
    --manifest ../hft-ops/experiments/my_new_experiment.yaml
```

### Step 3: Signal Export
```bash
cd lob-model-trainer
python3 scripts/export_hmhp_signals.py \
    --experiment outputs/experiments/my_experiment \
    --split test
```

### Step 4: Backtesting
```bash
cd lob-backtester
python3 scripts/run_readability_backtest.py \
    --signals ../lob-model-trainer/outputs/experiments/my_experiment/signals/test/ \
    --name my_backtest_h10 \
    --exchange XNAS \
    --manifest ../hft-ops/experiments/my_new_experiment.yaml
```

---

## Tracking Experiments

### Dashboard
See all experiments and their pipeline status:
```bash
cd hft-ops
python3 scripts/experiment_dashboard.py
```

### Comparison
Compare experiments side-by-side:
```bash
python3 scripts/compare_experiments.py
python3 scripts/compare_experiments.py --format markdown
```

### Validation
Check manifest consistency against pipeline state:
```bash
python3 scripts/validate_manifest.py experiments/nvda_hmhp_40feat_xnas_h10.yaml
```

---

## Data Flow and Contracts

```
pipeline_contract.toml (Schema 2.3)
    ├── [features] → Feature indices for extractor + trainer
    ├── [labels] → Label encoding for extractor + trainer
    ├── [signals] → Signal format for trainer → backtester
    ├── [export.metadata] → Metadata fields for extractor output
    └── [normalization] → Which features to exclude

Extractor → {day}_sequences.npy + metadata.json
    → validated by hft_contracts.validate_export_contract()

Trainer → model checkpoint + training_history.json
    → registered in ExperimentRegistry

Signal Export → predictions.npy + agreement_ratio.npy + ...
    → format defined by [signals] contract

Backtester → BacktestResult + equity_curve.npy
    → registered in BacktestRegistry
```

---

## Manifest Format Reference

```yaml
experiment:
  name: "unique_experiment_id"
  description: "What this experiment tests"
  hypothesis: "What we expect to find"
  contract_version: "2.3"
  tags: [nvda, hmhp, ...]

pipeline_root: ".."

stages:
  extraction:
    enabled: true
    config: "path/to/extraction.toml"
    output_dir: "path/to/output"
    feature_count: 128
    labeling_strategy: tlob
    horizons: [10, 20, 50, 60, 100, 200, 300]

  training:
    enabled: true
    config: "path/to/training.yaml"
    model_type: hmhp
    feature_preset: short_term_40
    input_size: 40
    hmhp_horizons: [10, 60, 300]
    output_dir: "path/to/training/output"

  signal_export:
    enabled: true
    checkpoint: "${stages.training.output_dir}/checkpoints/best.pt"
    split: test
    output_dir: "${stages.training.output_dir}/signals/test"

  backtesting:
    enabled: true
    signals_dir: "${stages.signal_export.output_dir}"
    readability:
      min_agreement: 1.0
      min_confidence: 0.65
      max_spread_bps: 1.05
    holding:
      type: horizon_aligned
      hold_events: 10
    costs:
      exchange: XNAS

profiler_references:
  statistical_basis:
    ofi_signal_r_h10: 0.30
    vwes_bps: 1.97
```

---

## Current Experiments

| Manifest | Model | Features | Exchange | Status |
|---|---|---|---|---|
| nvda_hmhp_128feat_xnas_h10 | HMHP | 119 | XNAS | Trained + Analyzed |
| nvda_hmhp_40feat_xnas_h10 | HMHP | 40 | XNAS | Trained + Backtested |
| nvda_hmhp_128feat_arcx_h10 | HMHP | 119 | ARCX | Trained + Analyzed |
