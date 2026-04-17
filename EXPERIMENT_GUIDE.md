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
pipeline_contract.toml (Schema 2.2)
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
  contract_version: "2.2"
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

---

## Running a Sweep (Phase 5 Preview + Phase 5 FULL-A, 2026-04-17)

Sweeps are parameter-grid experiments authored as a single YAML manifest under
`hft-ops/experiments/sweeps/`. Phase 5 FULL-A adds cross-stage axis routing so
a sweep can vary extraction bin size × training horizon × backtest spread in
one manifest, rather than copy-pasting N configs.

### Authoring pattern

```yaml
experiment:
  name: "my_sweep"
  hypothesis: "Required per hft-rules §13 — state what you expect and why."
  contract_version: "2.2"

stages:
  # Base pipeline — fields here become defaults for every grid point.
  extraction: {enabled: true, config: "...", output_dir: "..."}
  training:
    enabled: true
    trainer_config: {...}       # inline base config (overridden per axis)
  signal_export: {...}
  backtesting: {...}

sweep:
  name: "my_sweep"
  strategy: "grid"              # Only "grid" implemented; zip/conditional/bayesian reserved
  axes:
    - name: "axis_a"
      values:
        - label: "val1"
          overrides:
            model.tlob_use_cvml: false        # Bare model.*/train.*/data.* → training.overrides
            extraction.config: "alt.toml"     # Stage-prefixed → stage dataclass
            backtesting.params.spread_bps: 2.0  # Nested dataclass walk
```

**Override-routing rules** (enforced by `expand_sweep_with_axis_values` with
HARD-FAIL on violations):

| Axis override key | Routes to | Example |
|---|---|---|
| `model.X`, `train.X`, `data.X` (bare) | `training.overrides[<key>]` (trainer YAML) | `model.tlob_use_cvml: true` |
| `horizon_value`, `output_dir`, `config` (bare, legacy) | `TrainingStage` dataclass field | `horizon_value: 10` |
| `training.overrides.<key>` (explicit) | `training.overrides[<key>]` | `training.overrides.model.dropout: 0.1` |
| `<stage>.<field>` | `stages.<stage>.<field>` | `extraction.config: "alt.toml"` |
| `<stage>.<nested>.<field>` | `stages.<stage>.<nested>.<field>` (walks into nested dataclass) | `backtesting.params.spread_bps: 2.0` |

Unknown stage prefixes or unknown fields on a known stage HARD-FAIL with a
message listing valid names.

### Post-expansion variable resolution (Phase 5 FULL-A Block 4)

Each grid point re-resolves `${...}` after axis overrides are applied. Useful
for per-point paths:

```yaml
extraction:
  output_dir: "data/exports/${sweep.point_name}"   # per-grid-point distinct path

training:
  trainer_config:
    data:
      data_dir: "${stages.extraction.output_dir}"   # rebinds to per-point value
```

`${sweep.point_name}` is composed from the axis labels (`{sweep.name}__{label1}_{label2}`).
`${sweep.axis_values.<axis_name>}` also available.

### Sweep-aggregate ledger record

Every sweep run writes a **parent** `record_type="sweep_aggregate"` record
summarizing the entire grid alongside the per-grid-point records. Identity:
`experiment_id = "{sweep_id}_aggregate"` (deterministic, overwrite on re-run).
Fingerprint: `sha256(canonical(sorted(child_fingerprints), sweep_name))` —
content-addressed, so two sweep invocations with equivalent content produce
equal aggregate fingerprints.

**hft-rules §13 per-experiment-documentation policy for sweeps**: one
`EXPERIMENT_INDEX.md` entry per sweep at the aggregate level. The
`sub_records` field in the aggregate carries per-grid-point summaries; drill
down into individual grid-point records as needed.

### CLI reference

```bash
hft-ops sweep expand  experiments/sweeps/<name>.yaml   # preview grid (dry)
hft-ops sweep run     experiments/sweeps/<name>.yaml   # execute all points
hft-ops sweep results <sweep_id>                        # compare grid points
```

`sweep_results` and `compare` CLI tables exclude the sweep-aggregate parent
record (filtered by `record_type != "sweep_aggregate"`). Aggregates are
ledger-visible for future dashboards that want to opt in.

### Shipped templates (current)

| Template | Axes | Points | Exercises |
|---|---|---|---|
| `e5_phase2_sweep.yaml` | `cvml × loss_delta` | 4 | Preview — same-stage training axes (back-compat path). |
| `loss_ablation.yaml` | `loss_type × delta` | 6 | Same-stage training axes (Huber vs MSE). |
| `horizon_sensitivity.yaml` | `bin_size × horizon_value` | 9 | **Cross-stage** — primary validator of Block 1 routing + Block 4 variable rebinding. |
| `backtest_cost_sensitivity.yaml` | `spread × threshold` | 9 | **Nested-dataclass** — `backtesting.params.*` axes. |

Backlog (additive, author as needed): `seed_stability`, `feature_set_ablation`,
`model_family`. See `experiments/sweeps/README.md`.
