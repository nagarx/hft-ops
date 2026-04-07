# hft-ops

Central experiment orchestrator for the HFT pipeline.

Defines, validates, runs, tracks, and compares experiments across all 7 pipeline modules from a single YAML manifest. Replaces the manual 3+ step workflow (write extractor TOML, write trainer YAML, run backtest script) with a single command that validates cross-module consistency, tracks provenance, and prevents duplicate experiments.

## Installation

```bash
cd hft-ops
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -e ../hft-contracts/
```

## Quick Start

```bash
# Run a complete experiment
hft-ops run experiments/nvda_tlob_h100_tb_volscaled.yaml

# Validate without executing (dry run)
hft-ops run experiments/nvda_tlob_h100_tb_volscaled.yaml --dry-run

# Validate manifest only
hft-ops validate experiments/nvda_tlob_h100_tb_volscaled.yaml

# Compare experiments
hft-ops compare --metric training_metrics.macro_f1 --sort desc --top 10

# Diff two experiments
hft-ops diff <experiment_id_1> <experiment_id_2>

# Browse the ledger
hft-ops ledger list
hft-ops ledger show <experiment_id>
hft-ops ledger search --tags "nvda,tlob" --min-f1 0.35

# Check for duplicates before running
hft-ops check-dup experiments/nvda_tlob_h100_tb_volscaled.yaml
```

## Experiment Manifest

Every experiment is defined by a single YAML file in `experiments/`:

```yaml
experiment:
  name: "nvda_tlob_h100_tb_volscaled"
  hypothesis: "Vol-scaled barriers improve directional accuracy"
  contract_version: "2.2"
  tags: [nvda, tlob, h100]

stages:
  extraction:
    config: "feature-extractor-MBO-LOB/configs/nvda_tb_volscaled.toml"
    output_dir: "data/exports/nvda_tb_volscaled"
    skip_if_exists: true

  training:
    config: "lob-model-trainer/configs/experiments/nvda_tlob_h100.yaml"
    overrides:
      data.data_dir: "${stages.extraction.output_dir}"
    horizon_value: 100  # resolved to index at runtime

  backtesting:
    model_checkpoint: "${stages.training.output_dir}/checkpoints/best.pt"
    data_dir: "${stages.extraction.output_dir}"
```

Key design: the manifest **never duplicates** parameters. `feature_count`, `window_size`, and `stride` are validated against the extractor config, not re-specified.

## Architecture

```
hft-ops/
├── src/hft_ops/
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # Global OpsConfig
│   ├── paths.py            # Pipeline path resolution
│   ├── manifest/           # Manifest parsing, loading, validation
│   ├── stages/             # Stage runners (subprocess wrappers)
│   ├── ledger/             # Experiment records, storage, dedup, comparison
│   └── provenance/         # Git hash, config hash, lineage
├── experiments/            # Experiment manifest YAMLs
├── ledger/                 # Experiment records (append-only)
└── tests/                  # 93 tests
```

## Dependencies

- `hft-contracts` -- contract validation
- `click` -- CLI framework
- `pyyaml` -- manifest parsing
- `rich` -- terminal output

## Running Tests

```bash
cd hft-ops
.venv/bin/python -m pytest tests/ -v
```
