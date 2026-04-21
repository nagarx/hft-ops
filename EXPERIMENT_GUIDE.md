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

## Sweep Compare — Pairwise-Bootstrap Significance (Phase V.B.4b, 2026-04-21)

Answer the research question **"is treatment A significantly better than
treatment B?"** for any sweep whose child records share test-split labels.

### When to use

Every multi-treatment sweep run (e.g., `seed_stability.yaml`,
`loss_ablation.yaml`, any `model_family_*.yaml` template). The tool
consumes each child record's paired `(regression_labels.npy,
predicted_returns.npy)` signal files, verifies byte-identical paired
structure via SHA-256, and invokes the hft-metrics paired
moving-block-bootstrap primitive (v0.1.7) for `K*(K-1)/2` pairwise
comparisons.

Output columns: observed statistics + delta + percentile CI +
two-sided ASL p-value + BH-corrected q-value + `n_nonfinite_replaced`
diagnostic. Rows sorted by BH q-value ascending (most-significant first).

### Basic invocation

```bash
hft-ops sweep compare <sweep_id> --metric val_ic --alpha 0.05
```

Example for a 5-seed stability sweep:
```bash
hft-ops sweep compare seed_stability_20260421 --metric val_ic --alpha 0.05
```

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--metric` | `val_ic` | Statistic to compare. MVP supports `val_ic` (Spearman IC). Classification metrics + additional statistics land in a follow-up. |
| `--alpha` | `0.05` | Significance level for CI + BH FDR correction. 0.05 = 95% CI. |
| `--n-bootstraps` | `10000` | Bootstrap iterations. Lower = faster + wider CI. |
| `--block-length` | auto | Moving-block bootstrap block length. Default: `ceil(n^(1/3))` per Politis-Romano 1994. |
| `--seed` | `42` | Random seed for reproducibility. Same seed → bit-identical output. |

### When `sweep compare` FAILS LOUD (per hft-rules §8)

- **Unpaired labels**: child records have byte-different `regression_labels.npy`
  (e.g., `feature_set_ablation` sweeps where FeatureSet changes →
  sample set differs). Error cites which two records diverged. Phase VI
  unpaired (Welch-style) tooling deferred.
- **NaN-row drop > 5%** (V.1 L1.3): above the `max_drop_frac`
  threshold, abort with per-record breakdown (`exp_id` +
  `x_nonfinite` + `y_nonfinite` counts). Operator can override with
  the caller-side `max_drop_frac=1.0` (documented as "bypasses
  hft-rules §8").
- **Duplicate treatment labels** (V.1 L3.2): two records produce the
  same axis_values label — ambiguous output rows. Sweep manifest has
  a collision; fix the YAML.
- **Primary_horizon mismatch**: records use different horizons (a
  horizon-sensitivity sweep is implicitly unpaired).
- **Non-finite observed statistic** (v0.1.7 L2.7): `statistic_fn`
  returns NaN on original data — upstream data-quality issue.

### Interpretation guide

- **BH q-value ≤ alpha AND CI excludes 0**: green ★ marker — pair is
  significant after FDR correction.
- **`n_nonfinite_replaced` high** (yellow column): substantial
  bootstrap-iter × treatment samples hit the conservative fallback to
  the observed statistic. Large values (> 10% of `2 × n_bootstraps`)
  mean the CI is substantially point-mass from fallback, NOT real
  bootstrap variance — treat p-values with skepticism.
- **NaN-dropped > 0 in summary**: some rows of paired (x, Y) had
  non-finite values and were dropped preserving pairing. Under 5% is
  the default tolerance; the summary panel reports the exact count
  and fraction.

### Research basis

- Paired bootstrap p-value: Efron & Tibshirani (1993), *An Introduction
  to the Bootstrap*, Ch 15 eq 15.22.
- Moving-block bootstrap: Künsch (1989), *Annals of Statistics*
  17(3):1217-1241.
- Default block_length: Politis & Romano (1994), *JASA* 89(428):1303-1313.
- FDR correction: Benjamini & Hochberg (1995), *JRSS-B* 57(1):289-300 §3.

See `hft-metrics/CODEBASE.md` §14c for the primitive-level reference.

---

## CompatibilityContract Fingerprint Filter (Phase V.A.4, 2026-04-21)

Every experiment run through `hft-ops run` post-V.A.4 carries a
64-hex SHA-256 `compatibility_fingerprint` computed over 11
shape-determining signal-boundary fields (`contract_version`,
`feature_count`, `window_size`, `feature_layout`, `data_source`,
`label_strategy_hash`, `calibration_method`, `primary_horizon_idx`,
`horizons`, `normalization_strategy`, `schema_version`). Query:

```bash
# Surface every experiment produced against a specific contract version
hft-ops ledger list --compatibility-fp <64-hex>

# Pre-V.A.4 records project empty string; to find those:
hft-ops ledger list --compatibility-fp ""
```

The fingerprint is **NOT** a fingerprint for dedup (treatments only go
into `compute_fingerprint`). It's an OBSERVATION surfaced as a
ledger-query trust column. See root CLAUDE.md §"Module Technical Map"
for the V.A.4 design rationale.

---

## Feature Importance (Phase 8C-α Stage C.1, 2026-04-20)

Post-training permutation importance quantifies each feature's contribution
to the trained model's predictive metric on a held-out eval split. The
trainer's `PermutationImportanceCallback` invokes at `on_train_end`, writes
`outputs/<experiment>/feature_importance_v1.json`, and the artifact is then
auto-routed by hft-ops Stage C.3 into
`hft-ops/ledger/feature_importance/{yyyy_mm}/<sha>.json` with a reference
on `ExperimentRecord.artifacts[]`.

### When to enable

- **After a successful experiment** — use an importance audit to rank which
  features actually drove the predictive metric (permuting a "real"
  predictor collapses the metric; permuting a noise feature barely moves
  it).
- **Before promoting a feature to STRONG-KEEP** — Phase 8C-β Stage C.5
  feedback-merge (deferred) will flip evaluator tiers based on K≥5
  importance artifacts per feature_set. To be feedback-merge-eligible,
  use a registered FeatureSet (`data.feature_set: <name>_v1`); otherwise
  the artifact emits a WARN and is dead-end for merge.

### How to enable (minimal YAML)

Add the `importance:` block to the trainer's `ExperimentConfig` — either
directly in a trainer YAML, or via the base fragment:

```yaml
# lob-model-trainer/configs/experiments/<your_experiment>.yaml
_base:
  - models/tlob_compact_regression.yaml
  - datasets/nvda_e5_60s.yaml
  - labels/regression_huber.yaml
  - train/regression_default.yaml
  - train/importance_default.yaml   # <-- enables post-training importance
name: my_experiment_with_importance
```

Or inline the block in an hft-ops unified manifest:

```yaml
# hft-ops/experiments/<your_experiment>.yaml
stages:
  training:
    trainer_config:
      # ... existing config ...
      importance:
        enabled: true
        n_permutations: 100       # baseline CI; bump to 500 for tight CI
        n_seeds: 3                # bump to 5 for Stage C.5 feedback-merge
        subsample: 5000           # eval-split subsample; -1 = full
        block_length_samples: 1   # 1 = element-wise; N = day-preserving
        seed: 42
        eval_split: "test"
        baseline_metric: "auto"   # auto = IC (regression) / accuracy (classification)
```

See `hft-ops/experiments/e5_60s_importance_audit.yaml` for a complete
working example replicating the E5 H10 training path with importance
enabled.

### What it produces

- **`outputs/<experiment>/feature_importance_v1.json`** — trainer-side
  artifact. Schema: `hft_contracts.FeatureImportanceArtifact v2`.
- **`hft-ops/ledger/feature_importance/{yyyy_mm}/<sha>.json`** — content-
  addressed ledger copy.
- **`ExperimentRecord.artifacts[]` entry** — `{kind: "feature_importance",
  path, sha256, bytes, method}` — surfaced via
  `index_entry()::artifact_kinds` for fast ledger queries.

Inspect via:
```bash
# ledger list, filter by artifact kind (once Stage C.4 query layer lands)
hft-ops ledger list | grep feature_importance

# Manual inspection
python3 -c "
from hft_contracts import FeatureImportanceArtifact
a = FeatureImportanceArtifact.load('outputs/<exp>/feature_importance_v1.json')
for f in sorted(a.features, key=lambda f: -f.importance_mean)[:10]:
    print(f'{f.feature_name:30s} importance={f.importance_mean:+.4f} ± {f.importance_std:.4f}  stability={f.stability:.2f}')
"
```

### Compute cost guidance

Compute: `n_permutations × n_seeds × n_features` model forward passes on
the subsample. For a 98-feature × T=20 TLOB-H10 model:

| Configuration | Approx compute | When |
|---|---|---|
| Defaults (100 × 3 × 5000) | 15-30 min CPU | Default audit |
| Tight CI (500 × 5 × 5000) | 2-4 hours CPU | Pre-Stage-C.5 feedback-merge input |
| Full eval (subsample=-1, 500 × 5) | 10-20 hours CPU | Publication-grade importance ranking |

Importance runs AFTER training completes — a failure in the callback
logs + swallows (training checkpoint is preserved; only the importance
artifact is missing). This is deliberate — importance is an observation,
not a treatment, and losing it should not invalidate a successful
training run.

### Deferred capabilities (Phase 8C-β / 8D)

- `hft-ops feedback-status --feature-set <name>` CLI — K-count readiness
  reporting. (Stage C.6.)
- `hft-ops ledger/query.py::find_feedback_artifacts(content_hash, K)` —
  query layer for Stage C.5 consumer. (Stage C.4.)
- Evaluator `merge_feedback_into_profiles()` — flips
  `FeatureProfile.reconciled_tier` based on K-seed aggregated importance.
  (Stage C.5.)
- SHAP / Integrated-Gradients / attention-map importance — separate
  `artifact_kinds` with their own schemas. (Phase 8D+.)

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

  validation:                                  # Phase 2b IC gate (pre-training)
    enabled: true                              # mandatory per hft-rules §13
    on_fail: abort                             # warn | abort | record_only
    min_ic: 0.05
    min_ic_count: 2
    min_return_std_bps: 5.0
    min_stability: 2.0

  training:
    enabled: true
    config: "path/to/training.yaml"
    model_type: hmhp
    feature_set: nvda_short_term_40_src128_v1  # Phase 7.1 FeatureSet registry entry
                                               # (was `feature_preset: short_term_40`
                                               # pre-2026-04-18; deprecated 2026-04-15,
                                               # ImportError deadline 2026-08-15)
    input_size: 40
    hmhp_horizons: [10, 60, 300]
    output_dir: "path/to/training/output"

  post_training_gate:                          # Phase 7 Stage 7.4 regression detector
    enabled: true                              # opt-in; default False
    on_regression: warn                        # warn | abort | record_only
    min_metric_floor: 0.05                     # rule §13 minimum signal
    min_ratio_vs_prior_best: 0.9               # within 10% of best matching prior
    cost_breakeven_bps: 1.4                    # IBKR Deep ITM 0DTE (informational)

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

Single-run manifests under `hft-ops/experiments/`:

| Manifest | Model | Features | Exchange | FeatureSet (Phase 7.1) | Status |
|---|---|---|---|---|---|
| nvda_hmhp_128feat_xnas_h10 | HMHP | 119 | XNAS | nvda_analysis_ready_119_src128_v1 | Trained + Analyzed |
| nvda_hmhp_40feat_xnas_h10 | HMHP | 40 | XNAS | nvda_short_term_40_src128_v1 | Trained + Backtested |
| nvda_hmhp_128feat_arcx_h10 | HMHP | 119 | ARCX | nvda_analysis_ready_119_src128_v1 | Trained + Analyzed |
| e5_60s_huber_cvml_unified | TLOB | 98 | XNAS | — (inline feature_indices) | Regression E5 reference |
| nvda_tlob_h100_tb_volscaled | TLOB | 128 | XNAS | — (inline) | Classification baseline |
| nvda_tlob_h10_v1 | TLOB | 128 | XNAS | — (inline) | Classification baseline |

Sweep templates under `hft-ops/experiments/sweeps/`:

| Sweep | Grid Points | Axes | Scope |
|---|---|---|---|
| loss_ablation | 6 | training.loss_type × training.delta | Same-stage |
| horizon_sensitivity | 9 | training.horizon × backtesting.hold_events | Cross-stage |
| backtest_cost_sensitivity | 9 | backtesting.costs.* (nested) | Backtest-only |
| e5_phase2_sweep | 4 | CVML on/off × Huber δ | Same-stage |

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
