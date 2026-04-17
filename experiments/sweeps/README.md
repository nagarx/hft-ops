# hft-ops Sweep Manifests

Grid-search manifests for Phase 5 sweep-first workflows. Each file here is a
manifest with `sweep.axes` populated — running it expands to N concrete
experiments (Cartesian product of axis values) and ledgers each one with its
own fingerprint.

## Running a Sweep

```bash
cd hft-ops
hft-ops sweep expand  experiments/sweeps/<name>.yaml     # preview grid (dry)
hft-ops sweep run     experiments/sweeps/<name>.yaml     # execute all points
hft-ops sweep results experiments/sweeps/<name>.yaml     # compare records
```

## Authoring Rules

1. **Training-only axes (default)** — Put override keys under trainer-config paths
   (`model.*`, `train.*`, `data.*`) whenever possible. These merge into
   `stages.training.overrides` and are applied by the trainer YAML materializer.
2. **Label regex** — Axis-value labels must match `[a-zA-Z0-9_-]+` (no dots,
   no spaces). Use `d12_6` not `12.6`.
3. **No cross-axis override-key conflicts** — Each dotted key may appear on
   exactly one axis. `validate_sweep` fails otherwise.
4. **axis_values are piped through, not re-derived** — as of Phase 5 Preview
   (2026-04-16), `expand_sweep_with_axis_values` returns the ground-truth
   `{axis_name: selected_label}` dict; the ledger reads it directly rather
   than heuristically matching overrides.
5. **Fingerprint normalization is automatic** — Phase 4 Batch 4c.3 ensures
   `feature_set: <name>` and the equivalent `feature_indices: [...]`
   fingerprint identically, so sweeps that switch between the two
   representations will correctly deduplicate in the ledger.

## Current Templates

| File | Axes | Points | Purpose |
|------|------|--------|---------|
| `e5_phase2_sweep.yaml` | `cvml × loss_delta` | 4 | Reproduces E5 Phase 2 ablation (CVML ON/OFF × Huber δ ∈ {12.6, 15.1}). Preview-era — training-only axes. |
| `loss_ablation.yaml` | `loss_type × delta` | 6 | Huber vs MSE × δ ∈ {12.6, 15.1, 20.0} on E5 60s training. Single-stage (training). Phase 5 FULL-A. |
| `horizon_sensitivity.yaml` | `bin_size × horizon_value` | 9 | 30s/60s/120s extraction bins × H10/H60/H300 horizons. **CROSS-STAGE** — primary validator of Block 1 routing + Block 4 variable rebinding. Phase 5 FULL-A. |
| `backtest_cost_sensitivity.yaml` | `spread × threshold` | 9 | Fixed training + signal export; backtest cost parameters vary via nested-dataclass `backtesting.params.*` axes. **NESTED-DATACLASS** routing validator. Phase 5 FULL-A. |

## Backlog (additive — author as needed)

Templates documented but not yet shipped in Phase 5 FULL-A. Author additively in
Tier 2 (researcher demand-driven). Note: `seed_stability` is specifically
flagged by `feedback_oos_validation_first.md` as important — prioritize this
one for early authoring.

| File (proposed) | Axes | Rationale |
|---|---|---|
| `seed_stability.yaml` | `train.seed` ∈ 5 values | Surfaces per-seed variance; counter to the one-seed-result anti-pattern. |
| `feature_set_ablation.yaml` | `data.feature_set × validation.min_ic` | Cross-stage. Compare Phase-4 registry FeatureSet choices + gate threshold sensitivity. |
| `model_family.yaml` | `model.model_type × data.feature_set` | Replaces hand-rolled `e4_baselines.py` / `e5_baselines.py` loops. |

## What Sweeps Replace

Legacy ad-hoc scripts (`scripts/e5_baselines.py`, `scripts/e4_baselines.py`,
`scripts/run_simple_model_ablation.py`, etc.) hand-rolled subprocess loops to
run parameter sweeps. Those scripts are archived under `archive/pre_hft_ops_scripts/`
and the per-axis logic moves here. The trainer never needs to be invoked
manually to vary a hyperparameter; author one sweep manifest instead.
